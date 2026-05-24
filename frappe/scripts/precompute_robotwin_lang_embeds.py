import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml
from tqdm import tqdm

from models.multimodal_encoder.t5_encoder import T5Embedder


def load_config(path):
    with open(path, "r") as fp:
        return yaml.safe_load(fp)


def collect_instruction_texts(data_path, desc_types):
    texts = []
    source_roots = set()
    for hdf5_path in Path(data_path).rglob("*.hdf5"):
        try:
            import h5py
            with h5py.File(hdf5_path, "r") as f:
                if "source_dataset_root" in f.attrs:
                    source_roots.add(str(f.attrs["source_dataset_root"]))
        except Exception:
            continue

    candidate_dirs = [Path(data_path) / "instructions"]
    candidate_dirs.extend(Path(root) / "instructions" for root in sorted(source_roots))

    for instruction_dir in candidate_dirs:
        if not instruction_dir.is_dir():
            continue
        for json_path in sorted(instruction_dir.glob("*.json")):
            with open(json_path, "r") as fp:
                payload = json.load(fp)
            if not isinstance(payload, dict):
                continue
            for desc_type in desc_types:
                values = payload.get(desc_type, [])
                if isinstance(values, list):
                    texts.extend(v for v in values if isinstance(v, str) and v.strip())

    # Preserve order while deduplicating.
    return list(dict.fromkeys(texts))


def main():
    parser = argparse.ArgumentParser(description="Precompute T5 language embeddings for RoboTwin instruction JSON files.")
    parser.add_argument("--model_config_path", required=True)
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--target_dir", default=None)
    parser.add_argument("--text_encoder", default="google/t5-v1_1-xxl")
    parser.add_argument("--config_path", default="configs/base.yaml")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--desc_types", default="seen")
    parser.add_argument("--offload_dir", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    model_config = load_config(args.model_config_path)
    base_config = load_config(args.config_path)
    data_path = args.data_path or model_config["data_path"]
    target_dir = Path(args.target_dir or Path(data_path) / "instructions")
    target_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = target_dir / "manifest.json"
    existing = list(target_dir.glob("lang_embed_*.pt"))
    if existing and manifest_path.exists() and not args.force:
        print(f"Found {len(existing)} existing embeddings in {target_dir}; use --force to recompute.")
        return

    desc_types = [item.strip() for item in args.desc_types.split(",") if item.strip()]
    instructions = collect_instruction_texts(data_path, desc_types)
    if not instructions:
        task_name = Path(data_path).name.split("-", 1)[0].replace("_", " ")
        instructions = [task_name]

    if args.force:
        for pt_path in existing:
            pt_path.unlink()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    text_embedder = T5Embedder(
        from_pretrained=args.text_encoder,
        model_max_length=base_config["dataset"]["tokenizer_max_length"],
        device=device,
        use_offload_folder=args.offload_dir,
    )
    tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model
    text_encoder.eval()

    manifest = []
    for start in tqdm(range(0, len(instructions), args.batch_size), desc="encoding instructions"):
        batch = instructions[start:start + args.batch_size]
        tokenized = tokenizer(batch, return_tensors="pt", padding="longest", truncation=True)
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)
        with torch.no_grad():
            embeddings = text_encoder(input_ids=input_ids, attention_mask=attention_mask)["last_hidden_state"].detach().cpu()
        masks = attention_mask.cpu().bool()
        for offset, instruction in enumerate(batch):
            index = start + offset
            embedding = embeddings[offset][masks[offset]]
            save_path = target_dir / f"lang_embed_{index}.pt"
            torch.save(embedding, save_path)
            manifest.append({"file": save_path.name, "instruction": instruction})

    with open(manifest_path, "w") as fp:
        json.dump({"count": len(manifest), "items": manifest}, fp, indent=2)
    print(f"Saved {len(manifest)} embeddings to {target_dir}")


if __name__ == "__main__":
    main()
