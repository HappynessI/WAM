#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
from pathlib import Path

# get current workspace
current_file = Path(__file__)

import json
import sys

parent_dir = current_file.parent
sys.path.append(str(parent_dir))

import os

import argparse

import threading
import time
import yaml
from collections import deque

import numpy as np
import torch
from PIL import Image as PImage
import cv2

import sys, os

# get current workspace
current_file = Path(__file__)
sys.path.append(os.path.join(current_file.parent, "models"))

from scripts.agilex_model import create_model
from multimodal_encoder.t5_encoder import T5Embedder

global_path = parent_dir.parent
DEFAULT_ENCODER_WEIGHTS_ROOT = os.environ.get("FRAPPE_RDT_WEIGHTS_ROOT", "/data1/weights/RDT")


class RDT:

    def __init__(
        self,
        pretrained_model_name_or_path,
        task_name,
        left_arm_dim,
        right_arm_dim,
        rdt_step,
        ema_model_path=None,
        encoder_weights_root=None,
    ):
        # set path
        current_file = Path(__file__)
        self.global_path = Path(os.environ.get("FRAPPE_POLICY_ROOT", current_file.parent.parent)).resolve()
        self.encoder_weights_root = self._resolve_encoder_weights_root(encoder_weights_root)
        runtime_config = self._load_runtime_config(pretrained_model_name_or_path)
        img_history_size = self._infer_img_history_size(runtime_config)
        pred_horizon = self._infer_pred_horizon(runtime_config)
        # load the config
        self.config = {
            "episode_len": 10000,  # args.max_publish_step
            "state_dim": left_arm_dim + 1 + right_arm_dim +
            1,  # 14 dims action:[left joint angles,left gripper,right joint angles,right gripper]
            "chunk_size": pred_horizon,
            "img_history_size": img_history_size,
            "camera_names": ["cam_high", "cam_right_wrist", "cam_left_wrist"],
        }
        # setup config
        self.args = {
            "max_publish_step": 10000,  # Maximum number of action publishing steps
            "seed": None,  # Random seed
            "ctrl_freq": 25,  # The control frequency of the robot
            "chunk_size": pred_horizon,  # Action chunk size
            # 'disable_puppet_arm': False,  # Whether to disable the puppet arm
            "config_path": os.path.join(self.global_path, "frappe/configs/robotwin_contact_base.yaml"),
            "pretrained_model_name_or_path": pretrained_model_name_or_path,
            "ema_model_path": ema_model_path,
        }

        # Load rdt model
        self.left_arm_dim, self.right_arm_dim = left_arm_dim, right_arm_dim
        self.policy = self.make_policy(self.args)
        self.max_publish_step = self.config["episode_len"]
        self.chunk_size = self.config["chunk_size"]
        self.img_history_size = self.config["img_history_size"]
        self.task_name = task_name
        self.observation_window = None
        self.img_size = (640, 480)
        self.set_language_embed()
        self.rdt_step = rdt_step

    def _load_runtime_config(self, pretrained_model_name_or_path):
        config_path = None
        if pretrained_model_name_or_path is not None and os.path.isdir(pretrained_model_name_or_path):
            candidate = os.path.join(pretrained_model_name_or_path, "config.json")
            if os.path.isfile(candidate):
                config_path = candidate
        if config_path is None:
            candidate = os.path.join(self.global_path, "frappe/configs/robotwin_contact_base.yaml")
            if os.path.isfile(candidate):
                with open(candidate, "r") as fp:
                    return yaml.safe_load(fp)
            return {}

        with open(config_path, "r") as fp:
            return json.load(fp)

    @staticmethod
    def _infer_img_history_size(runtime_config):
        if "common" in runtime_config:
            return int(runtime_config["common"].get("img_history_size", 1))

        img_pos_embed_config = runtime_config.get("img_pos_embed_config", [])
        if img_pos_embed_config:
            _, shape = img_pos_embed_config[0]
            if isinstance(shape, (list, tuple)) and len(shape) >= 1:
                return int(shape[0])
        return 1

    @staticmethod
    def _infer_pred_horizon(runtime_config):
        if "common" in runtime_config:
            return int(runtime_config["common"].get("action_chunk_size", 64))
        return int(runtime_config.get("pred_horizon", 64))

    def _resolve_encoder_weights_root(self, encoder_weights_root):
        candidates = [
            encoder_weights_root,
            os.environ.get("RDT_WEIGHTS_ROOT"),
            os.environ.get("FRAPPE_RDT_WEIGHTS_ROOT"),
            DEFAULT_ENCODER_WEIGHTS_ROOT,
            os.path.join(self.global_path, "skele-wam/weights"),
        ]
        for candidate in candidates:
            if candidate and os.path.isdir(candidate):
                print(f"Using encoder weights from: {candidate}")
                return candidate
        raise FileNotFoundError(
            "Could not find encoder weights root. Checked: "
            + ", ".join(str(candidate) for candidate in candidates if candidate)
        )

    def _require_encoder_path(self, name):
        path = os.path.join(self.encoder_weights_root, name)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Missing encoder weights at: {path}")
        return path

    # set img_size
    def set_img_size(self, img_size):
        self.img_size = img_size

    def set_language_embed(self):
        MODEL_PATH = self._require_encoder_path("t5-v1_1-xxl")
        CONFIG_PATH = os.path.join(self.global_path, "frappe/configs/robotwin_contact_base.yaml")
        with open(CONFIG_PATH, "r") as fp:
            config = yaml.safe_load(fp)
        device = torch.device(self.policy.device)
        text_embedder = T5Embedder(
            from_pretrained=MODEL_PATH,
            model_max_length=config["dataset"]["tokenizer_max_length"],
            device=device,
            use_offload_folder=None,
        )
        self.tokenizer, self.text_encoder = text_embedder.tokenizer, text_embedder.model
        self.text_encoder.eval()

    # set language randomly
    def random_set_language(self, instruction=None):
        assert instruction is not None, "Missing input instruction"
        self.set_language_instruction(instruction)

    # encoding language
    def set_language_instruction(self, language_instruction, save_dir=None, task_name=None):
        assert ((save_dir is None) ^ (task_name is None)) == False, "input error"

        if os.path.isfile(language_instruction):
            lang_dict = torch.load(language_instruction)
            print(f"Running with instruction: \"{lang_dict['instruction']}\" from \"{lang_dict['name']}\"")
            self.lang_embeddings = lang_dict["embeddings"]
            print("loading instruction from pre-embed path")
        else:
            device = next(self.text_encoder.parameters()).device
            with torch.no_grad():
                tokens = self.tokenizer(
                    language_instruction,
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                )["input_ids"].to(device)
                tokens = tokens.view(1, -1)
                output = self.text_encoder(tokens)
                pred = output.last_hidden_state.detach().cpu()

            if save_dir is not None:
                save_path = os.path.join(save_dir, f"{task_name}.pt")
                torch.save({
                    "name": task_name,
                    "instruction": language_instruction,
                    "embeddings": pred,
                }, save_path)

            del tokens, output
            torch.cuda.empty_cache()
            self.lang_embeddings = pred

        print(f"successfully set instruction: {language_instruction}")

    # Update the observation window buffer
    def update_observation_window(self, img_arr, state):
        # JPEG transformation
        # Align with training
        def jpeg_mapping(img):
            if img is None:
                return None
            img = cv2.imencode(".jpg", img)[1].tobytes()
            img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
            return img

        def resize_img(img, size):
            return cv2.resize(img, size)

        if self.observation_window is None:
            self.observation_window = deque(maxlen=self.img_history_size)

        img_front, img_right, img_left, puppet_arm = (
            img_arr[0],
            img_arr[1],
            img_arr[2],
            state,
        )
        # img resize
        img_front = resize_img(img_front, self.img_size)
        img_left = resize_img(img_left, self.img_size)
        img_right = resize_img(img_right, self.img_size)
        # img jprg encoding
        img_front = jpeg_mapping(img_front)
        img_left = jpeg_mapping(img_left)
        img_right = jpeg_mapping(img_right)

        qpos = np.array(puppet_arm)
        qpos = torch.from_numpy(qpos).float().to(torch.device(self.policy.device))
        self.observation_window.append({
            "qpos": qpos,
            "images": {
                self.config["camera_names"][0]: img_front,
                self.config["camera_names"][1]: img_right,
                self.config["camera_names"][2]: img_left,
            },
        })

    def get_action(self, img_arr=None, state=None):
        assert (img_arr is None) ^ (state is None) == False, "input error"
        if (img_arr is not None) and (state is not None):
            self.update_observation_window(img_arr, state)

        with torch.inference_mode():
            action_buffer = inference_fn(
                self.config,
                self.policy,
                self.lang_embeddings,
                self.observation_window,
            ).copy()

        return action_buffer

    def reset_obsrvationwindows(self):
        self.lang_embeddings = None
        self.observation_window = None
        print("successfully unset obs and language intruction")

    # Initialize the model
    def make_policy(self, args):
        with open(args["config_path"], "r") as fp:
            config_base_yaml = yaml.safe_load(fp)
        args["config"] = config_base_yaml
        args["config"]["arm_dim"] = {
            "left_arm_dim": self.left_arm_dim,
            "right_arm_dim": self.right_arm_dim,
        }
        # pretrained_text_encoder_name_or_path = "weights/RDT/t5-v1_1-xxl"
        pretrained_vision_encoder_name_or_path = self._require_encoder_path("siglip-so400m-patch14-384")
        model = create_model(
            args=args["config"],
            dtype=torch.bfloat16,
            pretrained=args["pretrained_model_name_or_path"],
            ema_model_path=args.get("ema_model_path"),
            # pretrained_text_encoder_name_or_path=pretrained_text_encoder_name_or_path,
            pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path,
            control_frequency=args["ctrl_freq"],
        )

        return model


# RDT inference
def inference_fn(config, policy, lang_embeddings, observation_window):

    # print(f"Start inference_thread_fn: t={t}")
    while True:
        time1 = time.time()

        padded_window = list(observation_window)
        while len(padded_window) < config.get("img_history_size", 1):
            padded_window.insert(0, {
                "qpos": None,
                "images": {
                    camera_name: None for camera_name in config["camera_names"]
                },
            })
        padded_window = padded_window[-config.get("img_history_size", 1):]

        # Fetch images in [t-h+1, ..., t] x [front, right, left] order.
        image_arrs = []
        for frame in padded_window:
            for camera_name in config["camera_names"]:
                image_arrs.append(frame["images"][camera_name])

        images = [PImage.fromarray(arr) if arr is not None else None for arr in image_arrs]

        # get last qpos in shape [14, ]
        proprio = padded_window[-1]["qpos"]
        # unsqueeze to [1, 14]
        proprio = proprio.unsqueeze(0)

        # actions shaped as [1, 64, 14] in format [left, right]
        actions = (policy.step(proprio=proprio, images=images, text_embeds=lang_embeddings).squeeze(0).cpu().numpy())
        # print(f"inference_actions: {actions.squeeze()}")

        # print(f"Model inference time: {time.time() - time1} s")

        # print(f"Finish inference_thread_fn: t={t}")
        return actions
