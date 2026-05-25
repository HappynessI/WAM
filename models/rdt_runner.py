import copy
import os
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler

current_file = Path(__file__)
sys.path.append(os.path.join(current_file.parent))

from hub_mixin import CompatiblePyTorchModelHubMixin
from rdt.model import RDT


class RDTRunner(
    nn.Module,
    CompatiblePyTorchModelHubMixin,
    repo_url="https://huggingface.co/robotics-diffusion-transformer/rdt-1b",
):
    def __init__(
        self,
        *,
        action_dim,
        pred_horizon,
        config,
        lang_token_dim,
        img_token_dim,
        state_token_dim,
        max_lang_cond_len,
        img_cond_len,
        lang_pos_embed_config=None,
        img_pos_embed_config=None,
        dtype=torch.bfloat16,
        contact_flag_dim=0,
        action_loss_weight=1.0,
        point_loss_weight=1.0,
        contact_loss_weight=1.0,
        heatmap_loss_weight=0.0,
        action_state_dim=None,
        hand_points_dim=0,
        contact_point_dim=0,
        heatmap_channels=0,
        heatmap_height=0,
        heatmap_width=0,
        heatmap_horizon=0,
        enc_type=None,
        resolution=None,
        accelerator=None,
        learnable_tokens=None,
    ):
        super().__init__()
        hidden_size = config['rdt']['hidden_size']

        self.pred_horizon = pred_horizon
        self.total_target_dim = int(action_dim)
        self.state_token_dim = state_token_dim
        self.contact_flag_dim = contact_flag_dim
        self.action_loss_weight = float(action_loss_weight)
        self.point_loss_weight = float(point_loss_weight)
        self.contact_loss_weight = float(contact_loss_weight)
        self.heatmap_loss_weight = float(heatmap_loss_weight)
        self.action_state_dim = min(
            int(action_state_dim if action_state_dim is not None else state_token_dim),
            self.total_target_dim,
        )
        remaining_dim = max(self.total_target_dim - self.action_state_dim, 0)
        self.hand_points_dim = min(int(hand_points_dim), remaining_dim)
        remaining_dim -= self.hand_points_dim
        self.contact_point_dim = min(int(contact_point_dim), remaining_dim)
        remaining_dim -= self.contact_point_dim
        if remaining_dim != 0:
            raise ValueError(
                f"Target dimensions do not match: total={self.total_target_dim}, "
                f"action={self.action_state_dim}, hand={self.hand_points_dim}, contact={self.contact_point_dim}"
            )
        self.point_dim = self.hand_points_dim + self.contact_point_dim
        self.action_state_slice = slice(0, self.action_state_dim)
        self.point_slice = slice(self.action_state_dim, self.total_target_dim)
        self.hand_points_slice = slice(
            self.action_state_slice.stop,
            self.action_state_slice.stop + self.hand_points_dim,
        )
        self.contact_point_slice = slice(
            self.hand_points_slice.stop,
            self.hand_points_slice.stop + self.contact_point_dim,
        )
        self.heatmap_channels = int(heatmap_channels)
        self.heatmap_height = int(heatmap_height)
        self.heatmap_width = int(heatmap_width)
        self.heatmap_horizon = min(int(heatmap_horizon), self.pred_horizon)
        self.heatmap_enabled = (
            self.heatmap_channels > 0
            and self.heatmap_height > 0
            and self.heatmap_width > 0
            and self.heatmap_horizon > 0
        )
        self.heatmap_output_dim = (
            self.heatmap_channels * self.heatmap_height * self.heatmap_width
            if self.heatmap_enabled else 0
        )

        self.model = RDT(
            output_dim=self.action_state_dim,
            horizon=pred_horizon,
            point_output_dim=self.point_dim,
            point_horizon=(pred_horizon if self.point_dim > 0 else 0),
            heatmap_query_output_dim=self.heatmap_output_dim,
            heatmap_query_horizon=(self.heatmap_horizon if self.heatmap_enabled else 0),
            hidden_size=hidden_size,
            depth=config['rdt']['depth'],
            num_heads=config['rdt']['num_heads'],
            max_lang_cond_len=max_lang_cond_len,
            img_cond_len=img_cond_len,
            lang_pos_embed_config=lang_pos_embed_config,
            img_pos_embed_config=img_pos_embed_config,
            dtype=dtype,
        )

        self.lang_adaptor = self.build_condition_adapter(
            config['lang_adaptor'],
            in_features=lang_token_dim,
            out_features=hidden_size,
        )
        self.img_adaptor = self.build_condition_adapter(
            config['img_adaptor'],
            in_features=img_token_dim,
            out_features=hidden_size,
        )
        self.state_adaptor = self.build_condition_adapter(
            config['state_adaptor'],
            in_features=state_token_dim * 2,
            out_features=hidden_size,
        )
        self.action_adaptor = self.build_condition_adapter(
            config['state_adaptor'],
            in_features=self.action_state_dim * 2,
            out_features=hidden_size,
        )
        self.point_adaptor = None
        if self.point_dim > 0:
            self.point_adaptor = self.build_condition_adapter(
                config['state_adaptor'],
                in_features=self.point_dim * 2,
                out_features=hidden_size,
            )
        self.contact_head = nn.Linear(hidden_size, contact_flag_dim) if contact_flag_dim > 0 else None

        noise_scheduler_config = config['noise_scheduler']
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=noise_scheduler_config['num_train_timesteps'],
            beta_schedule=noise_scheduler_config['beta_schedule'],
            prediction_type=noise_scheduler_config['prediction_type'],
            clip_sample=noise_scheduler_config['clip_sample'],
        )
        self.noise_scheduler_sample = DPMSolverMultistepScheduler(
            num_train_timesteps=noise_scheduler_config['num_train_timesteps'],
            beta_schedule=noise_scheduler_config['beta_schedule'],
            prediction_type=noise_scheduler_config['prediction_type'],
        )

        self.num_train_timesteps = noise_scheduler_config['num_train_timesteps']
        self.num_inference_timesteps = noise_scheduler_config['num_inference_timesteps']
        self.prediction_type = noise_scheduler_config['prediction_type']

        modules = [self.model, self.lang_adaptor, self.img_adaptor, self.state_adaptor, self.action_adaptor]
        if self.point_adaptor is not None:
            modules.append(self.point_adaptor)
        total_params = sum(p.numel() for module in modules for p in module.parameters())
        if self.contact_head is not None:
            total_params += sum(p.numel() for p in self.contact_head.parameters())
        print(f"Diffusion params: {total_params:e}")

    def build_condition_adapter(self, projector_type, in_features, out_features):
        projector = None
        if projector_type == 'linear':
            projector = nn.Linear(in_features, out_features)
        else:
            mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))
                modules = [nn.Linear(in_features, out_features)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU(approximate="tanh"))
                    modules.append(nn.Linear(out_features, out_features))
                projector = nn.Sequential(*modules)
        if projector is None:
            raise ValueError(f'Unknown projector type: {projector_type}')
        return projector

    def _normalize_state_mask(self, state_mask, batch_size, device, dtype):
        if state_mask is None:
            state_mask = torch.ones((batch_size, self.state_token_dim), device=device, dtype=dtype)
        elif state_mask.dim() == 3:
            state_mask = state_mask[:, 0, :]
        return state_mask.to(device=device, dtype=dtype)

    def _normalize_action_mask(self, action_mask, batch_size, device, dtype):
        if action_mask is None:
            action_mask = torch.ones((batch_size, self.pred_horizon, self.total_target_dim), device=device, dtype=dtype)
        elif action_mask.dim() == 2:
            action_mask = action_mask.unsqueeze(1).expand(-1, self.pred_horizon, -1)
        elif action_mask.dim() == 3 and action_mask.shape[1] == 1:
            action_mask = action_mask.expand(-1, self.pred_horizon, -1)
        if action_mask.shape[-1] > self.total_target_dim:
            action_mask = action_mask[..., :self.total_target_dim]
        elif action_mask.shape[-1] < self.total_target_dim:
            pad_width = self.total_target_dim - action_mask.shape[-1]
            action_mask = F.pad(action_mask, (0, pad_width), value=0.0)
        return action_mask.to(device=device, dtype=dtype)

    @staticmethod
    def _masked_mean(loss_map, loss_mask):
        return (loss_map * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)

    def _normalize_action_tensor(self, action_tensor, batch_size, device, dtype):
        if action_tensor.shape[-1] > self.total_target_dim:
            action_tensor = action_tensor[..., :self.total_target_dim]
        elif action_tensor.shape[-1] < self.total_target_dim:
            pad_width = self.total_target_dim - action_tensor.shape[-1]
            action_tensor = F.pad(action_tensor, (0, pad_width), value=0.0)
        if action_tensor.shape[1] > self.pred_horizon:
            action_tensor = action_tensor[:, :self.pred_horizon]
        elif action_tensor.shape[1] < self.pred_horizon:
            pad_steps = self.pred_horizon - action_tensor.shape[1]
            action_tensor = F.pad(action_tensor, (0, 0, 0, pad_steps), value=0.0)
        return action_tensor.to(device=device, dtype=dtype)

    def _normalize_heatmap_targets(self, heatmap_gt, heatmap_mask, batch_size, device, dtype):
        if not self.heatmap_enabled or heatmap_gt is None:
            return None, None

        heatmap_gt = heatmap_gt.to(device=device, dtype=dtype)
        if heatmap_gt.shape[1] > self.heatmap_horizon:
            heatmap_gt = heatmap_gt[:, :self.heatmap_horizon]
        elif heatmap_gt.shape[1] < self.heatmap_horizon:
            pad_steps = self.heatmap_horizon - heatmap_gt.shape[1]
            heatmap_gt = F.pad(heatmap_gt, (0, 0, 0, 0, 0, 0, 0, pad_steps), value=0.0)

        if heatmap_mask is None:
            heatmap_mask = torch.ones(
                (batch_size, self.heatmap_horizon, self.heatmap_channels),
                device=device,
                dtype=dtype,
            )
        else:
            heatmap_mask = heatmap_mask.to(device=device, dtype=dtype)
            if heatmap_mask.shape[1] > self.heatmap_horizon:
                heatmap_mask = heatmap_mask[:, :self.heatmap_horizon]
            elif heatmap_mask.shape[1] < self.heatmap_horizon:
                pad_steps = self.heatmap_horizon - heatmap_mask.shape[1]
                heatmap_mask = F.pad(heatmap_mask, (0, 0, 0, pad_steps), value=0.0)

        return heatmap_gt, heatmap_mask

    def _reshape_heatmap_logits(self, heatmap_query_pred, batch_size):
        if heatmap_query_pred is None or not self.heatmap_enabled:
            return None
        return heatmap_query_pred.reshape(
            batch_size,
            self.heatmap_horizon,
            self.heatmap_channels,
            self.heatmap_height,
            self.heatmap_width,
        )

    def _split_targets(self, tensor):
        action = tensor[..., self.action_state_slice]
        point = tensor[..., self.point_slice] if self.point_dim > 0 else None
        return action, point

    def _merge_targets(self, action_tensor, point_tensor=None):
        if self.point_dim <= 0 or point_tensor is None:
            return action_tensor
        return torch.cat([action_tensor, point_tensor], dim=-1)

    def _compute_segment_losses(self, diffusion_loss_map, action_mask):
        segment_losses = {}

        if self.action_state_slice.stop > self.action_state_slice.start:
            segment_mask = action_mask[..., self.action_state_slice]
            segment_losses['action_loss'] = self._masked_mean(
                diffusion_loss_map[..., self.action_state_slice],
                segment_mask,
            )

        if self.hand_points_slice.stop > self.hand_points_slice.start:
            segment_mask = action_mask[..., self.hand_points_slice]
            segment_losses['hand_points_loss'] = self._masked_mean(
                diffusion_loss_map[..., self.hand_points_slice],
                segment_mask,
            )

        if self.contact_point_slice.stop > self.contact_point_slice.start:
            segment_mask = action_mask[..., self.contact_point_slice]
            segment_losses['contact_points_loss'] = self._masked_mean(
                diffusion_loss_map[..., self.contact_point_slice],
                segment_mask,
            )

        if self.point_dim > 0:
            aux_mask = action_mask[..., self.point_slice]
            segment_losses['point_loss'] = self._masked_mean(
                diffusion_loss_map[..., self.point_slice],
                aux_mask,
            )

        return segment_losses

    def adapt_conditions(self, lang_tokens, img_tokens, state_tokens, action_tokens=None, point_tokens=None):
        adapted_lang = self.lang_adaptor(lang_tokens)
        adapted_img = self.img_adaptor(img_tokens)
        adapted_state = self.state_adaptor(state_tokens)
        outputs = [adapted_lang, adapted_img, adapted_state]
        if action_tokens is not None:
            outputs.append(self.action_adaptor(action_tokens))
        if point_tokens is not None and self.point_adaptor is not None:
            outputs.append(self.point_adaptor(point_tokens))
        return tuple(outputs)

    def _model_forward(self, sequence_tokens, ctrl_freqs, timesteps, lang_cond, img_cond, lang_attn_mask, return_hidden=False):
        outputs = self.model(
            sequence_tokens,
            ctrl_freqs,
            timesteps,
            lang_cond,
            img_cond,
            lang_mask=lang_attn_mask,
            return_hidden=return_hidden,
        )
        if torch.is_tensor(outputs):
            if return_hidden:
                return outputs, None, None, None, None, None
            return outputs, None, None, None, None, None

        if return_hidden:
            if len(outputs) == 6:
                return outputs
            if len(outputs) == 4:
                action_pred, point_pred, action_hidden, point_hidden = outputs
                return action_pred, point_pred, None, action_hidden, point_hidden, None
            if len(outputs) == 2:
                action_pred, action_hidden = outputs
                return action_pred, None, None, action_hidden, None, None
        else:
            if len(outputs) == 3:
                action_pred, point_pred, heatmap_pred = outputs
                return action_pred, point_pred, heatmap_pred, None, None, None
            if len(outputs) == 2:
                action_pred, point_pred = outputs
                return action_pred, point_pred, None, None, None, None

        raise ValueError(f"Unexpected model output format with return_hidden={return_hidden}: {type(outputs)}")

    def _build_sequence(self, adapted_state, adapted_action, adapted_point=None):
        seqs = [adapted_state, adapted_action]
        if adapted_point is not None:
            seqs.append(adapted_point)
        return torch.cat(seqs, dim=1)

    def conditional_sample(self, lang_cond, lang_attn_mask, img_cond, state_traj, action_mask, ctrl_freqs):
        device = state_traj.device
        dtype = state_traj.dtype
        batch_size = state_traj.shape[0]
        total_mask = self._normalize_action_mask(action_mask, batch_size, device, dtype)
        action_mask_branch, point_mask_branch = self._split_targets(total_mask)
        noisy_action = torch.randn(
            size=(batch_size, self.pred_horizon, self.action_state_dim),
            dtype=dtype,
            device=device,
        )
        noisy_point = None
        if self.point_dim > 0:
            noisy_point = torch.randn(
                size=(batch_size, self.pred_horizon, self.point_dim),
                dtype=dtype,
                device=device,
            )

        action_scheduler = copy.deepcopy(self.noise_scheduler_sample)
        action_scheduler.set_timesteps(self.num_inference_timesteps)
        point_scheduler = None
        if noisy_point is not None:
            point_scheduler = copy.deepcopy(self.noise_scheduler_sample)
            point_scheduler.set_timesteps(self.num_inference_timesteps)

        for timestep in action_scheduler.timesteps:
            adapted_action = self.action_adaptor(torch.cat([noisy_action, action_mask_branch], dim=-1))
            adapted_point = None
            if noisy_point is not None:
                adapted_point = self.point_adaptor(torch.cat([noisy_point, point_mask_branch], dim=-1))
            sequence_tokens = self._build_sequence(state_traj, adapted_action, adapted_point)
            action_pred, point_pred, _, _, _, _ = self._model_forward(
                sequence_tokens,
                ctrl_freqs,
                timestep.unsqueeze(-1).to(device),
                lang_cond,
                img_cond,
                lang_attn_mask,
                return_hidden=False,
            )
            noisy_action = action_scheduler.step(action_pred, timestep, noisy_action).prev_sample.to(dtype)
            if noisy_point is not None and point_scheduler is not None:
                noisy_point = point_scheduler.step(point_pred, timestep, noisy_point).prev_sample.to(dtype)

        action_pred = noisy_action * action_mask_branch
        point_pred = None if noisy_point is None else noisy_point * point_mask_branch
        return self._merge_targets(action_pred, point_pred)

    def compute_loss(
        self,
        lang_tokens,
        lang_attn_mask,
        img_tokens,
        state_tokens,
        action_gt,
        action_mask,
        ctrl_freqs,
        state_mask=None,
        contact_gt=None,
        contact_mask=None,
        heatmap_gt=None,
        heatmap_mask=None,
        return_dict=False,
    ):
        batch_size = lang_tokens.shape[0]
        device = lang_tokens.device
        dtype = lang_tokens.dtype

        state_mask = self._normalize_state_mask(state_mask, batch_size, device, dtype)
        total_mask = self._normalize_action_mask(action_mask, batch_size, device, dtype)
        action_gt = self._normalize_action_tensor(action_gt, batch_size, device, dtype)
        action_mask_branch, point_mask_branch = self._split_targets(total_mask)
        action_gt_branch, point_gt_branch = self._split_targets(action_gt)
        heatmap_gt, heatmap_mask = self._normalize_heatmap_targets(
            heatmap_gt,
            heatmap_mask,
            batch_size,
            device,
            dtype,
        )

        action_noise = torch.randn_like(action_gt_branch)
        timesteps = torch.randint(0, self.num_train_timesteps, (batch_size,), device=device).long()
        noisy_action = self.noise_scheduler.add_noise(action_gt_branch, action_noise, timesteps)
        point_noise = None
        noisy_point = None
        if point_gt_branch is not None:
            point_noise = torch.randn_like(point_gt_branch)
            noisy_point = self.noise_scheduler.add_noise(point_gt_branch, point_noise, timesteps)

        state_tokens = torch.cat([state_tokens, state_mask.unsqueeze(1)], dim=-1)
        action_tokens = torch.cat([noisy_action, action_mask_branch], dim=-1)
        point_tokens = None if noisy_point is None else torch.cat([noisy_point, point_mask_branch], dim=-1)
        adapted = self.adapt_conditions(
            lang_tokens,
            img_tokens,
            state_tokens,
            action_tokens,
            point_tokens,
        )
        if point_tokens is not None:
            lang_cond, img_cond, adapted_state, adapted_action, adapted_point = adapted
        else:
            lang_cond, img_cond, adapted_state, adapted_action = adapted
            adapted_point = None
        sequence_tokens = self._build_sequence(adapted_state, adapted_action, adapted_point)
        action_pred, point_pred, heatmap_query_pred, action_hidden, point_hidden, heatmap_query_hidden = self._model_forward(
            sequence_tokens,
            ctrl_freqs,
            timesteps,
            lang_cond,
            img_cond,
            lang_attn_mask,
            return_hidden=self.contact_head is not None,
        )

        if self.prediction_type == 'epsilon':
            target_action = action_noise
            target_point = point_noise
        elif self.prediction_type == 'sample':
            target_action = action_gt_branch
            target_point = point_gt_branch
        else:
            raise ValueError(f"Unsupported prediction type {self.prediction_type}")

        model_pred_total = self._merge_targets(action_pred, point_pred)
        target_total = self._merge_targets(target_action, target_point)
        diffusion_loss_map = F.mse_loss(model_pred_total, target_total, reduction='none')
        diffusion_loss = self._masked_mean(diffusion_loss_map, total_mask)
        segment_losses = self._compute_segment_losses(diffusion_loss_map, total_mask)
        action_loss = segment_losses.get('action_loss', diffusion_loss)
        point_loss = segment_losses.get('point_loss')
        total_loss = self.action_loss_weight * action_loss
        if point_loss is not None:
            total_loss = total_loss + self.point_loss_weight * point_loss
        result = {
            'loss': total_loss,
            'diffusion_loss': diffusion_loss.detach(),
        }
        for name, value in segment_losses.items():
            result[name] = value.detach()

        if heatmap_gt is not None and heatmap_query_pred is not None:
            heatmap_logits = self._reshape_heatmap_logits(heatmap_query_pred, batch_size)
            heatmap_probs = torch.sigmoid(heatmap_logits)
            heatmap_loss_map = F.mse_loss(heatmap_probs, heatmap_gt, reduction='none')
            heatmap_loss_per_channel = heatmap_loss_map.flatten(start_dim=-2).mean(dim=-1)
            heatmap_loss = self._masked_mean(heatmap_loss_per_channel, heatmap_mask)
            total_loss = total_loss + self.heatmap_loss_weight * heatmap_loss
            result['heatmap_loss'] = heatmap_loss.detach()

        if self.contact_head is not None and contact_gt is not None:
            contact_hidden = point_hidden if point_hidden is not None else action_hidden
            contact_logits = self.contact_head(contact_hidden)
            if contact_mask is None:
                contact_mask = torch.ones_like(contact_gt)
            else:
                contact_mask = contact_mask.to(device=device, dtype=dtype)
            contact_gt = contact_gt.to(device=device, dtype=dtype)
            contact_loss_map = F.binary_cross_entropy_with_logits(contact_logits, contact_gt, reduction='none')
            contact_loss = (contact_loss_map * contact_mask).sum() / contact_mask.sum().clamp_min(1.0)
            total_loss = total_loss + self.contact_loss_weight * contact_loss
            result['contact_loss'] = contact_loss.detach()
            result['contact_logits'] = contact_logits

        result['loss'] = total_loss
        if return_dict:
            return result
        return total_loss

    def predict_action(
        self,
        lang_tokens,
        lang_attn_mask,
        img_tokens,
        state_tokens,
        action_mask,
        ctrl_freqs,
        state_mask=None,
        return_dict=False,
    ):
        batch_size = state_tokens.shape[0]
        device = state_tokens.device
        dtype = state_tokens.dtype
        state_mask = self._normalize_state_mask(state_mask, batch_size, device, dtype)
        total_mask = self._normalize_action_mask(action_mask, batch_size, device, dtype)
        action_mask_branch, point_mask_branch = self._split_targets(total_mask)

        state_tokens = torch.cat([state_tokens, state_mask.unsqueeze(1)], dim=-1)
        lang_cond, img_cond, state_traj = self.adapt_conditions(lang_tokens, img_tokens, state_tokens)
        trajectory_pred = self.conditional_sample(
            lang_cond,
            lang_attn_mask,
            img_cond,
            state_traj,
            total_mask,
            ctrl_freqs,
        )

        if self.contact_head is None and not self.heatmap_enabled and not return_dict:
            return trajectory_pred

        output = {'trajectory': trajectory_pred}
        if self.contact_head is not None or self.heatmap_enabled:
            final_timesteps = torch.zeros((batch_size,), device=device, dtype=torch.long)
            action_pred_branch, point_pred_branch = self._split_targets(trajectory_pred)
            adapted_action = self.action_adaptor(torch.cat([action_pred_branch, action_mask_branch], dim=-1))
            adapted_point = None
            if point_pred_branch is not None:
                adapted_point = self.point_adaptor(torch.cat([point_pred_branch, point_mask_branch], dim=-1))
            clean_sequence = self._build_sequence(state_traj, adapted_action, adapted_point)
            _, _, heatmap_query_pred, action_hidden, point_hidden, _ = self._model_forward(
                clean_sequence,
                ctrl_freqs,
                final_timesteps,
                lang_cond,
                img_cond,
                lang_attn_mask,
                return_hidden=True,
            )
            if self.contact_head is not None:
                contact_hidden = point_hidden if point_hidden is not None else action_hidden
                contact_logits = self.contact_head(contact_hidden)
                output['contact_logits'] = contact_logits
                output['contact_probs'] = torch.sigmoid(contact_logits)
            if self.heatmap_enabled and heatmap_query_pred is not None:
                heatmap_logits = self._reshape_heatmap_logits(heatmap_query_pred, batch_size)
                output['heatmap_logits'] = heatmap_logits
                output['heatmap_probs'] = torch.sigmoid(heatmap_logits)

        if return_dict:
            return output
        return output['trajectory']

    def forward(self, *args, **kwargs):
        return self.compute_loss(*args, **kwargs)
