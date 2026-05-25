# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DiT: https://github.com/facebookresearch/DiT
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
from collections import OrderedDict
from pathlib import Path
import os
import sys

import torch
import torch.nn as nn

current_file = Path(__file__)
sys.path.append(str(current_file.parent.parent))

from rdt.blocks import (
    FinalLayer,
    RDTBlock,
    TimestepEmbedder,
    get_1d_sincos_pos_embed_from_grid,
    get_multimodal_cond_pos_embed,
)


class RDT(nn.Module):
    """Robotics Diffusion Transformer backbone."""

    def __init__(
        self,
        output_dim=128,
        horizon=32,
        point_output_dim=0,
        point_horizon=0,
        heatmap_query_output_dim=0,
        heatmap_query_horizon=0,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        max_lang_cond_len=1024,
        img_cond_len=4096,
        lang_pos_embed_config=None,
        img_pos_embed_config=None,
        dtype=torch.bfloat16,
    ):
        super().__init__()
        self.horizon = horizon
        self.point_horizon = point_horizon
        self.heatmap_query_horizon = heatmap_query_horizon
        self.total_horizon = horizon + point_horizon
        self.total_output_tokens = horizon + point_horizon + heatmap_query_horizon
        self.hidden_size = hidden_size
        self.max_lang_cond_len = max_lang_cond_len
        self.img_cond_len = img_cond_len
        self.dtype = dtype
        self.lang_pos_embed_config = lang_pos_embed_config
        self.img_pos_embed_config = img_pos_embed_config

        self.t_embedder = TimestepEmbedder(hidden_size, dtype=dtype)
        self.freq_embedder = TimestepEmbedder(hidden_size, dtype=dtype)

        # [timestep; ctrl_freq; state; action_1 ... action_H; point_1 ... point_H; heatmap_query_1 ... heatmap_query_K]
        self.x_pos_embed = nn.Parameter(torch.zeros(1, self.total_output_tokens + 3, hidden_size))
        self.lang_cond_pos_embed = nn.Parameter(torch.zeros(1, max_lang_cond_len, hidden_size))
        self.img_cond_pos_embed = nn.Parameter(torch.zeros(1, img_cond_len, hidden_size))
        self.heatmap_query_tokens = (
            nn.Parameter(torch.zeros(1, heatmap_query_horizon, hidden_size))
            if heatmap_query_output_dim > 0 and heatmap_query_horizon > 0 else None
        )

        self.blocks = nn.ModuleList([RDTBlock(hidden_size, num_heads) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, output_dim)
        self.point_final_layer = (
            FinalLayer(hidden_size, point_output_dim)
            if point_output_dim > 0 and point_horizon > 0 else None
        )
        self.heatmap_query_final_layer = (
            FinalLayer(hidden_size, heatmap_query_output_dim)
            if heatmap_query_output_dim > 0 and heatmap_query_horizon > 0 else None
        )
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        mm_cond_lens = OrderedDict([
            ('timestep', 1),
            ('ctrl_freq', 1),
            ('state', 1),
            ('action', self.horizon),
        ])
        if self.point_horizon > 0:
            mm_cond_lens['point'] = self.point_horizon
        if self.heatmap_query_horizon > 0:
            mm_cond_lens['heatmap_query'] = self.heatmap_query_horizon
        x_pos_embed = get_multimodal_cond_pos_embed(
            embed_dim=self.hidden_size,
            mm_cond_lens=mm_cond_lens,
        )
        self.x_pos_embed.data.copy_(torch.from_numpy(x_pos_embed).float().unsqueeze(0))

        if self.lang_pos_embed_config is None:
            lang_cond_pos_embed = get_1d_sincos_pos_embed_from_grid(
                self.hidden_size,
                torch.arange(self.max_lang_cond_len),
            )
        else:
            lang_cond_pos_embed = get_multimodal_cond_pos_embed(
                embed_dim=self.hidden_size,
                mm_cond_lens=OrderedDict(self.lang_pos_embed_config),
                embed_modality=False,
            )
        self.lang_cond_pos_embed.data.copy_(torch.from_numpy(lang_cond_pos_embed).float().unsqueeze(0))

        if self.img_pos_embed_config is None:
            img_cond_pos_embed = get_1d_sincos_pos_embed_from_grid(
                self.hidden_size,
                torch.arange(self.img_cond_len),
            )
        else:
            img_cond_pos_embed = get_multimodal_cond_pos_embed(
                embed_dim=self.hidden_size,
                mm_cond_lens=OrderedDict(self.img_pos_embed_config),
                embed_modality=False,
            )
        self.img_cond_pos_embed.data.copy_(torch.from_numpy(img_cond_pos_embed).float().unsqueeze(0))

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.freq_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.freq_embedder.mlp[2].weight, std=0.02)

        nn.init.constant_(self.final_layer.ffn_final.fc2.weight, 0)
        nn.init.constant_(self.final_layer.ffn_final.fc2.bias, 0)
        if self.point_final_layer is not None:
            nn.init.constant_(self.point_final_layer.ffn_final.fc2.weight, 0)
            nn.init.constant_(self.point_final_layer.ffn_final.fc2.bias, 0)
        if self.heatmap_query_final_layer is not None:
            nn.init.constant_(self.heatmap_query_final_layer.ffn_final.fc2.weight, 0)
            nn.init.constant_(self.heatmap_query_final_layer.ffn_final.fc2.bias, 0)
        if self.heatmap_query_tokens is not None:
            nn.init.normal_(self.heatmap_query_tokens, mean=0.0, std=0.02)

        self.to(self.dtype)

    def forward(self, x, freq, t, lang_c, img_c, lang_mask=None, img_mask=None, return_hidden=False):
        """
        x: (B, 1 + H_action + H_point, D) hidden state + hidden action/point-token sequence.
        freq: (B,)
        t: (B,) or (1,)
        lang_c: (B, L_lang, D)
        img_c: (B, L_img, D)
        """
        t = self.t_embedder(t).unsqueeze(1)
        freq = self.freq_embedder(freq).unsqueeze(1)
        if t.shape[0] == 1:
            t = t.expand(x.shape[0], -1, -1)
        x = torch.cat([t, freq, x], dim=1)
        if self.heatmap_query_tokens is not None:
            x = torch.cat([x, self.heatmap_query_tokens.expand(x.shape[0], -1, -1)], dim=1)

        x = x + self.x_pos_embed[:, :x.shape[1]]
        lang_c = lang_c + self.lang_cond_pos_embed[:, :lang_c.shape[1]]
        img_c = img_c + self.img_cond_pos_embed[:, :img_c.shape[1]]

        conds = [lang_c, img_c]
        masks = [lang_mask, img_mask]
        for index, block in enumerate(self.blocks):
            cond, mask = conds[index % 2], masks[index % 2]
            x = block(x, cond, mask)

        token_hidden = x[:, -self.total_output_tokens:]
        cursor = 0
        action_hidden = token_hidden[:, cursor:cursor + self.horizon]
        cursor += self.horizon
        action_pred = self.final_layer(action_hidden)

        point_hidden = None
        point_pred = None
        if self.point_final_layer is not None:
            point_hidden = token_hidden[:, cursor:cursor + self.point_horizon]
            cursor += self.point_horizon
            point_pred = self.point_final_layer(point_hidden)

        heatmap_query_hidden = None
        heatmap_query_pred = None
        if self.heatmap_query_final_layer is not None:
            heatmap_query_hidden = token_hidden[:, cursor:cursor + self.heatmap_query_horizon]
            heatmap_query_pred = self.heatmap_query_final_layer(heatmap_query_hidden)

        if self.point_final_layer is None and self.heatmap_query_final_layer is None:
            if return_hidden:
                return action_pred, None, None, action_hidden, None, None
            return action_pred
        if return_hidden:
            return action_pred, point_pred, heatmap_query_pred, action_hidden, point_hidden, heatmap_query_hidden
        return action_pred, point_pred, heatmap_query_pred
