# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modifications Copyright (c) 2026 DUET-DINO contributors.
#
# Adapted from V-JEPA2 (https://github.com/facebookresearch/vjepa2).
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from functools import partial

import torch
import torch.nn as nn

from src_vjepa2.models_v2.utils.modules import ACBlock as Block
from src_vjepa2.models_v2.utils.modules import CrossAttentionBlock
from src_vjepa2.models_v2.utils.modules import build_action_block_causal_attention_mask
from src_vjepa2.utils.tensors import trunc_normal_


class VisionTransformerPredictorAC(nn.Module):
    """Action Conditioned Vision Transformer Predictor"""

    def __init__(
        self,
        img_size=(224, 224),
        patch_size=16,
        num_frames=1,
        tubelet_size=2,
        embed_dim=768,
        predictor_embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        uniform_power=True,
        use_silu=False,
        wide_silu=True,
        is_frame_causal=True,
        use_activation_checkpointing=False,
        use_rope=True,
        action_embed_dim=7,
        use_extrinsics=False,
        **kwargs
    ):
        super().__init__()
        self.is_frame_causal = is_frame_causal
        self.use_extrinsics = use_extrinsics

        # Map input to predictor dimension
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        # TODO: from nn.leanear to beast
        self.action_encoder = nn.Linear(action_embed_dim, predictor_embed_dim, bias=True)
        self.state_encoder = nn.Linear(action_embed_dim, predictor_embed_dim, bias=True)
        self.extrinsics_encoder = nn.Linear(action_embed_dim - 1, predictor_embed_dim, bias=True)

        # Determine positional embedding
        if type(img_size) is int:
            img_size = (img_size, img_size)
        self.img_height, self.img_width = img_size
        self.patch_size = patch_size
        # --
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.is_video = num_frames > 1

        self.grid_height = img_size[0] // self.patch_size
        self.grid_width = img_size[1] // self.patch_size
        self.use_activation_checkpointing = use_activation_checkpointing

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        # Position embedding
        self.uniform_power = uniform_power

        # Attention Blocks
        self.use_rope = use_rope
        self.predictor_blocks = nn.ModuleList(
            [
                Block(
                    use_rope=use_rope,
                    grid_size=self.grid_height,
                    dim=predictor_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=nn.SiLU if use_silu else nn.GELU,
                    wide_silu=wide_silu,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        # Normalize & project back to input dimension
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

        # ------ initialize weights
        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

        attn_mask = None
        if self.is_frame_causal:
            grid_depth = self.num_frames // self.tubelet_size
            grid_height = self.img_height // self.patch_size
            grid_width = self.img_width // self.patch_size
            attn_mask = build_action_block_causal_attention_mask(
                grid_depth, grid_height, grid_width, add_tokens=3 if use_extrinsics else 2
            )
        self.register_buffer("attn_mask", attn_mask, persistent=False)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def forward(self, x, actions, states, extrinsics=None):
        """
        :param x: context tokens
        """
        # Map tokens to predictor dimensions
        x = self.predictor_embed(x)
        B, N_ctxt, D = x.size()
        T = N_ctxt // (self.grid_height * self.grid_width)

        # Interleave action tokens
        s = self.state_encoder(states).unsqueeze(2)
        a = self.action_encoder(actions).unsqueeze(2)
        x = x.view(B, T, self.grid_height * self.grid_width, D)  # [B, T, H*W, D]
        if self.use_extrinsics:
            e = self.extrinsics_encoder(extrinsics).unsqueeze(2)
            x = torch.cat([a, s, e, x], dim=2).flatten(1, 2)  # [B, T*(H*W+3), D]
        else:
            x = torch.cat([a, s, x], dim=2).flatten(1, 2)  # [B, T*(H*W+2), D]

        cond_tokens = 3 if self.use_extrinsics else 2
        # One frame has no future tokens to mask; None also allows Flash Attention.
        attn_mask = (
            self.attn_mask[: x.size(1), : x.size(1)]
            if self.attn_mask is not None and T > 1 else None
        )

        # Fwd prop
        for i, blk in enumerate(self.predictor_blocks):
            if self.use_activation_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    blk,
                    x,
                    mask=None,
                    attn_mask=attn_mask,
                    T=T,
                    H=self.grid_height,
                    W=self.grid_width,
                    action_tokens=cond_tokens,
                    use_reentrant=False,
                )
            else:
                x = blk(
                    x,
                    mask=None,
                    attn_mask=attn_mask,
                    T=T,
                    H=self.grid_height,
                    W=self.grid_width,
                    action_tokens=cond_tokens,
                )

        # Split out action and frame tokens
        x = x.view(B, T, cond_tokens + self.grid_height * self.grid_width, D)  # [B, T, K+H*W, D]
        x = x[:, :, cond_tokens:, :].flatten(1, 2)

        x = self.predictor_norm(x)
        x = self.predictor_proj(x)

        return x


def vit_ac_predictor(**kwargs):
    model = VisionTransformerPredictorAC(
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


# Dual-view predictor with bidirectional cross-attention and separate heads
class DualViewCrossAttentionPredictorAC(nn.Module):
    """Dual-view predictor with frame-local cross-attention and separate causal heads."""

    def __init__(
        self,
        cross_attn_num_heads=16,
        mlp_ratio=4.0,
        **predictor_kwargs
    ):
        super().__init__()
        embed_dim = predictor_kwargs["embed_dim"]

        # shared attention blocks
        self.left_attends_wrist = CrossAttentionBlock(
            dim=embed_dim,
            num_heads=cross_attn_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            act_layer=nn.GELU,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )
        self.wrist_attends_left = CrossAttentionBlock(
            dim=embed_dim,
            num_heads=cross_attn_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            act_layer=nn.GELU,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )

        # separate predictor heads
        self.left_predictor = VisionTransformerPredictorAC(
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **predictor_kwargs
        )
        self.wrist_predictor = VisionTransformerPredictorAC(
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **predictor_kwargs
        )

    def forward(self, 
                left_x, 
                wrist_x, 
                actions, 
                states, 
                left_extrinsics=None, 
                wrist_extrinsics=None):
        # Fuse synchronized images independently so future frames cannot leak
        # through cross-attention before the temporal causal predictor.
        batch_size, num_tokens, embed_dim = left_x.shape
        patches_per_frame = self.left_predictor.grid_height * self.left_predictor.grid_width
        num_frames = num_tokens // patches_per_frame
        left_frames = left_x.reshape(batch_size * num_frames, patches_per_frame, embed_dim)
        wrist_frames = wrist_x.reshape(batch_size * num_frames, patches_per_frame, embed_dim)
        
        left_fused_emb = self.left_attends_wrist(left_frames, wrist_frames).reshape(
            batch_size, num_tokens, embed_dim
        )
        wrist_fused_emb = self.wrist_attends_left(wrist_frames, left_frames).reshape(
            batch_size, num_tokens, embed_dim
        )

        # predict
        left_out = self.left_predictor(left_fused_emb, 
                                       actions, 
                                       states, 
                                       left_extrinsics)
        wrist_out = self.wrist_predictor(wrist_fused_emb, 
                                        actions, 
                                        states, 
                                        wrist_extrinsics)
        return left_out, wrist_out



# Dual-view predictor with no bidirectional cross-attention and separate heads
class DualViewPredictorAC(nn.Module):
    """Dual-view predictor with no bidirectional cross-attention and separate heads."""

    def __init__(
        self,
        cross_attn_num_heads=16,
        mlp_ratio=4.0,
        **predictor_kwargs
    ):
        super().__init__()
        embed_dim = predictor_kwargs["embed_dim"]

        # separate predictor heads
        self.left_predictor = VisionTransformerPredictorAC(
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **predictor_kwargs
        )
        self.wrist_predictor = VisionTransformerPredictorAC(
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **predictor_kwargs
        )

    def forward(self, 
                left_x, 
                wrist_x, 
                actions, 
                states, 
                left_extrinsics=None, 
                wrist_extrinsics=None):

        # predict
        left_out = self.left_predictor(left_x, 
                                       actions, 
                                       states, 
                                       left_extrinsics)
        wrist_out = self.wrist_predictor(wrist_x, 
                                        actions, 
                                        states, 
                                        wrist_extrinsics)
        return left_out, wrist_out


def vit_ac_dual_predictor(**kwargs):
    model = DualViewCrossAttentionPredictorAC(**kwargs)
    return model

def vit_ac_nocrossattn_dual_predictor(**kwargs):
    model = DualViewPredictorAC(**kwargs)
    return model
