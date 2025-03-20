"""MIRROR Implementation
Copyright (c) 2025, Tianyi Wang @ The University of Sydney
All rights reserved.

Based on the timm codebase by Ross Wightman
https://github.com/huggingface/pytorch-image-models

Based on the TransMIL codebase by Zhuchen Shao
https://github.com/szc19990412/TransMIL

Based on the swav codebase by Meta Research
https://github.com/facebookresearch/swav

Based on the mae codebase by Meta Research
https://github.com/facebookresearch/mae

Licensed under the GNU General Public License v3.0, see LICENSE for details
"""

import logging
import math
from functools import partial
from typing import Callable, Final, Literal, Optional, Type

import numpy as np
import torch
import torch.nn.functional as F
from nystrom_attention import NystromAttention
from timm.layers import (
    DropPath,
    LayerType,
    Mlp,
    get_act_layer,
    get_norm_layer,
    trunc_normal_,
    use_fused_attn,
)
from timm.models import register_model
from timm.models.vision_transformer import LayerScale
from torch import nn
from torch.distributions import Normal


_logger = logging.getLogger(__name__)


# ===========================================
#  Transformer for Transcriptomics Data
# ===========================================
class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,  # type: ignore[assignment]
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N = x.shape  # noqa: N806
        qkv = (
            self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim).permute(1, 0, 2, 3)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,  # type: ignore[assignment]
        norm_layer: nn.Module = nn.LayerNorm,  # type: ignore[assignment]
        mlp_layer: nn.Module = Mlp,  # type: ignore[assignment]
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class TransFormer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 768,
        depth: int = 2,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        init_values: Optional[float] = None,
        gene_embed: str = "learn",
        pre_norm: bool = False,
        final_norm: bool = True,
        embed_drop_rate: float = 0.0,
        pos_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        weight_init: Literal["skip", "jax", "jax_nlhb", "moco", ""] = "",
        fix_init: bool = False,
        embed_layer: Callable = Mlp,
        norm_layer: Optional[LayerType] = None,
        act_layer: Optional[LayerType] = None,
        block_fn: Type[nn.Module] = Block,
        mlp_layer: Type[nn.Module] = Mlp,
    ) -> None:
        super().__init__()
        assert gene_embed in ("", "none", "learn")
        norm_layer = get_norm_layer(norm_layer) or partial(nn.LayerNorm, eps=1e-6)
        act_layer = get_act_layer(act_layer) or nn.GELU  # type: ignore[arg-type]

        self.num_features = self.head_hidden_size = self.embed_dim = (
            embed_dim  # for consistency with other models
        )

        self.embedding = embed_layer(
            in_features=input_dim,
            hidden_features=embed_dim * 2,
            out_features=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop=embed_drop_rate,
        )

        if not gene_embed or gene_embed == "none":
            self.gene_embed = None
        else:
            self.gene_embed = nn.Parameter(torch.randn(1, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.Sequential(
            *[
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    init_values=init_values,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    mlp_layer=mlp_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim) if final_norm else nn.Identity()

        if weight_init != "skip":
            self.init_weights(weight_init)
        if fix_init:
            self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, _layer_id):
            param.div_(math.sqrt(2.0 * _layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def init_weights(self, mode: str = "") -> None:
        assert mode in ("jax", "jax_nlhb", "moco", "")
        if self.gene_embed is not None:
            trunc_normal_(self.gene_embed, std=0.02)

    def _gene_embed(self, x):
        if self.gene_embed is None:
            return x

        x = x + self.gene_embed

        return self.pos_drop(x)

    def forward(self, x):
        x = self.embedding(x)
        x = self._gene_embed(x)
        x = self.norm_pre(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x


# ===========================================
#  TransMIL for Histopathology Data
# ===========================================
class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,
            # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,
            # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1,
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):  # noqa: N803
        B, _, C = x.shape  # noqa: N806
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class FeatureTransMIL(nn.Module):
    def __init__(self, input_dim=1024, embed_dim=512):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        self.pos_layer = PPEG(dim=self.embed_dim)
        self._fc1 = nn.Sequential(nn.Linear(self.input_dim, self.embed_dim), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.layer1 = TransLayer(dim=self.embed_dim)
        self.layer2 = TransLayer(dim=self.embed_dim)
        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(self, h):
        h = h.float()  # [B, n, 1024]

        h = self._fc1(h)  # [B, n, 512]

        # ---->pad
        H = h.shape[1]  # noqa: N806
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))  # noqa: N806
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, 512]

        # ---->cls_token
        B = h.shape[0]  # noqa: N806
        cls_tokens = self.cls_token.expand(B, -1, -1)
        h = torch.cat((cls_tokens, h), dim=1)

        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 512]

        # ---->PPEG
        h = self.pos_layer(h, _H, _W)  # [B, N, 512]

        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512]

        # ---->cls_token
        h = self.norm(h)[:, 0]

        return h


# ===========================================
#  TransFormer for Pre-training
# ===========================================
class TransFormerHybrid(TransFormer):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 768,
        depth: int = 2,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        init_values: Optional[float] = None,
        gene_embed: str = "learn",
        pre_norm: bool = False,
        final_norm: bool = True,
        embed_drop_rate: float = 0.0,
        pos_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        weight_init: Literal["skip", "jax", "jax_nlhb", "moco", ""] = "",
        fix_init: bool = False,
        embed_layer: Callable = Mlp,
        norm_layer: Optional[LayerType] = None,
        act_layer: Optional[LayerType] = None,
        block_fn: Type[nn.Module] = Block,
        mlp_layer: Type[nn.Module] = Mlp,
        retention_decoder_depth: int = 1,
    ):
        super().__init__(
            input_dim=input_dim,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            init_values=init_values,
            gene_embed=gene_embed,
            pre_norm=pre_norm,
            final_norm=final_norm,
            embed_drop_rate=embed_drop_rate,
            pos_drop_rate=pos_drop_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            weight_init=weight_init,
            fix_init=fix_init,
            embed_layer=embed_layer,
            norm_layer=norm_layer,
            act_layer=act_layer,
            block_fn=block_fn,
            mlp_layer=mlp_layer,
        )

        norm_layer = get_norm_layer(norm_layer) or partial(nn.LayerNorm, eps=1e-6)
        act_layer = get_act_layer(act_layer) or nn.GELU  # type: ignore[arg-type]

        self.alignment_head = nn.Linear(embed_dim, embed_dim)

        self.retention_embed = nn.Linear(embed_dim, embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1))
        self.retention_gene_embed = nn.Parameter(torch.randn(1, embed_dim) * 0.02)
        self.retention_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    init_values=init_values,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=0,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    mlp_layer=mlp_layer,
                )
                for i in range(retention_decoder_depth)
            ]
        )
        self.retention_norm = norm_layer(embed_dim)
        self.retention_head = nn.Linear(embed_dim, embed_dim)

        self.init_weights_()

    def init_weights_(self):
        torch.nn.init.normal_(self.mask_token, std=0.02)
        trunc_normal_(self.retention_gene_embed, std=0.02)

        def rescale(param, _layer_id):
            param.div_(math.sqrt(2.0 * _layer_id))

        for layer_id, layer in enumerate(self.retention_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def random_masking(self, x, mask_ratio):
        B, N = x.shape  # noqa: N806
        len_keep = int(N * (1 - mask_ratio))

        noise = torch.rand(B, N, device=x.device)

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep)
        mask_tokens = self.mask_token.repeat(
            B, ids_restore.shape[1] - x_masked.shape[1]
        )
        x_masked = torch.cat([x_masked, mask_tokens], dim=1)
        x_masked = torch.gather(x_masked, dim=1, index=ids_restore)

        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask

    def forward_encoder(self, x):
        return super().forward(x)

    def forward_alignment_head(self, x):
        eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        x = nn.functional.normalize(x, dim=-1, p=2, eps=eps)
        alignment_x = self.alignment_head(x)
        return alignment_x

    def forward_retention_head(self, x, mask_ratio):
        retention_x = self.retention_embed(x)
        retention_x, mask = self.random_masking(retention_x, mask_ratio)
        retention_x = retention_x + self.retention_gene_embed
        for blk in self.retention_blocks:
            retention_x = blk(retention_x)
        retention_x = self.retention_norm(retention_x)
        retention_x = self.retention_head(retention_x)
        return retention_x, mask

    def forward_decoders(self, x, mask_ratio):
        alignment_x = self.forward_alignment_head(x)
        retention_x, mask = self.forward_retention_head(x, mask_ratio)
        return alignment_x, retention_x, mask

    def forward(self, x, mask_ratio=0.75):
        x = self.forward_encoder(x)
        alignment_x, retention_x, mask = self.forward_decoders(x, mask_ratio)
        retention_target_x = x
        return alignment_x, retention_x, retention_target_x, mask


# ===========================================
#  TransMIL for Pre-training
# ===========================================
class FeatureTransMILHybrid(FeatureTransMIL):
    def __init__(
        self,
        input_dim=1024,
        embed_dim=512,
        num_tokens=2048,
        retention_decoder_depth=1,
    ):
        super().__init__(input_dim, embed_dim)

        self.num_tokens = num_tokens

        self.alignment_head = nn.Linear(embed_dim, embed_dim)

        self.retention_embed = nn.Linear(embed_dim, embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.retention_gene_embed = nn.Parameter(
            torch.randn(1, num_tokens + 1, embed_dim) * 0.02
        )
        self.retention_blocks = nn.ModuleList(
            [TransLayer(dim=embed_dim) for _ in range(retention_decoder_depth)]
        )
        self.retention_norm = nn.LayerNorm(embed_dim)
        self.retention_head = nn.Linear(embed_dim, embed_dim)

        self.init_weights()

    def init_weights(self):
        torch.nn.init.normal_(self.mask_token, std=0.02)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        trunc_normal_(self.retention_gene_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, h, mask_ratio):
        B, N, C = h.shape  # noqa: N806
        len_keep = int(N * (1 - mask_ratio))

        noise = torch.rand(B, N, device=h.device)

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        h_masked = torch.gather(h, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, C))
        mask_tokens = self.mask_token.repeat(
            B, ids_restore.shape[1] - h_masked.shape[1], 1
        )
        h_masked = torch.cat([h_masked, mask_tokens], dim=1)
        h_masked = torch.gather(
            h_masked, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, C)
        )

        mask = torch.ones([B, N], device=h.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return h_masked, mask

    def forward_encoder(self, h):
        h = h.float()  # [B, n, 1024]

        h = self._fc1(h)  # [B, n, 512]

        # ---->pad
        H = h.shape[1]  # noqa: N806
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))  # noqa: N806
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, 512]

        # ---->cls_token
        B = h.shape[0]  # noqa: N806
        cls_tokens = self.cls_token.expand(B, -1, -1)
        h = torch.cat((cls_tokens, h), dim=1)

        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 512]

        # ---->PPEG
        h = self.pos_layer(h, _H, _W)  # [B, N, 512]

        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512]

        # ---->cls_token
        h = self.norm(h)

        return h[:, : h.shape[1] - add_length, :]

    def forward_alignment_head(self, h):
        eps = 1e-6 if h.dtype == torch.float16 else 1e-12
        h = nn.functional.normalize(h, dim=-1, p=2, eps=eps)
        alignment_h = self.alignment_head(h[:, 0, :])
        return alignment_h

    def forward_retention_head(self, h, mask_ratio):
        retention_h = self.retention_embed(h)
        retention_h_, mask = self.random_masking(retention_h[:, 1:, :], mask_ratio)
        retention_h = torch.cat([retention_h[:, :1, :], retention_h_], dim=1)
        retention_h = retention_h + self.retention_gene_embed
        for blk in self.retention_blocks:
            retention_h = blk(retention_h)
        retention_h = self.retention_norm(retention_h)
        retention_h = self.retention_head(retention_h)
        retention_h = retention_h[:, 1:, :]
        return retention_h, mask

    def forward_decoders(self, h, mask_ratio):
        alignment_h = self.forward_alignment_head(h)
        retention_h, mask = self.forward_retention_head(h, mask_ratio)
        return alignment_h, retention_h, mask

    def forward(self, h, mask_ratio=0.75):
        h = self.forward_encoder(h)
        alignment_h, retention_h, mask = self.forward_decoders(h, mask_ratio)
        retention_target_h = h[:, 1:, :]
        return alignment_h, retention_h, retention_target_h, mask


# ===========================================
#  MIRROR for Pre-training
# ===========================================
class MIRROR(nn.Module):
    def __init__(
        self,
        wsi_embed_dim,
        rna_embed_dim,
        embed_dim,
        wsi_num_tokens=2048,
        wsi_retention_decoder_depth=1,
        rna_encoder_depth=2,
        rna_gene_embed="learn",
        rna_mlp_ratio=2.572,
        rna_pos_drop_rate=0.0,
        rna_proj_drop_rate=0.1,
        rna_attn_drop_rate=0.0,
        rna_drop_path_rate=0.0,
        rna_norm_layer=None,
        rna_act_layer=None,
        rna_retention_decoder_depth=1,
        init_logit_scale=np.log(1 / 0.07),  # noqa: B008
        style_mlp_hidden_dim=512,
        style_mlp_out_dim=256,
        style_norm_layer=None,
        style_act_layer=None,
        style_latent_dim=128,
        num_prototypes=3000,
    ):
        super().__init__()

        self.wsi_embed_dim = wsi_embed_dim
        self.rna_embed_dim = rna_embed_dim
        self.embed_dim = embed_dim
        self.wsi_num_tokens = wsi_num_tokens
        self.wsi_retention_decoder_depth = wsi_retention_decoder_depth
        self.rna_encoder_depth = rna_encoder_depth
        self.rna_gene_embed = rna_gene_embed
        self.rna_mlp_ratio = rna_mlp_ratio
        self.rna_pos_drop_rate = rna_pos_drop_rate
        self.rna_proj_drop_rate = rna_proj_drop_rate
        self.attn_drop_rate = rna_attn_drop_rate
        self.drop_path_rate = rna_drop_path_rate
        self.rna_norm_layer = rna_norm_layer
        self.rna_act_layer = rna_act_layer
        self.rna_retention_decoder_depth = rna_retention_decoder_depth

        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)

        self.wsi_encoder = FeatureTransMILHybrid(
            input_dim=self.wsi_embed_dim,
            embed_dim=self.embed_dim,
            num_tokens=self.wsi_num_tokens,
            retention_decoder_depth=self.wsi_retention_decoder_depth,
        )

        self.rna_encoder = TransFormerHybrid(
            input_dim=self.rna_embed_dim,
            embed_dim=self.embed_dim,
            depth=self.rna_encoder_depth,
            gene_embed=self.rna_gene_embed,
            mlp_ratio=self.rna_mlp_ratio,
            pos_drop_rate=self.rna_pos_drop_rate,
            proj_drop_rate=self.rna_proj_drop_rate,
            attn_drop_rate=self.attn_drop_rate,
            drop_path_rate=self.drop_path_rate,
            norm_layer=self.rna_norm_layer,
            act_layer=self.rna_act_layer,
            retention_decoder_depth=self.rna_retention_decoder_depth,
        )

        style_act_layer = get_act_layer(style_act_layer) or nn.GELU

        self.style_encoder_mlp = Mlp(
            in_features=embed_dim,
            hidden_features=style_mlp_hidden_dim,
            out_features=style_mlp_out_dim,
            act_layer=style_act_layer,
            norm_layer=style_norm_layer,
            drop=0.0,
        )
        self.style_mu = nn.Linear(style_mlp_out_dim, style_latent_dim)
        self.style_logstd = nn.Linear(style_mlp_out_dim, style_latent_dim)
        self.style_decoder = nn.Linear(style_latent_dim, embed_dim)

        self.prototypes = nn.Linear(embed_dim, num_prototypes, bias=False)
        nn.init.orthogonal_(self.prototypes.weight)

    def reparameterize(self, mu, logstd):
        std = torch.exp(0.5 * logstd)
        dist = Normal(mu, std)
        return dist.rsample()

    def forward_style_clustering(self, wsi_emb, rna_emb):
        wsi_emb = self.style_encoder_mlp(wsi_emb)
        wsi_mu = self.style_mu(wsi_emb)
        wsi_logstd = self.style_logstd(wsi_emb)
        wsi_z = self.reparameterize(wsi_mu, wsi_logstd)
        wsi_z = self.style_decoder(wsi_z)
        wsi_score = self.prototypes(wsi_z)

        rna_emb = self.style_encoder_mlp(rna_emb)
        rna_mu = self.style_mu(rna_emb)
        rna_logstd = self.style_logstd(rna_emb)
        rna_z = self.reparameterize(rna_mu, rna_logstd)
        rna_z = self.style_decoder(rna_z)
        rna_score = self.prototypes(rna_z)
        return wsi_score, wsi_mu, wsi_logstd, rna_score, rna_mu, rna_logstd

    def forward(self, wsi_emb, rna_emb, wsi_mask_ratio=0.75, rna_mask_ratio=0.75):
        wsi_emb = self.wsi_encoder.forward_encoder(wsi_emb)
        wsi_alignment_emb, wsi_retention_emb, wsi_mask = (
            self.wsi_encoder.forward_decoders(wsi_emb, mask_ratio=wsi_mask_ratio)
        )
        wsi_retention_target = wsi_emb[:, 1:, :]

        rna_emb = self.rna_encoder.forward_encoder(rna_emb)
        rna_alignment_emb, rna_retention_emb, rna_mask = (
            self.rna_encoder.forward_decoders(rna_emb, mask_ratio=rna_mask_ratio)
        )
        rna_retention_target = rna_emb

        wsi_score, wsi_mu, wsi_logstd, rna_score, rna_mu, rna_logstd = (
            self.forward_style_clustering(wsi_emb[:, 0, :], rna_emb)
        )

        return (
            wsi_alignment_emb,
            wsi_retention_emb,
            wsi_retention_target,
            wsi_mask,
            wsi_score,
            wsi_mu,
            wsi_logstd,
            rna_alignment_emb,
            rna_retention_emb,
            rna_retention_target,
            rna_mask,
            rna_score,
            rna_mu,
            rna_logstd,
            self.logit_scale.exp(),
        )


# ===========================================
#  MIRROR for Downstream Task
# ===========================================
class MIRRORClassifier(nn.Module):
    def __init__(
        self,
        wsi_embed_dim,
        rna_embed_dim,
        embed_dim,
        num_classes,
        rna_encoder_depth=2,
        rna_gene_embed="learn",
        rna_mlp_ratio=2.572,
        rna_pos_drop_rate=0.0,
        rna_proj_drop_rate=0.1,
        rna_attn_drop_rate=0.0,
        rna_drop_path_rate=0.0,
        rna_norm_layer=None,
        rna_act_layer=None,
        fusion="concat",
    ):
        super().__init__()

        self.wsi_embed_dim = wsi_embed_dim
        self.rna_embed_dim = rna_embed_dim
        self.embed_dim = embed_dim
        self.rna_encoder_depth = rna_encoder_depth
        self.rna_gene_embed = rna_gene_embed
        self.rna_mlp_ratio = rna_mlp_ratio
        self.rna_pos_drop_rate = rna_pos_drop_rate
        self.rna_proj_drop_rate = rna_proj_drop_rate
        self.attn_drop_rate = rna_attn_drop_rate
        self.drop_path_rate = rna_drop_path_rate
        self.rna_norm_layer = rna_norm_layer
        self.rna_act_layer = rna_act_layer
        self.num_classes = num_classes
        self.fusion = fusion
        assert self.fusion in ["add", "concat"], "Fusion must be either add or concat"

        self.wsi_encoder = FeatureTransMIL(
            input_dim=self.wsi_embed_dim,
            embed_dim=self.embed_dim,
        )

        self.rna_encoder = TransFormer(
            input_dim=self.rna_embed_dim,
            embed_dim=self.embed_dim,
            depth=self.rna_encoder_depth,
            gene_embed=self.rna_gene_embed,
            mlp_ratio=self.rna_mlp_ratio,
            pos_drop_rate=self.rna_pos_drop_rate,
            proj_drop_rate=self.rna_proj_drop_rate,
            attn_drop_rate=self.attn_drop_rate,
            drop_path_rate=self.drop_path_rate,
            norm_layer=self.rna_norm_layer,
            act_layer=self.rna_act_layer,
        )

        if self.fusion == "add" or self.fusion == "none":
            self.head = nn.Linear(self.embed_dim, self.num_classes)
        elif self.fusion == "concat":
            self.head = nn.Linear(self.embed_dim * 2, self.num_classes)

    def forward(self, wsi_emb, rna_emb=None):
        wsi_emb = self.wsi_encoder(wsi_emb)
        if rna_emb is not None:
            rna_emb = self.rna_encoder(rna_emb)

        fused_emb = None
        if rna_emb is not None:
            if self.fusion == "add":
                fused_emb = wsi_emb + rna_emb
            elif self.fusion == "concat":
                fused_emb = torch.cat((wsi_emb, rna_emb), dim=1)
        if fused_emb is not None:
            pred = self.head(fused_emb)
        else:
            pred = self.head(wsi_emb)
        return pred


@register_model
def mirror(**kwargs):
    accepted_args = {
        "wsi_embed_dim",
        "rna_embed_dim",
        "embed_dim",
        "wsi_num_tokens",
        "wsi_retention_decoder_depth",
        "rna_encoder_depth",
        "rna_gene_embed",
        "rna_mlp_ratio",
        "rna_pos_drop_rate",
        "rna_proj_drop_rate",
        "rna_attn_drop_rate",
        "rna_drop_path_rate",
        "rna_norm_layer",
        "rna_act_layer",
        "rna_retention_decoder_depth",
        "init_logit_scale",
        "style_mlp_hidden_dim",
        "style_mlp_out_dim",
        "style_norm_layer",
        "style_act_layer",
        "style_latent_dim",
        "num_prototypes",
    }

    filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted_args}
    filtered_out_args = {k: v for k, v in kwargs.items() if k not in accepted_args}

    if filtered_out_args:
        _logger.warning(
            "Filtered model kwargs: %s", ", ".join(filtered_out_args.keys())
        )

    return MIRROR(**filtered_kwargs)


@register_model
def mirror_classifier(**kwargs):
    accepted_args = {
        "wsi_embed_dim",
        "rna_embed_dim",
        "embed_dim",
        "rna_encoder_depth",
        "rna_gene_embed",
        "rna_mlp_ratio",
        "rna_pos_drop_rate",
        "rna_proj_drop_rate",
        "rna_attn_drop_rate",
        "rna_drop_path_rate",
        "rna_norm_layer",
        "rna_act_layer",
        "num_classes",
        "fusion",
    }

    filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted_args}
    filtered_out_args = {k: v for k, v in kwargs.items() if k not in accepted_args}

    if filtered_out_args:
        _logger.warning(
            "Filtered model kwargs: %s", ", ".join(filtered_out_args.keys())
        )

    return MIRRORClassifier(**filtered_kwargs)
