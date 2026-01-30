# code adapted from https://github.com/mrabiabrn/robustbev
#

import math
import numpy as np
import torch
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.vision_transformer import VisionTransformer
from einops import rearrange
from safetensors import safe_open
from safetensors.torch import save_file
from .grid_mask import GridMask
from navsim.agents.drivoR.utils import pylogger
log = pylogger.get_pylogger(__name__)

class timm_ViT(VisionTransformer):

    def _pos_embed(self, x: torch.Tensor, scene_tokens: torch.Tensor = None) -> torch.Tensor:
        """Apply positional embedding to input."""
        if self.pos_embed is None:
            return x.view(x.shape[0], -1, x.shape[-1])

        if self.dynamic_img_size:
            B, H, W, C = x.shape
            prev_grid_size = self.patch_embed.grid_size
            pos_embed = resample_abs_pos_embed(
                self.pos_embed,
                new_size=(H, W),
                old_size=prev_grid_size,
                num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
            )
            x = x.view(B, -1, C)
        else:
            pos_embed = self.pos_embed

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            x = x + pos_embed

        # concatenate the scene tokens
        if scene_tokens is not None:
            x = torch.cat([scene_tokens,x], dim=1)


        return self.pos_drop(x)


    def forward_features(self, x: torch.Tensor, scene_tokens: torch.Tensor = None, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass through feature layers (embeddings, transformer blocks, post-transformer norm)."""

        x = self.patch_embed(x)
        x = self._pos_embed(x, scene_tokens)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        if attn_mask is not None:
            # If mask provided, we need to apply blocks one by one
            for blk in self.blocks:
                x = blk(x, attn_mask=attn_mask)
        elif self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)

        x = self.norm(x)
        return x



class _LoRA_qkv_timm(nn.Module):
    """In timm it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    """
    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
        linear_a_k: nn.Module,
        linear_b_k: nn.Module,
        layer_norm_q: nn.Module = None,
        layer_norm_v: nn.Module = None,
        layer_norm_k: nn.Module = None,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.linear_a_k = linear_a_k
        self.linear_b_k = linear_b_k
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

        self.layernorm_q = layer_norm_q
        self.layernorm_v = layer_norm_v
        self.layernorm_k = layer_norm_k

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(self.layernorm_q(x)))
        new_v = self.linear_b_v(self.linear_a_v(self.layernorm_v(x)))
        #new_k = self.linear_b_k(self.linear_a_k(self.layernorm_k(x)))
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim :] += new_v
        #qkv[:, :, self.dim : 2 * self.dim] += new_k
        return qkv


class LoRA_ViT_timm(nn.Module):
    def __init__(self, vit_model: timm_ViT, r: int, lora_layer=None, use_layer_norm=False, use_qkv=False):
        super(LoRA_ViT_timm, self).__init__()

        if r == 0:
            for param in vit_model.parameters():
                param.requires_grad = False
            self.lora_vit = vit_model
            
        else:
            if lora_layer:
                self.lora_layer = lora_layer
            else:
                self.lora_layer = list(range(len(vit_model.blocks)))

            # dim = vit_model.head.in_features
            # create for storage, then we can init them or load weights
            self.w_As = []  # These are linear layers
            self.w_Bs = []

            # lets freeze first
            for param in vit_model.parameters():
                param.requires_grad = False

            # Here, we do the surgery
            for t_layer_i, blk in enumerate(vit_model.blocks):
                # If we only want few lora layer instead of all
                if t_layer_i not in self.lora_layer:
                    continue
                w_qkv_linear = blk.attn.qkv
                self.dim = w_qkv_linear.in_features
                w_a_linear_q = nn.Linear(self.dim, r, bias=False)
                w_b_linear_q = nn.Linear(r, self.dim, bias=False)
                w_a_linear_v = nn.Linear(self.dim, r, bias=False)
                w_b_linear_v = nn.Linear(r, self.dim, bias=False)
                w_a_linear_k = nn.Identity()
                w_b_linear_k = nn.Identity()
                if use_qkv:
                    w_a_linear_k = nn.Linear(self.dim, r, bias=False)
                    w_b_linear_k = nn.Linear(r, self.dim, bias=False)
                layer_norm_q = nn.Identity()
                layer_norm_v = nn.Identity()
                layer_norm_k = nn.Identity()
                if use_layer_norm:
                    layer_norm_q = nn.LayerNorm(self.dim)
                    layer_norm_v = nn.LayerNorm(self.dim)
                    if use_qkv:
                        layer_norm_k = nn.LayerNorm(self.dim)
                self.w_As.append(w_a_linear_q)
                self.w_Bs.append(w_b_linear_q)
                self.w_As.append(w_a_linear_v)
                self.w_Bs.append(w_b_linear_v)
                blk.attn.qkv = _LoRA_qkv_timm(
                    w_qkv_linear,
                    w_a_linear_q,
                    w_b_linear_q,
                    w_a_linear_v,
                    w_b_linear_v,
                    w_a_linear_k,
                    w_b_linear_k,
                    layer_norm_q,
                    layer_norm_v,
                    layer_norm_k,
                )
            self.reset_parameters()
            self.lora_vit = vit_model

    def save_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        
        save both lora and fc parameters.
        """

        assert filename.endswith(".safetensors")

        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        
        merged_dict = {**a_tensors, **b_tensors}
        save_file(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.\
            
        load both lora and fc parameters.
        """

        assert filename.endswith(".safetensors")

        with safe_open(filename, framework="pt") as f:
            for i, w_A_linear in enumerate(self.w_As):
                saved_key = f"w_a_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_A_linear.weight = Parameter(saved_tensor)

            for i, w_B_linear in enumerate(self.w_Bs):
                saved_key = f"w_b_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_B_linear.weight = Parameter(saved_tensor)
                

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, x: Tensor, scene_tokens: torch.Tensor = None) -> Tensor:
        return self.lora_vit.forward_features(x, scene_tokens)


class ImgEncoder(torch.nn.Module):
    """Extract embeddings from images using timm's Dinov2 models"""
    model_names = (
        "timm/vit_small_patch14_dinov2.lvd142m",
        "timm/vit_base_patch14_dinov2.lvd142m",
        "timm/vit_large_patch14_dinov2.lvd142m",
        "timm/vit_giant_patch14_dinov2.lvd142m",
        "timm/vit_small_patch14_reg4_dinov2.lvd142m",
        "timm/vit_base_patch14_reg4_dinov2.lvd142m",
        "timm/vit_large_patch14_reg4_dinov2.lvd142m",
        "timm/vit_giant_patch14_reg4_dinov2.lvd142m",
        "timm/vit_small_patch16_dinov3.lvd1689m",
        "timm/vit_large_patch16_dinov3.lvd1689m"
        )

    def __init__(self, 
                 config,
    ):
        super().__init__()


        model_name = config.model_name
        self.num_prefix_tokens = config.num_scene_tokens
        if model_name not in self.model_names:
            raise ValueError(f"Unknown model name: {repr(model_name)}")
        else:
            print("loading ", model_name)
        pretrained_cfg_overlay = {
            "file": config.model_weights,
        }
        
        in_chans = config.in_chans if "in_chans" in config else 3
        
        # HACK: to deal with new numpy version that does not allow pickle by default
        # Create a context manager to temporarily modify np.load
        np_load_old = np.load
        np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
        try:
            self.model = timm.create_model(
                model_name,
                pretrained=True,
                pretrained_cfg_overlay=pretrained_cfg_overlay,
                img_size=(config.image_size[1], config.image_size[0]),
                num_classes=0,
                in_chans=in_chans)
        except:
            self.model = timm.create_model(
                model_name,
                pretrained=True,
                img_size=(config.image_size[1], config.image_size[0]),
                num_classes=0,
                in_chans=in_chans)
        np.load = np_load_old
        
        self.model.__class__ = timm_ViT
        self.patch_size = self.model.patch_embed.patch_size[0]
        self.use_lora = config.use_lora
        self.finetune = config.finetune

        self.neck = torch.nn.Linear(self.model.num_features,config.tf_d_model)

        self.num_features = self.model.num_features

        # Adaptation Setting
        if self.use_lora:
            self.model = LoRA_ViT_timm(self.model, r=config.lora_rank)
        # Finetuning
        elif self.finetune:
            for param in self.model.parameters():
                param.requires_grad = True
            self.model.train()
        # Frozen
        else:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
            # self.model.train() # here we let it in train mode

        # train the patch embedder
        if in_chans != 3:
            log.info("Training patch embed and pos embed as in channels != 3")
            for name, param in self.model.named_parameters():
                if "patch_embed" in name:
                    param.requires_grad = True
                elif "pos_embed" in name:
                    param.requires_grad = True

        self.grid_mask = GridMask( True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = True

        # feature pooling
        self.use_feature_pooling = config.use_feature_pooling
        if self.use_feature_pooling:
            self.pool_proj = torch.nn.Sequential(
                torch.nn.AdaptiveAvgPool1d(self.num_prefix_tokens)
            )
        
        # focus front cam
        self.focus_front_cam = config.focus_front_cam
        self.compress_fc = config.compress_fc
        if self.compress_fc:
            self.compress_fc_layer = torch.nn.Linear(3957, self.num_prefix_tokens)


    
    # def forward(self, data_dict):
    def forward(self, img, scene_tokens):


        B, N, C, H, W = img.size()
        # print("img.shape ", img.shape)
        img = rearrange(img, 'b n c h w -> (b n) c h w')
        # img = img.reshape(B * N, C, H, W)
        if self.use_grid_mask:
            img = self.grid_mask(img)

        scene_tokens = rearrange(scene_tokens, 'b n t c -> (b n) t c')

        # model inference
        if self.use_lora:
            tokens = self.model(img, scene_tokens)
        elif self.finetune:
            tokens = self.model.forward_features(img, scene_tokens)
        else:
            # self.model.eval()
            # with torch.no_grad():
            tokens = self.model.forward_features(img, scene_tokens)

        if self.use_feature_pooling:
            B_, T, D = tokens.shape  # (B*N, num_tokens, dim)
            # Project the sequence of tokens to `num_prefix_tokens` summary tokens
            tokens = self.pool_proj(tokens.transpose(1, 2))  # shape: (B*N, D, T) → (B*N, D, num_prefix_tokens)
            # print("self.use_feature_pooling: ", self.use_feature_pooling)
            tokens = tokens.transpose(1, 2)  # → (B*N, num_prefix_tokens, D)
        elif self.focus_front_cam:
            B_, T, D = tokens.shape  # (B*N, num_tokens, dim)
            tokens = rearrange(tokens, '(b n) t c -> b n t c', b=B, n=N)
            # all front-cam tokens (camera index 0): [B, T, D]
            front =  tokens[:, 0,  :,                       :] 
            if self.compress_fc:
                # print("before front.shape ", front.shape)  
                front = self.compress_fc_layer(front.transpose(1,2)).transpose(1,2)
                # print("front.shape ", front.shape)                     
            # first K tokens from every other camera: [B, N-1, K, D] -> [B, (N-1)*K, D]
            others = tokens[:, 1:, :self.num_prefix_tokens, :].reshape(B, -1, D)
            # concatenate per batch, preserving order: front first, then cam1..camN-1 prefixes
            tokens = torch.cat([front, others], dim=1)  # [B, (N-1)*K, D]
        elif self.num_prefix_tokens > 0:
            tokens = tokens[:,:self.num_prefix_tokens]
        else:
            tokens = tokens

        tokens = self.neck(tokens)
        if not self.focus_front_cam:
            tokens = rearrange(tokens, '(b n) t c -> b (n t) c', b=B, n=N)

        return tokens