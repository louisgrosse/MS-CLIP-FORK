# msclip/inference/clearclip.py
import math
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------
# Utilities (OpenAI/OpenCLIP compatible)
# -----------------------
def _as_modulelist(lst): return nn.ModuleList(lst)

def _find_vit_blocks(vision_encoder: nn.Module) -> Tuple[List[nn.Module], str]:
    """
    Try several common attribute paths to fetch the list of transformer blocks.
    Returns (blocks_list, key) where key is a string describing which path matched.
    """
    if hasattr(vision_encoder, 'blocks') and isinstance(vision_encoder.blocks, nn.ModuleList):
        return list(vision_encoder.blocks), 'timm.blocks'

    for name in ['transformer', 'encoder', 'trunk', 'visual', 'model', 'vision_model']:
        mod = getattr(vision_encoder, name, None)
        if mod is None:
            continue

        if hasattr(mod, 'resblocks'):
            blocks = getattr(mod, 'resblocks')
            if isinstance(blocks, (list, nn.ModuleList, tuple)):
                return list(blocks), f'{name}.resblocks'

        if hasattr(mod, 'encoder') and hasattr(mod.encoder, 'layers'):
            layers = mod.encoder.layers
            if isinstance(layers, (list, nn.ModuleList, tuple)):
                return list(layers), f'{name}.encoder.layers'

        if hasattr(mod, 'blocks') and isinstance(mod.blocks, nn.ModuleList):
            return list(mod.blocks), f'{name}.blocks'

    if hasattr(vision_encoder, 'layers') and isinstance(vision_encoder.layers, nn.ModuleList):
        return list(vision_encoder.layers), 'layers'

    raise RuntimeError("Could not locate transformer blocks in the provided vision encoder.")

def _set_blocks(vision_encoder: nn.Module, new_blocks: List[nn.Module], key: str):
    """Write back the patched blocks to the encoder based on the discovered key."""
    if key == 'timm.blocks' and hasattr(vision_encoder, 'blocks'):
        vision_encoder.blocks = _as_modulelist(new_blocks); return
    parts = key.split('.')
    obj = vision_encoder
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], _as_modulelist(new_blocks))

# -----------------------
# (A) Your original ClearCLIP block (residual / FFN control)
# -----------------------
class ClearCLIPBlock(nn.Module):
    """
    Wrap a ViT encoder block to optionally:
      - remove residual connection around attention
      - drop the FFN (MLP) entirely
    Keep the original attention stack (Q=K=V from x, i.e., vanilla self-attn).
    """
    def __init__(self, orig_block: nn.Module, keep_ffn: bool, keep_residual: bool):
        super().__init__()
        self.orig = orig_block
        self.keep_ffn = keep_ffn
        self.keep_residual = keep_residual

        self.has_pre_norm_attn = hasattr(orig_block, 'norm1') and hasattr(orig_block, 'attn')
        self.has_post_mlp     = hasattr(orig_block, 'mlp') and hasattr(orig_block, 'norm2')

        if not self.has_pre_norm_attn:
            # OpenAI/OpenCLIP style
            self.ln_1 = getattr(orig_block, 'ln_1', None)
            self.attn = getattr(orig_block, 'attn', None)
            self.mlp  = getattr(orig_block, 'mlp',  None)
            self.ln_2 = getattr(orig_block, 'ln_2', None)
            self.oai_style = all([self.ln_1 is not None, self.attn is not None])
        else:
            self.oai_style = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.has_pre_norm_attn:
            h = self.orig.norm1(x)
            attn_out = self.orig.attn(h)
            if isinstance(attn_out, tuple):
                attn_out = attn_out[0]
            x = (x + attn_out) if self.keep_residual else attn_out
            if self.keep_ffn and self.has_post_mlp:
                x = x + self.orig.mlp(self.orig.norm2(x))
            return x

        if self.oai_style:
            h = self.ln_1(x)
            attn_out = self.attn(h)
            if isinstance(attn_out, tuple):
                attn_out = attn_out[0]
            x = (x + attn_out) if self.keep_residual else attn_out
            if self.keep_ffn and (self.mlp is not None) and (self.ln_2 is not None):
                x = x + self.mlp(self.ln_2(x))
            return x

        return self.orig(x)

# -----------------------
# (B) ClearCLIP + SSA (GEM-like) building blocks
# -----------------------
# Minimal helpers taken/adapted from ClearCLIP to support OpenCLIP VisionTransformer
def _expand_token(t: torch.Tensor, batch_size: int):
    return t.unsqueeze(0).repeat(batch_size, 1, 1)

def _to_2tuple(x):
    return (x, x) if isinstance(x, int) else tuple(x)

def resample_abs_pos_embed(
        posemb: torch.Tensor,
        new_size: List[int],
        old_size: Optional[List[int]] = None,
        num_prefix_tokens: int = 1,
        interpolation: str = 'bicubic',
        antialias: bool = True
):
    new_size = _to_2tuple(new_size)
    new_ntok = new_size[0] * new_size[1]
    if not old_size:
        old_size = int(math.sqrt(posemb.shape[1] - num_prefix_tokens))
    old_size = _to_2tuple(old_size)
    if new_size == old_size:
        return posemb

    if num_prefix_tokens:
        posemb_prefix, posemb = posemb[:, :num_prefix_tokens], posemb[:, num_prefix_tokens:]
    else:
        posemb_prefix, posemb = None, posemb

    posemb = posemb.reshape(1, old_size[0], old_size[1], -1).permute(0, 3, 1, 2)
    posemb = F.interpolate(posemb, size=new_size, mode=interpolation, antialias=antialias)
    posemb = posemb.permute(0, 2, 3, 1).reshape(1, new_ntok, -1)

    if posemb_prefix is not None:
        posemb = torch.cat([posemb_prefix, posemb], dim=1)
    return posemb

class SelfSelfAttention(nn.Module):
    """
    ClearCLIP/GEM Self-Self Attention with (q,k,v)-space iterative refinement.
    We will copy qkv/proj weights from the original attn to keep it zero-shot.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., ss_attn_iter=1, ss_attn_temp=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.ss_attn_iter = ss_attn_iter
        self.ss_attn_temp = ss_attn_temp

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_bias=None, prev_attn=None):
        # expect OpenAI/OpenCLIP attn API: return [x_gem, x_ori] in LND (seq-first) layout
        x = x.transpose(0, 1)  # LND-> B N D
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # original attention path
        attn_ori = (q @ k.transpose(-2, -1)) * self.scale
        attn_ori = attn_ori.softmax(dim=-1)
        attn_ori = self.attn_drop(attn_ori)
        x_ori = (attn_ori @ v).transpose(1, 2).reshape(B, N, C)
        x_ori = self.proj_drop(self.proj(x_ori))

        # GEM refinement on q/k/v spaces
        xs1, xs2, xs3 = v, k, q
        if self.ss_attn_temp is None:
            pre_norm = torch.norm(x, dim=-1).mean(dim=-1, keepdim=True).unsqueeze(1).unsqueeze(-1)
            inv_temp = pre_norm * self.scale
        else:
            inv_temp = self.ss_attn_temp

        for _ in range(self.ss_attn_iter):
            xs1 = F.normalize(xs1, dim=-1)
            xs2 = F.normalize(xs2, dim=-1)
            xs3 = F.normalize(xs3, dim=-1)

            attn1 = (xs1 @ xs1.transpose(-2, -1)) * inv_temp
            attn2 = (xs2 @ xs2.transpose(-2, -1)) * inv_temp
            attn3 = (xs3 @ xs3.transpose(-2, -1)) * inv_temp

            xs1 = attn1.softmax(dim=-1) @ xs1
            xs2 = attn2.softmax(dim=-1) @ xs2
            xs3 = attn3.softmax(dim=-1) @ xs3

        xs1 = F.normalize(xs1, dim=-1)
        xs2 = F.normalize(xs2, dim=-1)
        xs3 = F.normalize(xs3, dim=-1)

        attn1 = (xs1 @ xs1.transpose(-2, -1)) * inv_temp
        attn2 = (xs2 @ xs2.transpose(-2, -1)) * inv_temp
        attn3 = (xs3 @ xs3.transpose(-2, -1)) * inv_temp

        xs = (attn1.softmax(dim=-1) @ v + attn2.softmax(dim=-1) @ v + attn3.softmax(dim=-1) @ v) / 3.0
        x_gem = xs.transpose(1, 2).reshape(B, N, C)
        x_gem = self.proj_drop(self.proj(x_gem))

        # return in LND with both branches
        return [x_gem.transpose(0, 1), x_ori.transpose(0, 1)]

class GEMResidualBlock(nn.Module):
    """
    Wrap an OpenAI/OpenCLIP ViT block whose .attn returns [x_gem, x_ori] (both LND),
    and combine with (optionally) disabled residual path for x_gem.
    """
    def __init__(self, res_block, ignore_residual: bool):
        super().__init__()
        self.res_block = res_block
        self.ignore_residual = ignore_residual

        # Resolve naming differences (OpenAI vs timm)
        self.ln_1 = getattr(res_block, 'ln_1', getattr(res_block, 'norm1', None))
        self.ln_2 = getattr(res_block, 'ln_2', getattr(res_block, 'norm2', None))
        self.mlp  = getattr(res_block, 'mlp', None)
        # layer scale optional (OpenCLIP ViT has ls_1/ls_2 sometimes)
        self.ls_1 = getattr(res_block, 'ls_1', nn.Identity())
        self.ls_2 = getattr(res_block, 'ls_2', nn.Identity())

    def forward(self, x: torch.Tensor):
        # Expect LND (seq-first) going through OpenAI CLIP blocks; if NLD comes in, permute earlier.
        x_norm = self.ln_1(x)
        x_gem_res, x_ori_res = self.res_block.attn(x=x_norm)  # both LND
        x_gem_res, x_ori_res = self.ls_1(x_gem_res), self.ls_1(x_ori_res)

        # Original branch keeps residuals & MLP
        x_ori = x + x_ori_res
        if self.mlp is not None and self.ln_2 is not None:
            x_ori = x_ori + self.ls_2(self.mlp(self.ln_2(x_ori)))

        # GEM branch can drop residual if requested
        x_gem = x_gem_res if self.ignore_residual else (x + x_gem_res)
        return [x_gem, x_ori]

# Optional: forward override for OpenCLIP VisionTransformer so .forward returns patch embeddings (NLD)
def _maybe_patch_visual_forward_to_gem(visual: nn.Module) -> bool:
    """
    Monkey-patch OpenCLIP VisionTransformer.forward so it returns patch embeddings (NLD)
    using GEM dual branch from the transformer (LND in / NLD out). Safe no-op if not OpenCLIP style.
    """
    required = all([hasattr(visual, 'conv1'),
                    hasattr(visual, 'positional_embedding'),
                    hasattr(visual, 'transformer'),
                    hasattr(visual, 'ln_pre')])

    if not required:
        return False

    def modified_vit_forward(self, x: torch.Tensor):
        # x: [B, 3, H, W]
        x = self.conv1(x)  # [B, C, gh, gw]
        grid_h, grid_w = x.shape[2:]
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # [B, gh*gw, C]

        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)

        if x.shape[1] != self.positional_embedding.shape[1]:
            pos_emb = resample_abs_pos_embed(self.positional_embedding.unsqueeze(0),
                                             new_size=[grid_h, grid_w],
                                             num_prefix_tokens=1,
                                             interpolation='bicubic',
                                             antialias=True)
        else:
            pos_emb = self.positional_embedding

        x = x + pos_emb.to(x.dtype)
        if hasattr(self, 'patch_dropout'):
            x = self.patch_dropout(x)
        x = self.ln_pre(x)

        # NLD -> LND for transformer
        x = x.permute(1, 0, 2)

        # Expect transformer to return [x_gem, x_ori] (both LND) when SSA wrapper is installed.
        x_gem, x_ori = self.transformer(x)
        x_gem = x_gem.permute(1, 0, 2)  # NLD
        x_ori = x_ori.permute(1, 0, 2)  # NLD

        # post
        x_gem = self.ln_post(x_gem)
        x_ori = self.ln_post(x_ori)
        if self.proj is not None:
            x_gem = x_gem @ self.proj
            x_ori = x_ori @ self.proj

        # Return patch embeddings (drop CLS) as in GEM usage
        return x_gem[:, 1:, :]

    # bind
    visual.forward = modified_vit_forward.__get__(visual, visual.__class__)
    return True

# -----------------------
# (C) ATTENTION ADAPTERS
# -----------------------
def _extract_attn_dims(attn: nn.Module) -> Tuple[int, int]:
    """
    Returns (dim, num_heads) for common CLIP/timm attn modules.
    """
    num_heads = getattr(attn, 'num_heads', None)
    head_dim  = getattr(attn, 'head_dim', None)

    if num_heads is not None and head_dim is not None:
        return head_dim * num_heads, num_heads

    # timm style: qkv is Linear(dim, 3*dim)
    if hasattr(attn, 'qkv') and isinstance(attn.qkv, nn.Linear):
        dim = attn.qkv.out_features // 3
        num_heads = getattr(attn, 'num_heads', 8)
        return dim, num_heads

    # OpenAI style: in_proj_weight: [3*dim, dim]
    if hasattr(attn, 'in_proj_weight'):
        dim = attn.in_proj_weight.shape[1]
        num_heads = getattr(attn, 'num_heads', 8)
        return dim, num_heads

    raise RuntimeError("Could not infer attention dimensions.")

def _copy_qkv_proj_to_ssa(src_attn: nn.Module, dst_ssa: SelfSelfAttention):
    """
    Copy qkv/proj weights from original attention to SSA for zero-shot behavior.
    Supports OpenAI (in_proj_weight/out_proj) and timm (qkv/proj).
    """
    if hasattr(src_attn, 'in_proj_weight') and hasattr(src_attn, 'out_proj'):
        # OpenAI/OpenCLIP (torch.nn.MultiheadAttention-like wrapper)
        with torch.no_grad():
            dst_ssa.qkv.weight.copy_(src_attn.in_proj_weight)
            if getattr(src_attn, 'in_proj_bias', None) is not None:
                dst_ssa.qkv.bias.copy_(src_attn.in_proj_bias)
            dst_ssa.proj.weight.copy_(src_attn.out_proj.weight)
            if getattr(src_attn.out_proj, 'bias', None) is not None:
                dst_ssa.proj.bias.copy_(src_attn.out_proj.bias)
        return

    if hasattr(src_attn, 'qkv') and hasattr(src_attn, 'proj'):
        # timm ViT attention
        with torch.no_grad():
            dst_ssa.qkv.weight.copy_(src_attn.qkv.weight)
            if getattr(src_attn.qkv, 'bias', None) is not None:
                dst_ssa.qkv.bias.copy_(src_attn.qkv.bias)
            dst_ssa.proj.weight.copy_(src_attn.proj.weight)
            if getattr(src_attn.proj, 'bias', None) is not None:
                dst_ssa.proj.bias.copy_(src_attn.proj.bias)
        return

    raise RuntimeError("Unsupported attention module for weight copy.")

# -----------------------
# (D) Patchers
# -----------------------
def apply_clearclip_residual(vision_encoder: nn.Module, *, apply_to_last_n: int, keep_ffn: bool, keep_residual: bool) -> int:
    """Your original ClearCLIP patcher (no SSA)."""
    blocks, key = _find_vit_blocks(vision_encoder)
    n = len(blocks)
    if n == 0 or apply_to_last_n <= 0:
        return 0
    k = min(apply_to_last_n, n)
    patched = list(blocks)
    for i in range(n - k, n):
        patched[i] = ClearCLIPBlock(blocks[i], keep_ffn=keep_ffn, keep_residual=keep_residual)
    _set_blocks(vision_encoder, patched, key)
    return k

def apply_clearclip_ssa(vision_encoder: nn.Module, *, apply_to_last_n: int,
                        ignore_residual: bool, ss_attn_iter: int = 1, ss_attn_temp: Optional[float] = None) -> int:
    """
    GEM-like patcher:
      - replace .attn with SelfSelfAttention (weights copied from original attn)
      - wrap block with GEMResidualBlock (controls residual on GEM branch)
      - make transformer return [x_gem, x_ori] along the layers
    """
    blocks, key = _find_vit_blocks(vision_encoder)
    n = len(blocks)
    if n == 0 or apply_to_last_n <= 0:
        return 0
    k = min(apply_to_last_n, n)
    patched = list(blocks)

    for i in range(n - k, n):
        blk = blocks[i]
        attn = getattr(blk, 'attn', None)
        if attn is None:
            raise RuntimeError("Selected block has no .attn")

        dim, num_heads = _extract_attn_dims(attn)
        ssa = SelfSelfAttention(dim=dim, num_heads=num_heads, qkv_bias=True,
                                ss_attn_iter=ss_attn_iter, ss_attn_temp=ss_attn_temp)
        _copy_qkv_proj_to_ssa(attn, ssa)

        # swap attention
        blk.attn = ssa

        patched[i] = GEMResidualBlock(blk, ignore_residual=ignore_residual)

    _set_blocks(vision_encoder, patched, key)
    return k


def maybe_patch_clearclip(vision_encoder: nn.Module, cfg: dict) -> int:
    """
    cfg example:
    clearclip:
      enabled: true
      apply_to_last_n: 7
      keep_ffn: false
      keep_residual: false
      use_self_self_attn: true
      ss_attn_iter: 1
      ss_attn_temp: null
      override_forward_for_dense: true
    """
    if not cfg or not cfg.get("enabled", False):
        return 0

    apply_to_last_n     = int(cfg.get("apply_to_last_n", 1))
    keep_ffn            = bool(cfg.get("keep_ffn", False))
    keep_residual       = bool(cfg.get("keep_residual", False))
    use_ssa             = bool(cfg.get("use_self_self_attn", False))
    ss_attn_iter        = int(cfg.get("ss_attn_iter", 1))
    ss_attn_temp        = cfg.get("ss_attn_temp", None)
    override_forward    = bool(cfg.get("override_forward_for_dense", False))

    if use_ssa:
        patched = apply_clearclip_ssa(
            vision_encoder,
            apply_to_last_n=apply_to_last_n,
            ignore_residual=not keep_residual,            
            ss_attn_iter=ss_attn_iter,
            ss_attn_temp=ss_attn_temp
        )
        if override_forward:
            _maybe_patch_visual_forward_to_gem(vision_encoder)
        return patched

    return apply_clearclip_residual(
        vision_encoder,
        apply_to_last_n=apply_to_last_n,
        keep_ffn=keep_ffn,
        keep_residual=keep_residual
    )
