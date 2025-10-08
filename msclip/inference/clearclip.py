# msclip/inference/clearclip.py
import torch
import torch.nn as nn
from typing import List, Tuple

class _Identity(nn.Module):
    def forward(self, x): return x

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

            if self.keep_residual:
                x = x + attn_out
            else:
                print("-------------------> Ca Clear Clip a balle")
                x = attn_out

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
    def _as_modulelist(lst): return nn.ModuleList(lst)

    if key == 'timm.blocks' and hasattr(vision_encoder, 'blocks'):
        vision_encoder.blocks = _as_modulelist(new_blocks); return

    parts = key.split('.')
    obj = vision_encoder
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], _as_modulelist(new_blocks))

def apply_clearclip(vision_encoder: nn.Module, *, apply_to_last_n: int = 1,
                    keep_ffn: bool = False, keep_residual: bool = False) -> int:
    """
    Replace the last N ViT blocks with ClearCLIP-wrapped versions.
    Returns number of blocks patched.
    """
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

def maybe_patch_clearclip(vision_encoder: nn.Module, cfg: dict) -> int:
    """
    Convenience: cfg should look like
    {
      "enabled": true,
      "apply_to_last_n": 1,
      "keep_ffn": false,
      "keep_residual": false
    }
    """
    if not cfg or not cfg.get("enabled", False):
        return 0
    return apply_clearclip(
        vision_encoder,
        apply_to_last_n=int(cfg.get("apply_to_last_n", 1)),
        keep_ffn=bool(cfg.get("keep_ffn", False)),
        keep_residual=bool(cfg.get("keep_residual", False)),
    )
