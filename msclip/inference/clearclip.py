from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

def _get_resblocks(image_encoder: nn.Module):
    """
    Return the ModuleList of ViT blocks in an OpenCLIP-style image encoder:
    image_encoder.model.visual.transformer.resblocks
    """
    return image_encoder.model.visual.transformer.resblocks

def _has_fsdpa():
    return hasattr(F, "scaled_dot_product_attention")


class ClearCLIPLastBlock(nn.Module):
    """
    Replacement for the LAST ViT block that implements ClearCLIP behavior:
      - uses the attention output only (X_attn)
      - optionally uses self-self attention (qq/kk/vv)
      - can drop residual and/or FFN
    It reuses the original block's LayerNorms, MHA, MLP, and LayerScales.
    """
    def __init__(
        self,
        src_block: nn.Module,
        attn_variant: str = "qq",         # "qk" (basic), "qq", "kk", "vv"
        keep_residual: bool = False,
        keep_ffn: bool = False,
    ):
        super().__init__()
        assert hasattr(src_block, "ln_1") and hasattr(src_block, "attn") and hasattr(src_block, "mlp") and hasattr(src_block, "ln_2")
        self.ln_1 = src_block.ln_1
        self.attn = src_block.attn                 # nn.MultiheadAttention
        self.ls_1 = getattr(src_block, "ls_1", nn.Identity())
        self.ln_2 = src_block.ln_2
        self.mlp  = src_block.mlp
        self.ls_2 = getattr(src_block, "ls_2", nn.Identity())

        self.attn_variant  = attn_variant.lower()
        self.keep_residual = keep_residual
        self.keep_ffn      = keep_ffn

        self.embed_dim  = self.attn.embed_dim
        self.num_heads  = self.attn.num_heads
        self.head_dim   = self.embed_dim // self.num_heads
        self.scale      = self.head_dim ** -0.5

    @torch.no_grad()
    def _has_in_proj(self):
        return hasattr(self.attn, "in_proj_weight") and self.attn.in_proj_weight is not None

    def _proj_qkv(self, x: torch.Tensor):
        """
        Project to q,k,v with the original MHA weights, but we'll override
        K/Q/V selections depending on attn_variant.
        """
        if self._has_in_proj():
            qkv = F.linear(x, self.attn.in_proj_weight, self.attn.in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)
        else:
            q = F.linear(x, self.attn.q_proj_weight, self.attn.in_proj_bias[:self.embed_dim])
            k = F.linear(x, self.attn.k_proj_weight, self.attn.in_proj_bias[self.embed_dim:2*self.embed_dim])
            v = F.linear(x, self.attn.v_proj_weight, self.attn.in_proj_bias[2*self.embed_dim:])

        if self.attn_variant == "qq":
            k = q
        elif self.attn_variant == "kk":
            q = k
        elif self.attn_variant == "vv":
            q = F.normalize(v, dim=-1)
            k = q

        return q, k, v

    def _attention_only(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        """
        Compute only the attention branch with the chosen self-self variant.
        x: (N, L, C) batch_first=True.
        Returns X_attn projected back to (N, L, C).
        """
        N, L, C = x.shape
        q, k, v = self._proj_qkv(x)  # (N, L, C)

        # -> (N, heads, L, head_dim)
        def _reshape(z):
            return z.view(N, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        q = _reshape(q)
        k = _reshape(k)
        v = _reshape(v)

        if _has_fsdpa():
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn.dropout if self.training else 0.0,
                is_causal=False,
            )
        else:
            attn = (q * self.scale) @ k.transpose(-2, -1)  # (N, heads, L, L)
            if attn_mask is not None:
                attn = attn + attn_mask
            attn = attn.softmax(dim=-1)
            if self.training and self.attn.dropout > 0:
                attn = F.dropout(attn, p=self.attn.dropout)
            out = attn @ v  # (N, heads, L, head_dim)

        out = out.permute(0, 2, 1, 3).contiguous().view(N, L, C)

        out = self.attn.out_proj(out)
        return out

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        """
        ClearCLIP at last block:
          z = LN1(x)
          a = Attn(z)      # with chosen self-self variant
          y = a            # discard residual
          (optional) y = y + FFN(LN2(y))   # keep_ffn
        """
        z = self.ln_1(x)
        a = self._attention_only(z, attn_mask=attn_mask)       # X_attn
        y = (x + self.ls_1(a)) if self.keep_residual else self.ls_1(a)
        if self.keep_ffn:
            y = y + self.ls_2(self.mlp(self.ln_2(y)))
        return y

def maybe_patch_clearclip(
    image_encoder: nn.Module,
    cfg: Dict[str, Any],
) -> int:
    """
    Patch the last N vision blocks to ClearCLIP behavior.
    cfg keys (all optional):
      - n_last: int = 1
      - attn_variant: str in {"qq","qk","kk","vv"}; default "qq"
      - keep_residual: bool = False
      - keep_ffn: bool = False
    Returns: number of blocks patched.
    """
    n_last       = int(cfg.get("n_last", 1))
    attn_variant = str(cfg.get("attn_variant", "qq")).lower()
    keep_res     = bool(cfg.get("keep_residual", False))
    keep_ffn     = bool(cfg.get("keep_ffn", False))

    resblocks = _get_resblocks(image_encoder)
    n = min(n_last, len(resblocks))
    for i in range(1, n + 1):
        old = resblocks[-i]
        resblocks[-i] = ClearCLIPLastBlock(
            old,
            attn_variant=attn_variant,
            keep_residual=keep_res,
            keep_ffn=keep_ffn,
        )
    return n
