import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._dynamo import disable
from torch.nn.attention import SDPBackend


def scaled_dot_product_attention_grouped_flash(
        queries: torch.Tensor,
        keys: torch.Tensor, 
        values: torch.Tensor, 
        scale: float, 
        is_causal: bool = False,
        dropout_p: float = 0.0,
        mask: torch.Tensor = None
        ) -> torch.Tensor:
    """
    Compute scaled dot-product attention with grouped queries.
    
    Args:
        queries (torch.Tensor): Query tensor of shape (B, T_q, C).
        keys (torch.Tensor): Key tensor of shape (B, T_k, C).
        values (torch.Tensor): Value tensor of shape (B, T_v, C).
        scale (float): Scaling factor for the dot product.
        
    Returns:
        torch.Tensor: Output tensor after applying attention.
    """
    q = queries.permute(0, 2, 1, 3)
    k = keys.permute(0, 2, 1, 3)
    v = values.permute(0, 2, 1, 3)

    bq, hq, nq, dq = q.shape
    bk, hk, nk, dk = k.shape
    bv, hv, nv, dv = v.shape

    repeat = hq // hk
    k = k.repeat_interleave(repeat, dim=1)  # (B, hq, Tk, d)
    v = v.repeat_interleave(repeat, dim=1)  # (B, hq, Tv, d)

    with torch.nn.attention.sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        out = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale
        )
    out = out.permute(0, 2, 1, 3)

    return out

class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout:int = 0.1):
        super(MLP, self).__init__()
        self.w1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.w2 = nn.Linear(4 * n_embd, n_embd, bias=False)
        self.w3 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x : torch.Tensor):
        h = F.silu(self.w1(x))
        h = self.dropout(h)
        out = self.w2(h * self.w3(x))
        return out

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)
