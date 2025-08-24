import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import  scaled_dot_product_attention_grouped_flash, MLP, RMSNorm
import torch._dynamo
from dataclasses import dataclass

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

@dataclass
class REXConfig:
    dim: int
    vocab_size: int
    encoder_layers: int
    decoder_layers: int
    num_heads: int
    max_length: int
    latent_dim: int
    dropout: float = 0.1


class GroupedQueryAttention(nn.Module):
    def __init__(self, 
                dim: int, 
                k_dim: int, 
                kv_heads: int, 
                query_heads: int, 
                max_length: int, 
                dropout: int = 0.1, 
                is_causal: bool = False, 
                apply_rotary: bool = True, 
                flash_attention: bool = False
                ):
        super().__init__()
        assert dim % query_heads == 0, "dim must be divisible by query_heads"
        self.dim = dim
        self.kv_heads = kv_heads
        self.query_heads = query_heads
        self.is_causal = is_causal
        self.max_length = max_length
        kv_dim = (dim // query_heads) * kv_heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(k_dim, kv_dim)
        self.v_proj = nn.Linear(k_dim, kv_dim)

        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.head_dim = dim // query_heads

        self.flash_attention = flash_attention

        self.apply_rotary = apply_rotary
        self.scale = self.head_dim**-0.5

    def rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, q, k, cos, sin):
    # Transpose to [B, H, L, D] for RoPE rotation
        q = q.permute(0, 2, 1, 3)  # [B, H, L, D]
        k = k.permute(0, 2, 1, 3)  # [B, H_kv, L, D]
    
        # Apply rotary embeddings
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
    
        # Transpose back to original layout [B, L, H, D]
        q_embed = q_embed.permute(0, 2, 1, 3)
        k_embed = k_embed.permute(0, 2, 1, 3)
    
        return q_embed, k_embed

    @torch._dynamo.disable()
    def generate_sin_cos_pos_emb(self, seq_len, device, rope_theta=10000, rope_factor=8.0):
        base, rope_factor, dim, max_seq_len = (
            rope_theta,
            rope_factor,
            self.head_dim,
            self.max_length
        )
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        if rope_factor > 1.0:  # Apply NTK dynamic scaling
            seq_len_eff = max(seq_len, max_seq_len)
            base_adjustment = ((rope_factor * seq_len_eff / max_seq_len) - (rope_factor - 1)) ** (dim / (dim - 2))
            adjusted_base = base * base_adjustment
            inv_freq = 1.0 / (adjusted_base ** (torch.arange(0, dim, 2, device=device).float() / dim))

        position_ids = torch.arange(seq_len, device=device, dtype=torch.float)
        if not self.is_causal:
            position_ids = position_ids - ((seq_len - 1) // 2)
        freqs = torch.einsum("i,j->ij", position_ids, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_emb = emb.cos()[None, None, :, :]
        sin_emb = emb.sin()[None, None, :, :]
        return cos_emb, sin_emb

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        bq, nq, dq = q.shape
        bk, nk, dk = k.shape
        bv, nv, dv = v.shape
        
        q = q.view(bq, nq, self.query_heads, dq // self.query_heads)
        k = k.view(bk, nk, self.kv_heads, dk // self.kv_heads)
        v = v.view(bv, nv, self.kv_heads, dv // self.kv_heads)

        if self.apply_rotary:
            cos_emb, sin_emb = self.generate_sin_cos_pos_emb(nq, device=q.device)
            cos_emb = cos_emb.to(q.device)
            sin_emb = sin_emb.to(q.device)
            q, k = self.apply_rotary_pos_emb(q, k, cos_emb, sin_emb)
            

        out = scaled_dot_product_attention_grouped_flash(q, k, v, self.scale, self.is_causal, mask=mask)
        out = out.reshape(out.size(0), out.size(1), out.size(2) * out.size(3))  # Flatten the heads
        out = self.out_proj(out)
        return out
    

class EncoderLayer(nn.Module):
    def __init__(self, 
                dim: int, 
                num_heads: int, 
                max_length: int, 
                dropout: float = 0.1, 
                is_causal: bool = False
                ):
        super().__init__()
        self.attention = GroupedQueryAttention(
            dim,
            dim,
            max(1, num_heads // 4),  # GQA typically uses half the number of heads
            num_heads, 
            max_length, 
            dropout, 
            is_causal,
        )

        self.norm_attn_in  = RMSNorm(dim)
        self.norm_attn_out = RMSNorm(dim)
        self.norm_mlp_in   = RMSNorm(dim)
        self.norm_mlp_out  = RMSNorm(dim)
        self.mlp = MLP(dim, dropout)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = self.norm_attn_in(x)
        attn_output_ = self.attention(x, x, x, mask=mask)
        attn_output = self.attn_dropout(attn_output_)
        x = x + attn_output
        x = self.norm_attn_out(x)
        x = self.norm_mlp_in(x)
        mlp_output = self.mlp(x)
        x = x + mlp_output
        x = self.norm_mlp_out(x)
        return x
    
class Encoder(nn.Module):
    def __init__(self, 
                dim: int, 
                vocab_size: int,
                num_layers: int, 
                num_heads: int, 
                max_length: int, 
                latent_dim: int, 
                dropout: float = 0.1, 
                is_causal: bool = False
                ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            EncoderLayer(dim, num_heads, max_length, dropout, is_causal) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(dim)
        self.latent_proj = nn.Linear(dim, latent_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, mask=mask)
        x = self.norm(x)
        return self.latent_proj(x)


class DecoderLayer(nn.Module):
    def __init__(self,
                dim: int, 
                latent_dim: int,
                num_heads: int, 
                max_length: int, 
                dropout: float = 0.1, 
                ):
        super(DecoderLayer, self).__init__()
        self.masked_attention = GroupedQueryAttention(
            dim,
            dim,
            max(1, num_heads // 4),
            num_heads, 
            max_length, 
            dropout, 
            is_causal=True
        )
        self.norm_masked_attn_in = RMSNorm(dim)
        self.norm_masked_attn_out = RMSNorm(dim)
        self.cross_attention = GroupedQueryAttention(
            dim,
            latent_dim,
            max(1, num_heads // 4),
            num_heads, 
            max_length, 
            dropout, 
            is_causal=False
        )
        self.norm_cross_attn_in = RMSNorm(dim)
        self.norm_cross_attn_out = RMSNorm(dim)
        self.mlp = MLP(dim, dropout)
        self.norm_mlp_in = RMSNorm(dim)
        self.norm_mlp_out = RMSNorm(dim)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, latent: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Masked self-attention
        x = self.norm_masked_attn_in(x)
        masked_attn_output = self.attn_dropout(self.masked_attention(x, x, x, mask=mask))
        x = x + masked_attn_output
        x = self.norm_masked_attn_out(x)

        # Cross-attention with latent representation
        x = self.norm_cross_attn_in(x)
        cross_attn_output = self.attn_dropout(self.cross_attention(x, latent, latent, mask=mask))
        x = x + cross_attn_output
        x = self.norm_cross_attn_out(x)

        # MLP
        x = self.norm_mlp_in(x)
        mlp_output = self.mlp(x)
        x = x + mlp_output
        x = self.norm_mlp_out(x)

        return x


class Decoder(nn.Module):
    def __init__(self, 
                dim: int, 
                vocab_size: int,
                num_layers: int, 
                num_heads: int, 
                max_length: int, 
                latent_dim: int, 
                dropout: float = 0.1,
                ):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            DecoderLayer(dim, latent_dim, num_heads, max_length, dropout) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(dim)
        self.out = nn.Linear(dim, vocab_size)

    def forward(self, x: torch.Tensor, latent: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, latent, mask=mask)
        x = self.norm(x)
        x = self.out(x)
        return x
    


class Transformer(nn.Module):
    def __init__(self, 
                dim: int, 
                vocab_size: int, 
                encoder_layers: int,
                decoder_layers: int, 
                num_heads: int, 
                max_length: int, 
                latent_dim: int, 
                dropout: float = 0.1, 
                is_causal: bool = False
                ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            dim=dim, 
            vocab_size=vocab_size, 
            num_layers=encoder_layers, 
            num_heads=num_heads, 
            max_length=max_length, 
            latent_dim=latent_dim, 
            dropout=dropout, 
            is_causal=is_causal
        )
        self.decoder = Decoder(
            dim=dim, 
            vocab_size=vocab_size, 
            num_layers=decoder_layers, 
            num_heads=num_heads, 
            max_length=max_length, 
            latent_dim=latent_dim, 
            dropout=dropout
        )

    def forward(self, input_ids: torch.Tensor, decoder_input_ids: torch.Tensor, attention_mask: torch.Tensor = None, decoder_attention_mask: torch.Tensor = None) -> torch.Tensor:
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :].to(torch.bool)
        if decoder_attention_mask is not None:
            decoder_attention_mask = decoder_attention_mask[:, None, None, :].to(torch.bool)
        latent = self.encoder(input_ids, mask=attention_mask)
        output = self.decoder(decoder_input_ids, latent, mask=decoder_attention_mask)
        return output
    
