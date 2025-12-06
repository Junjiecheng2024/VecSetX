# Copyright (c) 2025, Biao Zhang.

import numpy as np

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

try:
    from torch_cluster import fps
except ImportError:
    fps = None

import math

try:
    from flash_attn import flash_attn_kvpacked_func
except ImportError:
    flash_attn_kvpacked_func = None

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context = None, mask = None, window_size=-1):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        kv = self.to_kv(context)

        q = rearrange(q, 'b n (h d) -> b n h d', h = h)
        kv = rearrange(kv, 'b n (p h d) -> b n p h d', h = h, p=2)

        # Fallback to manual implementation
        # q: b n h d
        # kv: b n 2 h d
        k = kv[:, :, 0]
        v = kv[:, :, 1]
        
        # SDPA expects (batch, heads, seq_len, dim)
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        
        # Use optimized PyTorch SDPA
        out = F.scaled_dot_product_attention(q_t, k_t, v_t, dropout_p=0.0, is_causal=False)
        
        # Transpose back to (batch, seq_len, heads, dim)
        out = out.transpose(1, 2)

        return self.to_out(rearrange(out, 'b n h d -> b n (h d)'))

class PointEmbed(nn.Module):
    def __init__(self, hidden_dim=48, dim=128, input_dim=3):
        super().__init__()

        assert hidden_dim % 6 == 0

        self.embedding_dim = hidden_dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), e,
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6), e]),
        ])
        self.register_buffer('basis', e)  # 3 x 16

        self.mlp = nn.Linear(self.embedding_dim+input_dim, dim)

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum(
            'bnd,de->bne', input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings
    
    def forward(self, input):
        # input: B x N x D (D >= 3)
        coords = input[:, :, :3]
        embed = self.embed(coords, self.basis)
        embed = self.mlp(torch.cat([embed, input], dim=2)) # B x N x C
        return embed


# class PointEmbed(nn.Module):
#     def __init__(self, hidden_dim=48, dim=128):
#         super().__init__()

#         assert hidden_dim % 6 == 0

#         self.embedding_dim = hidden_dim
#         e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
#         e = torch.stack([
#             torch.cat([e, torch.zeros(self.embedding_dim // 6),
#                         torch.zeros(self.embedding_dim // 6)]),
#             torch.cat([torch.zeros(self.embedding_dim // 6), e,
#                         torch.zeros(self.embedding_dim // 6)]),
#             torch.cat([torch.zeros(self.embedding_dim // 6),
#                         torch.zeros(self.embedding_dim // 6), e]),
#         ])
#         self.register_buffer('basis', e)

#         self.mlp = nn.Linear(self.embedding_dim, dim)

#     @staticmethod
#     def embed(input, basis):
#         projections = torch.einsum('bnd,de->bne', input, basis)
#         embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
#         return embeddings
    
#     def forward(self, input):
#         # input: B x N x 3
#         embed = self.mlp(self.embed(input, self.basis)) # B x N x C
#         return embed


class DiagonalGaussianDistribution(object):
    def __init__(self, mean, logvar, deterministic=False):
        self.mean = mean
        self.logvar = logvar
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.mean.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.mean.device)
        return x
def subsample(pc, N, M):
    # pc: B x N x D
    B, N0, D = pc.shape
    assert N == N0
    
    ###### fps
    flattened = pc.view(B*N, D)
    coords = flattened[:, :3] # Use only coordinates for FPS

    batch = torch.arange(B).to(pc.device)
    batch = torch.repeat_interleave(batch, N)

    ratio = 1.0 * M / N

    if fps is not None:
        idx = fps(coords, batch, ratio=ratio)
    else:
        # Fallback to random sampling if fps is not available
        # We need to sample M points for each batch index
        # fps returns indices into flattened
        # We can just sample M points randomly for each batch
        # But fps output is flattened indices.
        
        # Let's use torch.randperm for each batch
        indices = []
        for b in range(B):
            perm = torch.randperm(N, device=pc.device)[:M]
            indices.append(perm + b * N)
        idx = torch.cat(indices)

    sampled_pc = flattened[idx]
    sampled_pc = sampled_pc.view(B, -1, D)
    ######

    return sampled_pc