import torch.nn as nn
from einops import rearrange
from flash_attn import flash_attn_kvpacked_func, flash_attn_qkvpacked_func


class Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        proj_drop = attn_drop if proj_drop == 0 else proj_drop

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        qkv = self.qkv(x)
        b, n, _ = qkv.shape
        qkv = rearrange(qkv, 'b n (x h d) -> b n x h d', x=3, h=self.num_heads, d=self.head_dim)
        x = flash_attn_qkvpacked_func(qkv, dropout_p=self.attn_drop)
        x = rearrange(x, 'b n h d -> b n (h d)')
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        proj_drop = attn_drop if proj_drop == 0 else proj_drop

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        q = self.q(x)
        kv = self.kv(y)
        q = rearrange(q, 'b n (h d) -> b n h d', h=self.num_heads, d=self.head_dim)
        kv = rearrange(kv, 'b n (x h d) -> b n x h d', x=2, h=self.num_heads, d=self.head_dim)
        x = flash_attn_kvpacked_func(q, kv, dropout_p=self.attn_drop)
        x = rearrange(x, 'b n h d -> b n (h d)')
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
