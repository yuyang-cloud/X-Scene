import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class SinusoidalEncoder(nn.Module):
    """Sinusoidal Positional Encoder used in Nerf."""

    def __init__(self, x_dim, min_deg, max_deg, use_identity: bool = True):
        super().__init__()
        self.x_dim = x_dim
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.use_identity = use_identity
        self.register_buffer(
            "scales", torch.tensor([2**i for i in range(min_deg, max_deg)])
        )

    @property
    def latent_dim(self) -> int:
        return (
            int(self.use_identity) + (self.max_deg - self.min_deg) * 2
        ) * self.x_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., x_dim]
        Returns:
            latent: [..., latent_dim]
        """
        if self.max_deg == self.min_deg:
            return x
        xb = torch.reshape(
            (x[Ellipsis, None, :] * self.scales[:, None]),
            list(x.shape[:-1]) + [(self.max_deg - self.min_deg) * self.x_dim],
        )
        latent = torch.sin(torch.cat([xb, xb + 0.5 * math.pi], dim=-1))
        if self.use_identity:
            latent = torch.cat([x] + [latent], dim=-1)
        return latent

class DecoderMLPSkipConcat(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, num_hidden_layers, posenc=0) -> None:
        super().__init__()
        self.posenc = posenc
        if posenc > 0:
            self.PE = SinusoidalEncoder(in_channels, 0, posenc, use_identity=True)
            in_channels = self.PE.latent_dim
        first_layer_list = [nn.Linear(in_channels, hidden_channels), nn.ReLU()]
        for _ in range(num_hidden_layers // 2):
            first_layer_list.append(nn.Linear(hidden_channels, hidden_channels))
            first_layer_list.append(nn.ReLU())
        self.first_layers = nn.Sequential(*first_layer_list)
        
        second_layer_list = [nn.Linear(in_channels + hidden_channels, hidden_channels), nn.ReLU()]
        for _ in range(num_hidden_layers // 2 - 1):
            second_layer_list.append(nn.Linear(hidden_channels, hidden_channels))
            second_layer_list.append(nn.ReLU())
        second_layer_list.append(nn.Linear(hidden_channels, out_channels))
        self.second_layers = nn.Sequential(*second_layer_list)
    
    def forward(self, x):
        if self.posenc > 0:
            x = self.PE(x)
        h = self.first_layers(x)
        h = torch.cat([x, h], dim=-1)
        h = self.second_layers(h)
        return h


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def compose_triplane_channelwise(feat_maps):
    h_xy, h_xz, h_yz = feat_maps # (H, W), (H, D), (W, D)
    assert h_xy.shape[1] == h_xz.shape[1] == h_yz.shape[1]
    C, H, W = h_xy.shape[-3:]
    D = h_xz.shape[-1]

    newH = max(H, W)
    newW = max(W, D)
    h_xy = F.pad(h_xy, (0, newW - W, 0, newH - H))
    h_xz = F.pad(h_xz, (0, newW - D, 0, newH - H))
    h_yz = F.pad(h_yz, (0, newW - D, 0, newH - W))
    h = torch.cat([h_xy, h_xz, h_yz], dim=1) # (B, 3C, H, W)

    return h, (H, W, D)


def decompose_triplane_channelwise(composed_map, sizes):
    H, W, D = sizes
    C = composed_map.shape[1] // 3
    h_xy = composed_map[:, :C, :H, :W]
    h_xz = composed_map[:, C:2*C, :H, :D]
    h_yz = composed_map[:, 2*C:, :W, :D]
    return h_xy, h_xz, h_yz


class TriplaneDeformableAttention(nn.Module):
    def __init__(self, K=4, pos_dim=36):
        super().__init__()
        self.K = K
        
        self.attn_xy = nn.Linear(pos_dim, K)
        self.attn_xz = nn.Linear(pos_dim, K)
        self.attn_yz = nn.Linear(pos_dim, K)

        self.offset_xy = nn.Linear(pos_dim, 2 * K)
        self.offset_xz = nn.Linear(pos_dim, 2 * K)
        self.offset_yz = nn.Linear(pos_dim, 2 * K)
        
        # Initialize offsets to 0 to start with standard center sampling
        nn.init.zeros_(self.offset_xy.weight)
        nn.init.zeros_(self.offset_xy.bias)
        nn.init.zeros_(self.offset_xz.weight)
        nn.init.zeros_(self.offset_xz.bias)
        nn.init.zeros_(self.offset_yz.weight)
        nn.init.zeros_(self.offset_yz.bias)

    def forward(self, query, pe, feat_maps):
        # query: [bs, N, 3] coordinates in [-1, 1]
        # pe: [bs, N, pos_dim] positional encoding
        # feat_maps: list of 3 tensors [bs, C, H, W]
        bs, N, _ = query.shape
        
        # Get base 2D projection coordinates
        coords_xy = query[..., [0, 1]]
        coords_xz = query[..., [0, 2]]
        coords_yz = query[..., [1, 2]]
        
        def sample_plane(coords, pe, feat_map, attn_layer, offset_layer):
            # 1. Calculate and apply softmax to attention weights
            attn = attn_layer(pe)           # [bs, N, K]
            attn = F.softmax(attn, dim=-1)  # [bs, N, K]
            
            # 2. Calculate offsets
            offsets = offset_layer(pe).view(bs, N, self.K, 2)  # [bs, N, K, 2]
            
            # 3. Add offsets to base coordinates
            sample_coords = coords.unsqueeze(2) + offsets      # [bs, N, K, 2]
            
            # 4. Bilinear interpolation sampling (N as H_out, K as W_out)
            sampled_feat = F.grid_sample(
                feat_map, 
                sample_coords.flip(-1), 
                align_corners=False, 
                padding_mode='border'
            ) # [bs, C, N, K]
            
            # 5. Weighted aggregation
            sampled_feat = sampled_feat.permute(0, 2, 3, 1)    # [bs, N, K, C]
            weighted_feat = sampled_feat * attn.unsqueeze(-1)  # [bs, N, K, C]
            
            return weighted_feat.sum(dim=2)                    # [bs, N, C]

        # Sample from all three planes and aggregate
        f_xy = sample_plane(coords_xy, pe, feat_maps[0], self.attn_xy, self.offset_xy)
        f_xz = sample_plane(coords_xz, pe, feat_maps[1], self.attn_xz, self.offset_xz)
        f_yz = sample_plane(coords_yz, pe, feat_maps[2], self.attn_yz, self.offset_yz)
        
        return f_xy + f_xz + f_yz


class TriplaneGroupResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up=False, ks=3, input_norm=True, input_act=True):
        super().__init__()
        in_channels *= 3
        out_channels *= 3

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        
        self.input_norm = input_norm
        if input_norm and input_act:
            self.in_layers = nn.Sequential(
                # nn.GroupNorm(num_groups=3, num_channels=in_channels, eps=1e-6, affine=True),
                SiLU(),
                nn.Conv2d(in_channels, out_channels, groups=3, kernel_size=ks, stride=1, padding=(ks - 1)//2)
            )
        elif not input_norm:
            if input_act:
                self.in_layers = nn.Sequential(
                    SiLU(),
                    nn.Conv2d(in_channels, out_channels, groups=3, kernel_size=ks, stride=1, padding=(ks - 1)//2)
                )
            else:
                self.in_layers = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, groups=3, kernel_size=ks, stride=1, padding=(ks - 1)//2)
                )
        else:
            raise NotImplementedError

        self.norm_xy = nn.InstanceNorm2d(out_channels//3, eps=1e-6, affine=True)
        self.norm_xz = nn.InstanceNorm2d(out_channels//3, eps=1e-6, affine=True)
        self.norm_yz = nn.InstanceNorm2d(out_channels//3, eps=1e-6, affine=True)

        self.out_layers = nn.Sequential(
            # nn.GroupNorm(num_groups=3, num_channels=out_channels, eps=1e-6, affine=True),
            SiLU(),
            # nn.Dropout(p=dropout),
            zero_module(
                nn.Conv2d(out_channels, out_channels, groups=3, kernel_size=ks, stride=1, padding=(ks - 1)//2)
            ),
        )

        if self.in_channels != self.out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, groups=3, kernel_size=1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()

    def forward(self, feat_maps):
        if self.input_norm:
            feat_maps = [self.norm_xy(feat_maps[0]), self.norm_xz(feat_maps[1]), self.norm_yz(feat_maps[2])]
        x, (H, W, D) = compose_triplane_channelwise(feat_maps)

        if self.up:
            raise NotImplementedError
        else:
            h = self.in_layers(x)
        
        h_xy, h_xz, h_yz = decompose_triplane_channelwise(h, (H, W, D))
        h_xy = self.norm_xy(h_xy)
        h_xz = self.norm_xz(h_xz)
        h_yz = self.norm_yz(h_yz)
        h, _ = compose_triplane_channelwise([h_xy, h_xz, h_yz])

        h = self.out_layers(h)
        h = h + self.shortcut(x)
        h_maps = decompose_triplane_channelwise(h, (H, W, D))
        return h_maps

class BeVplaneGroupResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up=False, ks=3, input_norm=True, input_act=True):
        super().__init__()
        in_channels 
        out_channels 

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        
        self.input_norm = input_norm
        if input_norm and input_act:
            self.in_layers = nn.Sequential(
                # nn.GroupNorm(num_groups=3, num_channels=in_channels, eps=1e-6, affine=True),
                SiLU(),
                nn.Conv2d(in_channels, out_channels,  kernel_size=ks, stride=1, padding=(ks - 1)//2)
            )
        elif not input_norm:
            if input_act:
                self.in_layers = nn.Sequential(
                    SiLU(),
                    nn.Conv2d(in_channels, out_channels,  kernel_size=ks, stride=1, padding=(ks - 1)//2)
                )
            else:
                self.in_layers = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=ks, stride=1, padding=(ks - 1)//2)
                )
        else:
            raise NotImplementedError

        self.norm_xy = nn.InstanceNorm2d(out_channels, eps=1e-6, affine=True)
        self.norm_xz = nn.InstanceNorm2d(out_channels, eps=1e-6, affine=True)
        self.norm_yz = nn.InstanceNorm2d(out_channels, eps=1e-6, affine=True)

        self.out_layers = nn.Sequential(
            # nn.GroupNorm(num_groups=3, num_channels=out_channels, eps=1e-6, affine=True),
            SiLU(),
            # nn.Dropout(p=dropout),
            zero_module(
                nn.Conv2d(out_channels, out_channels,  kernel_size=ks, stride=1, padding=(ks - 1)//2)
            ),
        )

        if self.in_channels != self.out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels,  kernel_size=1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()

    def forward(self, feat_maps):
        if self.input_norm:
            feat_maps = [self.norm_xy(feat_maps[0]), self.norm_xz(feat_maps[1]), self.norm_yz(feat_maps[2])]
        
        x = feat_maps[0]
        if self.up:
            raise NotImplementedError
        else:
            h = self.in_layers(x)
        
        h = self.norm_xy(h)
        h = self.out_layers(h)
        h = h + self.shortcut(x)
        h_maps = [h, feat_maps[1], feat_maps[2]]
        return h_maps

def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels):
    if in_channels <= 2:
        num_groups = 2
    elif in_channels <= 32:
        num_groups = in_channels // 4
    else:
        num_groups = 32
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb=None):
        h = x
        # h = nonlinearity(h)
        h = self.norm1(h)
        h = self.conv1(h)
        h = F.relu(h)
        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        # h = nonlinearity(h)
        h = self.norm2(h)
        h = self.conv2(h)
        h = self.dropout(h)
        h = F.relu(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


# Shifts src_tf dim to dest dim
# i.e. shift_dim(x, 1, -1) would be (b, c, t, h, w) -> (b, t, h, w, c)
def shift_dim(x, src_dim=-1, dest_dim=-1, make_contiguous=True):
    n_dims = len(x.shape)
    if src_dim < 0:
        src_dim = n_dims + src_dim
    if dest_dim < 0:
        dest_dim = n_dims + dest_dim

    assert 0 <= src_dim < n_dims and 0 <= dest_dim < n_dims

    dims = list(range(n_dims))
    del dims[src_dim]

    permutation = []
    ctr = 0
    for i in range(n_dims):
        if i == dest_dim:
            permutation.append(src_dim)
        else:
            permutation.append(dims[ctr])
            ctr += 1
    x = x.permute(permutation)
    if make_contiguous:
        x = x.contiguous()
    return x


# reshapes tensor start from dim i (inclusive)
# to dim j (exclusive) to the desired shape
# e.g. if x.shape = (b, thw, c) then
# view_range(x, 1, 2, (t, h, w)) returns
# x of shape (b, t, h, w, c)
def view_range(x, i, j, shape):
    shape = tuple(shape)

    n_dims = len(x.shape)
    if i < 0:
        i = n_dims + i

    if j is None:
        j = n_dims
    elif j < 0:
        j = n_dims + j

    assert 0 <= i < j <= n_dims

    x_shape = x.shape
    target_shape = x_shape[:i] + shape + x_shape[j:]
    return x.view(target_shape)


def scaled_dot_product_attention(q, k, v, mask=None, attn_dropout=0.0, training=True):
    # Performs scaled dot-product attention over the second to last dimension dn

    # (b, n_head, d1, ..., dn, d)
    attn = torch.matmul(q, k.transpose(-1, -2))
    attn = attn / np.sqrt(q.shape[-1])
    if mask is not None:
        attn = attn.masked_fill(mask == 0, float("-inf"))
    attn_float = F.softmax(attn, dim=-1)
    attn = attn_float.type_as(attn)  # b x n_head x d1 x ... x dn x d
    attn = F.dropout(attn, p=attn_dropout, training=training)

    a = torch.matmul(attn, v)  # b x n_head x d1 x ... x dn x d

    return a

class MultiHeadAttention(nn.Module):
    def __init__(self, shape, dim_q, dim_kv, n_head, n_layer, causal, attn_type, attn_kwargs):
        super().__init__()
        self.causal = causal
        self.shape = shape

        self.d_k = dim_q // n_head
        self.d_v = dim_kv // n_head
        self.n_head = n_head

        self.w_qs = nn.Linear(dim_q, n_head * self.d_k, bias=False)  # q
        self.w_qs.weight.data.normal_(std=1.0 / np.sqrt(dim_q))

        self.w_ks = nn.Linear(dim_kv, n_head * self.d_k, bias=False)  # k
        self.w_ks.weight.data.normal_(std=1.0 / np.sqrt(dim_kv))

        self.w_vs = nn.Linear(dim_kv, n_head * self.d_v, bias=False)  # v
        self.w_vs.weight.data.normal_(std=1.0 / np.sqrt(dim_kv))

        self.fc = nn.Linear(n_head * self.d_v, dim_q, bias=True)  # c
        self.fc.weight.data.normal_(std=1.0 / np.sqrt(dim_q * n_layer))

        if attn_type == "full":
            self.attn = FullAttention(shape, causal, **attn_kwargs)
        elif attn_type == "axial":
            assert not causal, "causal axial attention is not supported"
            self.attn = AxialAttention(len(shape), **attn_kwargs)

        self.cache = None

    def forward(self, q, k, v, decode_step=None, decode_idx=None):
        """Compute multi-head attention Args q, k, v: a [b, d1, ..., dn, c]
        tensor or a [b, 1, ..., 1, c] tensor if decode_step is not None.

        Returns     The output after performing attention
        """

        # compute k, q, v
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        q = view_range(self.w_qs(q), -1, None, (n_head, d_k))
        k = view_range(self.w_ks(k), -1, None, (n_head, d_k))
        v = view_range(self.w_vs(v), -1, None, (n_head, d_v))

        # b x n_head x seq_len x d
        # (b, *d_shape, n_head, d) -> (b, n_head, *d_shape, d)
        q = shift_dim(q, -2, 1)
        k = shift_dim(k, -2, 1)
        v = shift_dim(v, -2, 1)

        # fast decoding
        if decode_step is not None:
            if decode_step == 0:
                if self.causal:
                    k_shape = (q.shape[0], n_head, *self.shape, self.d_k)
                    v_shape = (q.shape[0], n_head, *self.shape, self.d_v)
                    self.cache = dict(
                        k=torch.zeros(k_shape, dtype=k.dtype, device=q.device),
                        v=torch.zeros(v_shape, dtype=v.dtype, device=q.device),
                    )
                else:
                    # cache only once in the non-causal case
                    self.cache = dict(k=k.clone(), v=v.clone())
            if self.causal:
                idx = (slice(None, None), slice(None, None), *[slice(i, i + 1) for i in decode_idx])
                self.cache["k"][idx] = k
                self.cache["v"][idx] = v
            k, v = self.cache["k"], self.cache["v"]

        a = self.attn(q, k, v, decode_step, decode_idx)

        # (b, *d_shape, n_head, d) -> (b, *d_shape, n_head * d)
        a = shift_dim(a, 1, -2).flatten(start_dim=-2)
        a = self.fc(a)  # (b x seq_len x embd_dim)

        return a


############## Attention #######################
class FullAttention(nn.Module):
    def __init__(self, shape, causal, attn_dropout):
        super().__init__()
        self.causal = causal
        self.attn_dropout = attn_dropout

        seq_len = np.prod(shape)
        if self.causal:
            self.register_buffer("mask", torch.tril(torch.ones(seq_len, seq_len)))

    def forward(self, q, k, v, decode_step, decode_idx):
        mask = self.mask if self.causal else None
        if decode_step is not None and mask is not None:
            mask = mask[[decode_step]]

        old_shape = q.shape[2:-1]
        q = q.flatten(start_dim=2, end_dim=-2)
        k = k.flatten(start_dim=2, end_dim=-2)
        v = v.flatten(start_dim=2, end_dim=-2)

        out = scaled_dot_product_attention(q, k, v, mask=mask, attn_dropout=self.attn_dropout, training=self.training)

        return view_range(out, 2, 3, old_shape)


class AxialAttention(nn.Module):
    def __init__(self, n_dim, axial_dim):
        super().__init__()
        if axial_dim < 0:
            axial_dim = 2 + n_dim + 1 + axial_dim
        else:
            axial_dim += 2  # account for batch, head, dim
        self.axial_dim = axial_dim

    def forward(self, q, k, v, decode_step, decode_idx):
        q = shift_dim(q, self.axial_dim, -2).flatten(end_dim=-3)
        k = shift_dim(k, self.axial_dim, -2).flatten(end_dim=-3)
        v = shift_dim(v, self.axial_dim, -2)
        old_shape = list(v.shape)
        v = v.flatten(end_dim=-3)

        out = scaled_dot_product_attention(q, k, v, training=self.training)
        out = out.view(*old_shape)
        out = shift_dim(out, -2, self.axial_dim)
        return out

class AxialBlock_hw(nn.Module):
    """
    Axial attention block for 4D tensors (B,C,H,W):
    sequentially applies attention along H and W.
    """
    def __init__(self, n_hiddens, n_head):
        super().__init__()
        kwargs = dict(
            shape=(0,0),  # only two spatial dims: H,W
            dim_q=n_hiddens, dim_kv=n_hiddens, n_head=n_head, n_layer=1, causal=False, attn_type="axial"
        )
        self.attn_h = MultiHeadAttention(attn_kwargs=dict(axial_dim=-3), **kwargs)  # H dim
        self.attn_w = MultiHeadAttention(attn_kwargs=dict(axial_dim=-2), **kwargs)  # W dim

    def forward(self, x):  # x: B,C,H,W
        x = shift_dim(x, 1, -1)  # B,H,W,C
        x = self.attn_h(x, x, x) + self.attn_w(x, x, x)
        x = shift_dim(x, -1, 1)  # B,C,H,W
        return x


class Attn_ResBlock_hw(nn.Module):
    """
    Residual block with axial attention (H,W) for 4D tensors.
    """
    def __init__(self, n_hiddens):
        super().__init__()
        self.block = nn.Sequential(
            Normalize(n_hiddens),
            AxialBlock_hw(n_hiddens, n_head=2)
        )

    def forward(self, x):
        return x + self.block(x)