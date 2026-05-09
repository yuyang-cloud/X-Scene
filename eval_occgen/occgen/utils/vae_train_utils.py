import torch
import torch.nn.functional as F


def get_pred_label(pred, dim=-1):
    pred = torch.softmax(pred, dim=dim)
    pred = pred.argmax(dim=dim)
    return pred


def sample_feature_plane2d(feat_map, x):
    """Sample feature map at given coordinates"""
    sample_coords = x.reshape(x.shape[0], 1, -1, 2)  # B, 1, N, 2
    feat = F.grid_sample(
        feat_map, sample_coords.flip(-1), align_corners=False, padding_mode='border'
    )  # B, C, 1, N
    feat = feat[:, :, 0, :]  # B, C, N
    feat = feat.transpose(1, 2)  # B, N, C
    return feat


def pred_to_voxels(preds, coords, grid_size, t):
    output = torch.zeros((preds.shape[0], t, *grid_size, preds.shape[-1]), device=preds.device, dtype=preds.dtype)
    for i in range(preds.shape[0]):
        output[i, coords[i, :, 3], coords[i, :, 0], coords[i, :, 1], coords[i, :, 2]] = preds[i]
    return output


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


def kl_loss(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def compose_hexplane_channelwise(feat_maps):
    h_xy, h_xz, h_yz, h_tx, h_ty, h_tz = feat_maps  # (W, H), (W, D), (H, D)
    sizes_xyz = B, C, T, W, H, D = *h_xy.shape[:2], h_tx.shape[2], *h_xy.shape[2:], h_xz.shape[3]

    w_new, h_new = max(W, H), max(H, D)
    h_xy = F.pad(h_xy, (0, h_new - H, 0, w_new - W))  # W, H -> w_new, h_new
    h_xz = F.pad(h_xz, (0, h_new - D, 0, w_new - W))  # W, D -> w_new, h_new
    h_yz = F.pad(h_yz, (0, h_new - D, 0, w_new - H))  # H, D -> w_new, h_new
    h = torch.cat([h_xy, h_xz, h_yz], dim=1)  # (B, 3C, w_new, h_new)

    sizes_t = B, C, T, W, H, D = *h_tx.shape[:2], h_tx.shape[2], h_tx.shape[3], h_ty.shape[3], h_tz.shape[3]
    t_new = max(W, H, D)
    h_tx = F.pad(h_tx, (0, t_new - W))
    h_ty = F.pad(h_ty, (0, t_new - H))
    h_tz = F.pad(h_tz, (0, t_new - D))
    t = torch.cat([h_tx, h_ty, h_tz], dim=1)
    return h, t, sizes_xyz, sizes_t


def decompose_hexplane_channelwise(h, t, sizes_xyz, sizes_t):
    B, C, N, W, H, D = sizes_xyz
    C = h.shape[1] // 3
    h_xy = h[:, : C, : W, : H]
    h_xz = h[:, C: 2 * C, : W, : D]
    h_yz = h[:, 2 * C:, : H, : D]

    B, C, N, W, H, D = sizes_t
    C = t.shape[1] // 3
    h_tx = t[:, : C, :, : W]
    h_ty = t[:, C: 2 * C, :, : H]
    h_tz = t[:, 2 * C:, :, : D]
    return [h_xy, h_xz, h_yz, h_tx, h_ty, h_tz]


def add_positional_encoding(voxel, pos_num_freq):
    B, T, X, Y, Z, C = voxel.shape
    t = torch.linspace(0, T - 1, T)
    x = torch.linspace(0, X - 1, X)
    y = torch.linspace(0, Y - 1, Y)
    z = torch.linspace(0, Z - 1, Z)
    tt, xx, yy, zz = torch.meshgrid(t, x, y, z, indexing='ij')  # t, x, y, z
    coords = torch.stack([tt, xx, yy, zz], dim=-1).to(voxel.device)  # t, x, y, z, 4
    coords = coords.unsqueeze(0).repeat(B, 1, 1, 1, 1, 1)  # b, t, x, y, z, 4

    positional_encoding = list()
    for freq in range(pos_num_freq):
        positional_encoding.append(torch.sin((2. ** freq) * coords))
        positional_encoding.append(torch.cos((2. ** freq) * coords))
    positional_encoding = torch.cat(positional_encoding, dim=-1)  # b, t, x, y, z, 4 * 2 * pos_num_freq)

    voxel = torch.cat([voxel, positional_encoding], dim=-1)
    return voxel
