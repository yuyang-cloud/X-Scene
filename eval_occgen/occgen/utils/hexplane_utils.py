import torch


def compose_featmaps(feat_xy, feat_xz, feat_yz, feat_tx, feat_ty, feat_tz, txyz=(4, 64, 64, 8)):
    """
    Combines hexplane as a single feature map,
    shape of the output: batch size, channels, h + d + 2t, w + d.
    composition (last two dims)
    - xy: :h, :w
    - xz: :h, w:
    - zy (transpose -1, -2 to obtain yz): h: h+d, :w
    - tx: h+d: h+d+t, :w
    - ty: h+d+t:, :h
    - tz: h+d: h+d+t, w:
    """
    B, C = feat_xy.shape[:-2]
    T, W, H, D = txyz
    assert W == H, 'compose_featmaps only works for H == W'

    feat_xy_xz = torch.cat([feat_xy, feat_xz], dim=-1)  # B, C, W, H + D

    feat_zy = feat_yz.transpose(-1, -2)  # B, C, D, H
    empty_zz = torch.zeros(B, C, D, D, dtype=feat_xy.dtype, device=feat_xy.device)  # B, C, D, D
    feat_zy_00 = torch.cat([feat_zy, empty_zz], dim=-1)  # B, C, D, H + D

    feat_tx_ty = torch.cat([feat_tx, feat_ty], dim=-2)  # B, C, T + T, W/H
    empty_tz = torch.zeros([B, C, T, D], dtype=feat_xy.dtype, device=feat_xy.device)  # B, C, T, D
    feat_tz_00 = torch.cat([feat_tz, empty_tz], dim=-2)  # B, C, T + T, D
    feat_t_all = torch.cat([feat_tx_ty, feat_tz_00], dim=-1)  # B, C, T + T, W/H + D

    composed_map = torch.cat([feat_xy_xz, feat_zy_00, feat_t_all], dim=-2)  # B, C, W + D + T + T, W/H + D

    w, h = composed_map.shape[-2:]
    composed_map = torch.nn.functional.pad(composed_map, (0, max(w, h) - h, 0, max(w, h) - w))

    return composed_map  # B, C, W+D+2T, W+D+2T


def decompose_featmaps(composed_map, txyz=(4, 64, 64, 8)):
    T, W, H, D = txyz
    feat_xy = composed_map[..., :W, :H]  # (B, C, W, H)
    feat_xz = composed_map[..., :W, H: H + D]  # (B, C, W, D)
    feat_yz = composed_map[..., W: W + D, :W].transpose(-1, -2)  # (B, C, H, D)
    feat_tx = composed_map[..., W + D: W + D + T, :W]  # (B, C, T, W)
    feat_ty = composed_map[..., W + D + T:, :H]  # (B, C, T, H)
    feat_tz = composed_map[..., W + D: W + D + T, W: W + D]  # (B, C, T, D)
    return feat_xy, feat_xz, feat_yz, feat_tx, feat_ty, feat_tz


def compose_featmaps_no(feat_xy, feat_xz, feat_yz, feat_tx, feat_ty, feat_tz, txyz=(4, 64, 64, 8)):
    """
    Combines hexplane as a single feature map,
    shape of the output: batch size, channels, max(W,H,D,T)*6, max(W,H,D,T)*6.
    composition (last two dims)
    - xy: 0:W, 0:H
    - xz: W:2W, 0:D
    - yz: 2W:3W, 0:D
    - tx: 3W:4W, 0:T
    - ty: 4W:5W, 0:T
    - tz: 5W:6W, 0:T
    """
    B, C = feat_xy.shape[:-2]
    T, W, H, D = txyz
    max_dim = max(W, H, D, T)

    # Pad each feature map to max_dim x max_dim
    feat_xy = torch.nn.functional.pad(feat_xy, (0, max_dim - H, 0, max_dim - W))
    feat_xz = torch.nn.functional.pad(feat_xz, (0, max_dim - D, 0, max_dim - W))
    feat_yz = torch.nn.functional.pad(feat_yz, (0, max_dim - D, 0, max_dim - H))
    feat_tx = torch.nn.functional.pad(feat_tx, (0, max_dim - W, 0, max_dim - T))
    feat_ty = torch.nn.functional.pad(feat_ty, (0, max_dim - H, 0, max_dim - T))
    feat_tz = torch.nn.functional.pad(feat_tz, (0, max_dim - D, 0, max_dim - T))

    # Concatenate along the -2 dimension
    composed_map = torch.cat([feat_xy, feat_xz, feat_yz, feat_tx, feat_ty, feat_tz], dim=-2)

    # Pad the result to make it square
    composed_map = torch.nn.functional.pad(composed_map, (0, max_dim * 5, 0, 0))

    return composed_map  # B, C, max_dim*6, max_dim*6


def decompose_featmaps_no(composed_map, txyz=(4, 64, 64, 8)):
    T, W, H, D = txyz
    max_dim = max(W, H, D, T)

    feat_xy = composed_map[..., 0:max_dim, :max_dim][:, :, :W, :H]
    feat_xz = composed_map[..., max_dim:2 * max_dim, :max_dim][:, :, :W, :D]
    feat_yz = composed_map[..., 2 * max_dim:3 * max_dim, :max_dim][:, :, :H, :D]
    feat_tx = composed_map[..., 3 * max_dim:4 * max_dim, :max_dim][:, :, :T, :W]
    feat_ty = composed_map[..., 4 * max_dim:5 * max_dim, :max_dim][:, :, :T, :H]
    feat_tz = composed_map[..., 5 * max_dim:6 * max_dim, :max_dim][:, :, :T, :D]

    return feat_xy, feat_xz, feat_yz, feat_tx, feat_ty, feat_tz


def compose_featmaps_transpose(feat_xy, feat_xz, feat_yz, feat_tx, feat_ty, feat_tz, txyz=(4, 64, 64, 8), pad=True):
    """
    Combines hexplane as a single feature map,
    shape of the output: batch size, channels, h + d + 2t, w + d.
    composition (last two dims)
    - xy: :w, :h
    - xz: :w, h:h+d
    - zy: w+t:, :h
    - tx: :w, h+d:
    - ty: w:w+t :h
    - tz: w:w+t h:h+d
    """
    B, C = feat_xy.shape[:-2]
    T, W, H, D = txyz
    assert W == H, 'compose_featmaps only works for H == W'

    if T % 2 == 1 and pad:
        T += 1
        feat_tx = torch.nn.functional.pad(feat_tx, (0, 0, 0, 1))
        feat_ty = torch.nn.functional.pad(feat_ty, (0, 0, 0, 1))
        feat_tz = torch.nn.functional.pad(feat_tz, (0, 0, 0, 1))

    feat_xy_xz_xt = torch.cat([feat_xy, feat_xz, feat_tx.transpose(-1, -2)], dim=-1)  # B, C, W, H + D

    feat_tt = torch.zeros(B, C, T, T, dtype=feat_xy.dtype, device=feat_xy.device)
    feat_ty_tz_tt = torch.cat([feat_ty, feat_tz, feat_tt], dim=-1)

    feat_zz = torch.zeros(B, C, D, D, dtype=feat_xy.dtype, device=feat_xy.device)
    feat_zt = torch.zeros(B, C, D, T, dtype=feat_xy.dtype, device=feat_xy.device)
    feat_zy_zz_zt = torch.cat([feat_yz.transpose(-1, -2), feat_zz, feat_zt], dim=-1)

    composed_map = torch.cat([feat_xy_xz_xt, feat_ty_tz_tt, feat_zy_zz_zt], dim=-2)  # B, C, W + T + D. W + T + D

    return composed_map  # B, C, W + T + D. W + T + D


def decompose_featmaps_transpose(composed_map, txyz=(4, 64, 64, 8)):
    T, W, H, D = txyz
    feat_xy = composed_map[..., :W, :H]  # (B, C, W, H)
    feat_xz = composed_map[..., :W, H: H + D]  # (B, C, W, D)
    feat_yz = composed_map[..., W + T:, :H].transpose(-1, -2)  # (B, C, H, D)
    feat_tx = composed_map[..., :W, H + D:].transpose(-1, -2)  # (B, C, T, W)
    feat_ty = composed_map[..., W: W + T, :H]  # (B, C, T, H)
    feat_tz = composed_map[..., W: W + T, H: H + D]  # (B, C, T, D)
    return feat_xy, feat_xz, feat_yz, feat_tx, feat_ty, feat_tz


def get_hexplane_mask(txyz):
    T, W, H, D = txyz
    return compose_featmaps(
        torch.ones(1, 1, W, H),
        torch.ones(1, 1, W, D),
        torch.ones(1, 1, H, D),
        torch.ones(1, 1, T, W),
        torch.ones(1, 1, T, H),
        torch.ones(1, 1, T, D),
        txyz=txyz
    )


def get_hexplane_mask_transpose(txyz, pad=True):
    T, W, H, D = txyz
    return compose_featmaps_transpose(
        torch.ones(1, 1, W, H),
        torch.ones(1, 1, W, D),
        torch.ones(1, 1, H, D),
        torch.ones(1, 1, T, W),
        torch.ones(1, 1, T, H),
        torch.ones(1, 1, T, D),
        txyz=txyz,
        pad=pad
    )


def get_hexplane_mask_no(txyz):
    T, W, H, D = txyz
    return compose_featmaps_no(
        torch.ones(1, 1, W, H),
        torch.ones(1, 1, W, D),
        torch.ones(1, 1, H, D),
        torch.ones(1, 1, T, W),
        torch.ones(1, 1, T, H),
        torch.ones(1, 1, T, D),
        txyz=txyz
    )


def get_hexplane_mask_bbox(txyz, bbox):
    T, W, H, D = txyz
    x0, xn, y0, yn = bbox

    mask_W_H = torch.zeros(1, 1, W, H)
    mask_W_H[:, :, x0:xn, y0:yn] = 1

    mask_W_D = torch.zeros(1, 1, W, D)
    mask_W_D[:, :, x0:xn, :] = 1

    mask_H_D = torch.zeros(1, 1, H, D)
    mask_H_D[:, :, y0:yn, :] = 1

    mask_T_W = torch.zeros(1, 1, T, W)
    mask_T_W[:, :, :, x0:xn] = 1

    mask_T_H = torch.zeros(1, 1, T, H)
    mask_T_H[:, :, :, y0:yn] = 1

    mask_T_D = torch.ones(1, 1, T, D)

    return compose_featmaps(
        mask_W_H,
        mask_W_D,
        mask_H_D,
        mask_T_W,
        mask_T_H,
        mask_T_D,
        txyz=txyz
    )


def get_hexplane_mask_transpose_bbox(txyz, t_range=None, x_range=None, y_range=None):
    T, W, H, D = txyz

    mask_W_H = torch.zeros(1, 1, W, H)
    mask_W_D = torch.zeros(1, 1, W, D)
    mask_H_D = torch.zeros(1, 1, H, D)
    mask_T_W = torch.zeros(1, 1, T, W)
    mask_T_H = torch.zeros(1, 1, T, H)
    mask_T_D = torch.zeros(1, 1, T, D)

    if x_range or y_range:
        x0, xn = x_range if x_range else (0, W)
        y0, yn = y_range if y_range else (0, H)

        mask_W_H[:, :, x0:xn, y0:yn] = 1
        mask_W_D[:, :, x0:xn, :] = 1
        mask_H_D[:, :, y0:yn, :] = 1
        mask_T_W[:, :, :, x0:xn] = 1
        mask_T_H[:, :, :, y0:yn] = 1
        mask_T_D[:, :, :, :] = 1

    if t_range:
        t0, tn = t_range
        mask_T_W[:, :, t0:tn, :] = 1
        mask_T_H[:, :, t0:tn, :] = 1
        mask_T_D[:, :, t0:tn, :] = 1

    return compose_featmaps_transpose(
        mask_W_H,
        mask_W_D,
        mask_H_D,
        mask_T_W,
        mask_T_H,
        mask_T_D,
        txyz=txyz
    )
