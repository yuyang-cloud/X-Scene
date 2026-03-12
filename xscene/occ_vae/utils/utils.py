from prettytable import PrettyTable
import os
import torch
import yaml
import numpy as np
from functools import lru_cache
from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def compose_featmaps(feat_xy, feat_xz, feat_yz, tri_size=(128,128,16) , transpose=True):
    H, W, D = tri_size

    empty_block = torch.zeros(list(feat_xy.shape[:-2]) + [D, D], dtype=feat_xy.dtype, device=feat_xy.device)
    if transpose:
        feat_yz = feat_yz.transpose(-1, -2)
    composed_map = torch.cat(                          # C,X,Y+Z
        [torch.cat([feat_xy, feat_xz], dim=-1),        # C,Z,Y+Z
         torch.cat([feat_yz, empty_block], dim=-1)],   # C,X+Z,Y+Z
        dim=-2
    )
    return composed_map, (H, W, D)


def decompose_featmaps(composed_map, tri_size=(128,128,16) , transpose=True):
    H, W, D = tri_size
    feat_xy = composed_map[..., :H, :W] # (C, H, W)
    feat_xz = composed_map[..., :H, W:] # (C, H, D)
    feat_yz = composed_map[..., H:, :W] # (C, W, D)
    if transpose:
        return feat_xy, feat_xz, feat_yz.transpose(-1, -2)
    else:
        return feat_xy, feat_xz, feat_yz

def unpack(compressed):
    ''' given a bit encoded voxel grid, make a normal voxel grid out of it.  '''
    uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
    uncompressed[::8] = compressed[:] >> 7 & 1
    uncompressed[1::8] = compressed[:] >> 6 & 1
    uncompressed[2::8] = compressed[:] >> 5 & 1
    uncompressed[3::8] = compressed[:] >> 4 & 1
    uncompressed[4::8] = compressed[:] >> 3 & 1
    uncompressed[5::8] = compressed[:] >> 2 & 1
    uncompressed[6::8] = compressed[:] >> 1 & 1
    uncompressed[7::8] = compressed[:] & 1
    return uncompressed

def load_label(path, learning_map, grid_size):
    label = np.fromfile(path, dtype=np.uint16).reshape((-1, 1))
    label = learning_map[label]
    label = torch.from_numpy(label).squeeze().type(torch.LongTensor).cuda().reshape(grid_size)
    label[label==255]=0
    return label

def write_result(args):
    os.umask(0)
    os.makedirs(args.save_path, mode=0o777, exist_ok=True)
    args_table = PrettyTable(['Arg', 'Value'])
    for arg, val in vars(args).items():
        args_table.add_row([arg, val])
    with open(os.path.join(args.save_path, 'results.txt'), "w") as f:
        f.write(str(args_table))

def point2voxel(args, preds, coords):
    if len(args.grid_size)==4:
        output = torch.zeros((preds.shape[0], args.grid_size[1], args.grid_size[2], args.grid_size[3]), device=preds.device)
    else :
        output = torch.zeros((preds.shape[0], args.grid_size[0], args.grid_size[1], args.grid_size[2]), device=preds.device)
    for i in range(preds.shape[0]):
        output[i, coords[i, :, 0], coords[i, :, 1], coords[i, :, 2]] = preds[i]
    return output

def point2voxel_class(args, preds, coords):
    output = torch.zeros((preds.shape[0], args.grid_size[0], args.grid_size[1], args.grid_size[2], preds.shape[-1]), device=preds.device, dtype=preds.dtype)
    for i in range(preds.shape[0]):
        output[i, coords[i, :, 0], coords[i, :, 1], coords[i, :, 2], :] = preds[i]
    return output

def visualization(args, coords, preds, folder, idx, learning_map_inv, training):
    output = point2voxel(args, preds, coords)
    return save_remap_lut(args, output, folder, idx, learning_map_inv, training)

def save_remap_lut(args, pred, folder, idx, learning_map_inv, training, make_numpy=True):
    if make_numpy:
        pred = pred.cpu().long().data.numpy()

    if learning_map_inv is not None:
        maxkey = max(learning_map_inv.keys())
        # +100 hack making lut bigger just in case there are unknown labels
        remap_lut_First = np.zeros((maxkey + 100), dtype=np.int32)
        remap_lut_First[list(learning_map_inv.keys())] = list(learning_map_inv.values())

        pred = pred.astype(np.uint32)
        pred = pred.reshape((-1))
        upper_half = pred >> 16  # get upper half for instances
        lower_half = pred & 0xFFFF  # get lower half for semantics
        lower_half = remap_lut_First[lower_half]  # do the remapping of semantics
        pred = (upper_half << 16) + lower_half  # reconstruct full label
        pred = pred.astype(np.uint32)

    if training:
        final_preds = pred.astype(np.uint16)        
        os.umask(0)
        os.makedirs(args.save_path+'/sample/'+str(folder), mode=0o777, exist_ok=True)
        if torch.is_tensor(idx):
            save_path = args.save_path+'/sample/'+str(folder)+'/'+str(idx.item()).zfill(3)+'.label'
        else : 
            save_path = args.save_path+'/sample/'+str(folder)+'/'+str(idx).zfill(3)+'.label'
        final_preds.tofile(save_path)
    else:
        return pred.astype(np.uint16)  
    

def cycle(dl):
    while True:
        for data in dl:
            yield data

@lru_cache(4)
def voxel_coord(voxel_shape):
    x = np.arange(voxel_shape[0])
    y = np.arange(voxel_shape[1])
    z = np.arange(voxel_shape[2])
    Y, X, Z = np.meshgrid(x, y, z)
    voxel_coord = np.concatenate((X[..., None], Y[..., None], Z[..., None]), axis=-1)
    return voxel_coord


def make_query(grid_size):
    gs = grid_size[1:]
    coords = torch.from_numpy(voxel_coord(gs))
    coords = coords.reshape(-1, 3)
    query = torch.zeros(coords.shape, dtype=torch.float32)
    query[:,0] = 2*coords[:,0]/float(gs[0]-1) -1
    query[:,1] = 2*coords[:,1]/float(gs[1]-1) -1
    query[:,2] = 2*coords[:,2]/float(gs[2]-1) -1
    
    query = query.reshape(-1, 3)
    return coords.unsqueeze(0), query.unsqueeze(0)

def analyze_latent_distributions(model, dataloader, device='cuda'):
    model.eval()
    all_xy_means = []
    all_xz_means = []
    all_yz_means = []

    all_xy_logvars = []
    all_xz_logvars = []
    all_yz_logvars = []

    with torch.no_grad():
        for vox, _, _, _, path, invalid in tqdm(dataloader):
            vox = vox.type(torch.LongTensor).to(device)
            invalid = invalid.type(torch.LongTensor).to(device)
            vox[invalid == 1] = 0
            (_, (xy_mean, xz_mean, yz_mean), (xy_logvar, xz_logvar, yz_logvar)) = model.encode(vox)

            bs, C, H, W = xy_mean.shape
            all_xy_means.append(xy_mean.flatten(2).mean(dim=2).cpu())     # [bs, C]
            all_xz_means.append(xz_mean.flatten(2).mean(dim=2).cpu())
            all_yz_means.append(yz_mean.flatten(2).mean(dim=2).cpu())

            all_xy_logvars.append(xy_logvar.flatten(2).mean(dim=2).cpu()) # [bs, C]
            all_xz_logvars.append(xz_logvar.flatten(2).mean(dim=2).cpu())
            all_yz_logvars.append(yz_logvar.flatten(2).mean(dim=2).cpu())

    # mean
    xy_means = torch.cat(all_xy_means, dim=0).numpy()
    xz_means = torch.cat(all_xz_means, dim=0).numpy()
    yz_means = torch.cat(all_yz_means, dim=0).numpy()

    # logvar
    xy_logvars = torch.cat(all_xy_logvars, dim=0).numpy()
    xz_logvars = torch.cat(all_xz_logvars, dim=0).numpy()
    yz_logvars = torch.cat(all_yz_logvars, dim=0).numpy()

    print(f"xy_means shape: {xy_means.shape}")

    # --- 均值分布直方图 ---
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.hist(xy_means.flatten(), bins=50, color='r', alpha=0.7)
    plt.title('xy_mean distribution')
    plt.subplot(1, 3, 2)
    plt.hist(xz_means.flatten(), bins=50, color='g', alpha=0.7)
    plt.title('xz_mean distribution')
    plt.subplot(1, 3, 3)
    plt.hist(yz_means.flatten(), bins=50, color='b', alpha=0.7)
    plt.title('yz_mean distribution')
    plt.tight_layout()
    plt.show()

    # --- logvar 分布直方图 ---
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.hist(xy_logvars.flatten(), bins=50, color='r', alpha=0.7)
    plt.title('xy_logvar distribution')
    plt.subplot(1, 3, 2)
    plt.hist(xz_logvars.flatten(), bins=50, color='g', alpha=0.7)
    plt.title('xz_logvar distribution')
    plt.subplot(1, 3, 3)
    plt.hist(yz_logvars.flatten(), bins=50, color='b', alpha=0.7)
    plt.title('yz_logvar distribution')
    plt.tight_layout()
    plt.show()

    # 协方差矩阵热图 (xy_mean)
    cov = np.cov(xy_means, rowvar=False)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cov, cmap='viridis')
    plt.title('xy_mean covariance matrix')
    plt.show()

    # PCA 可视化
    pca = PCA(n_components=2)
    xy_pca = pca.fit_transform(xy_means)
    plt.figure(figsize=(5, 5))
    plt.scatter(xy_pca[:, 0], xy_pca[:, 1], s=3, alpha=0.5)
    plt.title('xy_mean PCA 2D')
    plt.show()

    # t-SNE 可视化
    n_samples = xy_means.shape[0]
    perplexity = min(30, max(2, n_samples // 3))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    xy_tsne = tsne.fit_transform(xy_means)
    plt.figure(figsize=(5, 5))
    plt.scatter(xy_tsne[:, 0], xy_tsne[:, 1], s=3, alpha=0.5)
    plt.title('xy_mean t-SNE 2D')
    plt.show()

    breakpoint()  # For debugging purposes, you can remove this line later