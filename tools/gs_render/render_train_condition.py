import copy
import os
import pickle
import tqdm
import h5py
import cv2
import numba as nb
import numpy as np
import torch
from tools.gs_render.gaussian_renderer import apply_depth_colormap, apply_semantic_colormap, render, concat_6_views
from pyquaternion import Quaternion
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tools.vis_occ import visualize_occ

cams = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT"]



def voxel2world(voxel, voxel_size=np.array([0.4, 0.4, 0.4]), pc_range=np.array([-40.0, -40.0, -1.0, 40.0, 40.0, 5.4])):
    """
    voxel: [N, 3]
    """
    return voxel * voxel_size[None, :] + pc_range[:3][None, :]


def world2voxel(wolrd, voxel_size=np.array([0.4, 0.4, 0.4]), pc_range=np.array([-40.0, -40.0, -1.0, 40.0, 40.0, 5.4])):
    """
    wolrd: [N, 3]
    """
    return (wolrd - pc_range[:3][None, :]) / voxel_size[None, :]

def load_occ_gt(occ_path, grid_size=np.array([200, 200, 16]), num_class=17, unoccupied=0):
    voxel_label = np.load(occ_path)['semantics']    # H,W,D under ego_vehicle coord

    xyzl = []
    # load voxel_coord and voxel_label
    for i in range(num_class+1):
        xyz = torch.nonzero(torch.Tensor(voxel_label) == i, as_tuple=False)
        xyzlabel = torch.nn.functional.pad(xyz, (0,1),'constant', value=i)
        xyzl.append(xyzlabel)
    xyzl = torch.cat(xyzl, dim=0)
    xyz = xyzl[:, :-1].numpy()
    label = xyzl[:, -1].numpy()

    # map invalid to 255
    invalid = (label == 0).astype(np.uint8)
    label[invalid == 1] = 255     # map invalid to 255
    label[label == 17] = 0  # map free to 0

    pcd_np_cor = voxel2world(xyz + 0.5)  # x y z
    pcd_np_cor = world2voxel(pcd_np_cor)

    # make sure the point is in the grid
    pcd_np_cor = np.clip(pcd_np_cor, np.array([0, 0, 0]), grid_size - 1)
    copy.deepcopy(pcd_np_cor)
    pcd_np = np.concatenate([pcd_np_cor, label[:, None]], axis=-1)

    # 255: noise, 1-16 normal classes, 0 unoccupied
    pcd_np = pcd_np[np.lexsort((pcd_np_cor[:, 0], pcd_np_cor[:, 1], pcd_np_cor[:, 2])), :]
    pcd_np = pcd_np.astype(np.int64)
    pcd_np[pcd_np[:, -1] == 0, -1] = 255
    
    processed_label = np.ones(grid_size, dtype=np.uint8) * unoccupied
    processed_label = nb_process_label(processed_label, pcd_np)
    noise_mask = processed_label == 255
    processed_label[noise_mask] = 0
    return processed_label

def load_nuscenes_occupancy(occ_path, grid_size=np.array([512, 512, 40]), unoccupied=0):
    voxel_label = np.load(occ_path) #  [z y x cls]
    
    label = voxel_label[..., -1:]
    label[label==0] = 255   # map invalid to 255

    # occupied -> voxel
    occupied_coord_label = voxel_label[..., [2,1,0,3]]   # [x y z cls]
    processed_label = np.ones(grid_size, dtype=np.uint8) * unoccupied
    processed_label[occupied_coord_label[..., 0], occupied_coord_label[..., 1], occupied_coord_label[..., 2]] = occupied_coord_label[..., 3]
    invalid = (processed_label == 255).astype(np.int64)

    # transform
    pcd_np_cor = voxel2world(voxel_label[..., [2,1,0]] + 0.5, voxel_size=np.array([0.2, 0.2, 0.2]), pc_range=np.array([-51.2, -51.2, -5, 51.2, 51.2, 3]))
    pcd_np_cor = world2voxel(pcd_np_cor, voxel_size=np.array([0.2, 0.2, 0.2]), pc_range=np.array([-51.2, -51.2, -5, 51.2, 51.2, 3]))

    # make sure the point is in the grid
    pcd_np_cor = np.clip(pcd_np_cor, np.array([0,0,0]), grid_size - 1)
    pcd_np = np.concatenate([pcd_np_cor, label], axis=-1)

    # 255: noise, 1-16 normal classes, 0 unoccupied
    pcd_np = pcd_np[np.lexsort((pcd_np_cor[:, 0], pcd_np_cor[:, 1], pcd_np_cor[:, 2])), :]
    pcd_np = pcd_np.astype(np.int64)
    pcd_np[pcd_np[:, -1] == 0, -1] = 255

    processed_label = np.ones(grid_size, dtype=np.uint8) * unoccupied
    processed_label = nb_process_label(processed_label, pcd_np)
    noise_mask = processed_label == 255
    processed_label[noise_mask] = 0
    return processed_label

# u1: uint8, u8: uint16, i8: int64
@nb.jit("u1[:,:,:](u1[:,:,:],i8[:,:])", nopython=True, cache=True, parallel=False)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)

    return processed_label


def create_full_center_coords(shape=(200, 200, 16), x_range=(-40.0, 40.0), y_range=(-40.0, 40.0), z_range=(-1, 5.4)):
    x = torch.linspace(x_range[0], x_range[1], shape[0]).view(-1, 1, 1).expand(shape)
    y = torch.linspace(y_range[0], y_range[1], shape[1]).view(1, -1, 1).expand(shape)
    z = torch.linspace(z_range[0], z_range[1], shape[2]).view(1, 1, -1).expand(shape)

    center_coords = torch.stack((x, y, z), dim=-1)

    return center_coords

def interpolate_bevlayout(bevlayout, bev_replace_idx, target_shape):
    interpolated_maps = []
    for idx in bev_replace_idx:
        mask_map = bevlayout[idx]
        interpolated_map = cv2.resize(mask_map.astype(np.float32), (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
        interpolated_map = (interpolated_map > 0.5).astype(np.uint8)
        interpolated_maps.append(interpolated_map)
    return interpolated_maps

def replace_occ_grid_with_bev(
    input_occ, bevlayout, driva_area_idx=11, bev_replace_idx=[1, 5, 6], occ_replace_new_idx=[17, 18, 19]
):
    # self.classes= ['drivable_area','ped_crossing','walkway','stop_line','carpark_area','road_divider','lane_divider','road_block']
    # occ road [11] drivable area

    # default shape: input_occ: [200,200,16]; bevlayout: [8,200,200]
    # ped_crossing: bevlayout[1,:,:] -> occ[17][occ[11]==1]=1   occ[17] = ped_crossing
    # road_divider: bevlayout[5,:,:] -> occ[18][occ[11]==1]=1   occ[18] = road_divider
    # lane_divider: bevlayout[6,:,:] -> occ[19][occ[11]==1]=1   occ[19] = lane_divider

    road_divider_mask = bevlayout[5, :, :].astype(np.uint8)
    lane_divider_mask = bevlayout[6, :, :].astype(np.uint8)

    # road_divider_mask = cv2.dilate(road_divider_mask, np.ones((3, 3), np.uint8))
    # lane_divider_mask = cv2.dilate(lane_divider_mask, np.ones((1, 1), np.uint8))

    bevlayout[5, :, :] = road_divider_mask.astype(bool)
    bevlayout[6, :, :] = lane_divider_mask.astype(bool)

    n = len(bev_replace_idx)
    x_max, y_max = input_occ.shape[0], input_occ.shape[1]
    output_occ = input_occ.copy()  # numpy copy() ; tensor clone()
    # occ3d 200x200x16
    if bevlayout.shape[1] == x_max and bevlayout.shape[2] == y_max:
        bev_replace_mask = []
        for i in range(n):
            bev_replace_mask.append(bevlayout[bev_replace_idx[i]] == 1)
    # nuscene_occupancy 512x512x40
    else:
        bev_replace_mask = interpolate_bevlayout(bevlayout, bev_replace_idx, target_shape=(x_max, y_max))

    # drive_area_coords = np.stack(np.where(input_occ == driva_area_idx), axis=-1)
    for x in range(x_max):
        for y in range(y_max):
            for i in range(n):
                if bev_replace_mask[i][x, y]:
                    occupancy_data = input_occ[x, y, :]

                    if driva_area_idx in occupancy_data:
                        max_11_index = np.where(occupancy_data == driva_area_idx)
                        output_occ[x, y, max_11_index] = occ_replace_new_idx[i]
                    # else:
                    #     dists = np.linalg.norm(drive_area_coords[:,:2] - np.array([x, y]), axis=1)
                    #     closest_index = np.argmin(dists)
                    #     output_occ[x, y, drive_area_coords[closest_index, 2]] = occ_replace_new_idx[i]
    return output_occ

@nb.njit
def one_hot_decode(data: np.ndarray, n: int):
    """
    returns (h, w, n) np.int64 {0, 1}
    """
    # shift = np.arange(n, dtype=np.int32)[None, None]
    shift = np.zeros((1, 1, n), np.int32)
    shift[0, 0, :] = np.arange(0, n, 1, np.int32)

    # x = np.array(data)[..., None]
    x = np.zeros((*data.shape, 1), data.dtype)
    x[..., 0] = data
    # after shift, numpy keeps int32, numba changes dtype to int64
    x = (x >> shift) & 1  # only keep the lowest bit, for each n

    x = x.transpose(2, 0, 1)
    return x

def load_occ_layout(sample_token, layout_path, num_map_classes=8):
    with h5py.File(layout_path, 'r') as cache_file:
        layout = one_hot_decode(
            cache_file['gt_masks_bev_static'][sample_token][:], num_map_classes)
    return layout


def render_occ_semantic_map(item_data, base_path, occ_base_path, layout_base_path, is_vis=False, scale_fator=0.15):
    scene_name = item_data["scene_name"]
    sample_token = item_data["token"]
    scene_token = item_data["scene_token"]
    lidar_token = item_data["lidar_token"]

    os.makedirs(os.path.join(base_path, scene_name, sample_token), exist_ok=True)
    sem_data_all_path = os.path.join(base_path, scene_name, sample_token, "semantic.npz")
    depth_data_all_path = os.path.join(base_path, scene_name, sample_token, "depth_data.npz")
    if os.path.exists(sem_data_all_path) and os.path.exists(depth_data_all_path):
        return

    # NOTE: Occ3D
    occ_path = os.path.join(occ_base_path, scene_name, sample_token, "labels.npz")
    occ_label = load_occ_gt(occ_path=occ_path, grid_size=np.array([200, 200, 16]))  # H,W,D=200,200,16 under ego_vehicle coord
    # NOTE: nuscene-occupancy
    # occ_path = os.path.join(occ_base_path, f'scene_{scene_token}', 'occupancy', f'{lidar_token}.npy')
    # occ_label = load_nuscenes_occupancy(occ_path=occ_path, grid_size=np.array([512, 512, 40]))  # H,W,D=512,512,40 under lidar coord

    bevlayout = load_occ_layout(sample_token=sample_token, layout_path=layout_base_path)
    bevlayout = torch.from_numpy(bevlayout.astype(np.float64))
    # NOTE: occ3d need rot90 to align with ego_coord
    # NOTE: nuscenes-occupancy do not need rot90 to keep as lidar_coord
    bevlayout = torch.rot90(bevlayout, k=3, dims=(1, 2))
    bevlayout = bevlayout.numpy()

    semantics = occ_label
    semantics = replace_occ_grid_with_bev(semantics, bevlayout)

    image_shape = (448, 800)

    semantics = torch.from_numpy(semantics.astype(np.float32))  # 200, 200, 16
    xyz = (
        ## NOTE: occ3d:200x200x16 nuscene_occupancy:512x512x40
        create_full_center_coords(shape=(200, 200, 16), x_range=(-40.0, 40.0), y_range=(-40.0, 40.0), z_range=(-1, 5.4))
        # create_full_center_coords(shape=(512, 512, 40), x_range=(-51.2, 51.2), y_range=(-51.2, 51.2), z_range=(-5, 3))
        .view(-1, 3)
        .cuda()
        .float()
    )

    # load semantic data ------------------------------------------------------------------------------
    semantics_gt = semantics.view(-1, 1)  # (200, 200, 16) -> (640000, 1)
    occ_mask = semantics_gt[:, 0] != 0
    semantics_gt = semantics_gt.permute(1, 0)

    opacity = (semantics_gt.clone() != 0).float()
    opacity = opacity.permute(1, 0).cuda()  # (640000, 1) non-empty=1, empty=0

    semantics = torch.zeros((20, semantics_gt.shape[1])).cuda().float() # (20, 640000) one-hot  0:void  1~16:occ  17:ped_crossing  18:road_divider  19:lane_divider
    color = torch.zeros((3, semantics_gt.shape[1])).cuda()
    for i in range(20):
        semantics[i] = semantics_gt == i

    rgb = color.permute(1, 0).float()
    feat = semantics.permute(1, 0).float()
    rot = torch.zeros((xyz.shape[0], 4)).cuda().float()
    rot[:, 0] = 1
    scale = torch.ones((xyz.shape[0], 3)).cuda().float() * scale_fator

    camera_semantic = []
    camera_depth = []
    semantic_images = []
    depth_images = []
    camera_images = []

    for cam in cams:
        cam_info = item_data["cams"][cam]
        camera_intrinsic = np.eye(3).astype(np.float32)
        camera_intrinsic[:3, :3] = cam_info["camera_intrinsics"]
        camera_intrinsic = torch.from_numpy(camera_intrinsic).cuda().float()

        # NOTE: occ3d:camera2ego
        c2e = Quaternion(cam_info["sensor2ego_rotation"]).transformation_matrix
        c2e[:3, 3] = np.array(cam_info["sensor2ego_translation"])
        c2e = torch.from_numpy(c2e).cuda().float()
        camera_extrinsic = c2e

        # # # NOTE: nuscene-occupancy:lidar2ego
        # c2l = np.eye(4).astype(np.float32)
        # c2l[:3, :3] = cam_info["sensor2lidar_rotation"]
        # c2l[:3, 3] = np.array(cam_info["sensor2lidar_translation"])
        # c2l = torch.from_numpy(c2l).cuda().float()
        # camera_extrinsic = c2l

        camera_intrinsic[0][0] = camera_intrinsic[0][0] / 2
        camera_intrinsic[1][1] = camera_intrinsic[1][1] / 2
        camera_intrinsic[0][2] = camera_intrinsic[0][2] / 2
        camera_intrinsic[1][2] = camera_intrinsic[1][2] / 2

        render_pkg = render(
            camera_extrinsic,
            camera_intrinsic,
            image_shape,
            xyz[occ_mask],
            rgb[occ_mask],
            feat[occ_mask],
            rot[occ_mask],
            scale[occ_mask],
            opacity[occ_mask],
            bg_color=[0, 0, 0],
        )

        render_pkg["render_color"]
        render_semantic = render_pkg["render_feat"]
        render_depth = render_pkg["render_depth"]
        render_pkg["render_alpha"]

        if is_vis:
            semantic_image = apply_semantic_colormap(render_semantic).cpu().permute(1, 2, 0).detach().numpy() * 255
            semantic_images.append(Image.fromarray(semantic_image.astype(np.uint8)))

            depth_image = apply_depth_colormap(render_depth).cpu().permute(1, 2, 0).detach().numpy() * 255
            depth_images.append(Image.fromarray(depth_image.astype(np.uint8)))

            camera_images.append(Image.open(cam_info['data_path']).resize((image_shape[1], image_shape[0])))

        semantic = torch.max(render_semantic, dim=0)[1].squeeze().cpu().numpy().astype(np.int8)
        camera_semantic.append(semantic)

        depth_data = render_depth[0].detach().cpu().numpy()
        camera_depth.append(depth_data)

    # vis
    if is_vis:
        alpha = 0.3

        semantic_images = concat_6_views(semantic_images)
        # semantic_images.save(os.path.join(base_path, scene_name, sample_token) + "/semantic_color.jpg")
        # transform semantic image to RGBA
        semantic_images = semantic_images.convert("RGBA")
        semantic_with_alpha = semantic_images.copy()
        alpha_channel = semantic_with_alpha.split()[3].point(lambda p: int(p * alpha))
        semantic_with_alpha.putalpha(alpha_channel)

        depth_images = concat_6_views(depth_images)
        # depth_images.save(os.path.join(base_path, scene_name, sample_token) + "/depth_color.jpg")
        # transform depth image to RGBA
        depth_images = depth_images.convert("RGBA")
        depth_with_alpha = depth_images.copy()
        alpha_channel = depth_with_alpha.split()[3].point(lambda p: int(p * alpha))
        depth_with_alpha.putalpha(alpha_channel)

        camera_images = concat_6_views(camera_images)
        camera_images = camera_images.convert("RGBA")
        # camera_images.save(os.path.join(base_path, scene_name, sample_token) + "/camera_color.jpg")
        blended_semantic = Image.alpha_composite(camera_images, semantic_with_alpha)
        blended_depth = Image.alpha_composite(camera_images, depth_with_alpha)
        blended_semantic.save(os.path.join(base_path, scene_name, sample_token) + "/semantic_color.png")
        blended_depth.save(os.path.join(base_path, scene_name, sample_token) + "/depth_color.png")

    ################################### update object to local ####################################
    np.savez(sem_data_all_path, camera_semantic)
    np.savez(depth_data_all_path, camera_depth)

    # print(f"Rendered {sample_token} to {base_path}/{scene_name}/{sample_token}")


def process_item(item, render_base_path, occ_base_path, layout_base_path, is_vis, scale_factor):
    """
    """
    render_occ_semantic_map(
        item, render_base_path, occ_base_path=occ_base_path, layout_base_path=layout_base_path, is_vis=is_vis, scale_fator=scale_factor
    )


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--dataset_path", type=str, default="./data/nuscenes/")
    parser.add_argument("--info_path", type=str, default="./data/nuscenes/nuscenes_mmdet3d-keyframes/")
    parser.add_argument("--occ_path", type=str, default="./data/nuscenes/gts/")
    parser.add_argument("--layout_path", type=str, default="./data/nuscenes/nuscenes_map_aux_12Hz_interp/")
    parser.add_argument("--render_path", type=str, default="./data/nuscenes/occ_render_map/")
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--scale_factor", type=float, default=0.135) # nuscenes-occupancy:0.07  occ3d:0.135
    parser.add_argument("--num_threads", type=int, default=10, help="Number of threads to use")

    args = parser.parse_args()

    # render train data
    render_base_path = os.path.join(args.render_path)
    occ_base_path = os.path.join(args.occ_path)
    layout_base_path = os.path.join(args.layout_path, f"{args.split}_200x200_12Hz_interp.h5")

    # data_info
    info_path = os.path.join(args.info_path, f"nuscenes_infos_{args.split}.pkl")
    data = pickle.load(open(info_path, "rb"))
    items_data = data["infos"]

    # for index, item in tqdm.tqdm(enumerate(items_data), total=len(items_data)):
    #     render_occ_semantic_map(
    #         item, render_base_path, occ_base_path=occ_base_path, layout_base_path=layout_base_path, is_vis=args.vis, scale_fator=args.scale_factor
    #     )

    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = [
            executor.submit(process_item, item, render_base_path, occ_base_path, layout_base_path, args.vis, args.scale_factor)
            for item in items_data
        ]

        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            try:
                future.result()
            except Exception as e:
                print(f"running error: {e}")
