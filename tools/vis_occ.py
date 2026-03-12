import os

import mmcv
import open3d as o3d
import numpy as np
import torch
import pickle
import argparse
import cv2
import matplotlib.pyplot as plt

FREE = 0
OCCUPIED = 1
FREE_LABEL = 0
NOISE_LABEL = 255

VOXEL_SIZE = {
    'Occ3D-nuScenes': [0.4, 0.4, 0.4],
    'nuScenes-Occupancy': [0.2, 0.2, 0.2],
}
POINT_CLOUD_RANGE = {
    'Occ3D-nuScenes': [-40, -40, -1, 40, 40, 5.4],
    'nuScenes-Occupancy': [-51.2, -51.2, -5, 51.2, 51.2, 3],
}
SPTIAL_SHAPE = {
    'Occ3D-nuScenes': [200, 200, 16],
    'nuScenes-Occupancy': [512, 512, 40],
}

colormap_to_colors = np.array(
    [
        [255, 255, 255, 255],  # 0 undefined
        [40, 40, 100, 255],  # 1 barrier  orange
        [156, 102, 102, 255],    # 2 bicycle  Blue
        [237, 125, 49, 255],   # 3 bus  Darkslategrey
        [255, 215, 0, 255],  # 4 car  Crimson
        [142, 250, 0, 255],   # 5 cons. Veh  Orangered
        [112, 48, 160, 255],  # 6 motorcycle  Darkorange
        [255, 0, 0, 255], # 7 pedestrian  Darksalmon
        [4, 50, 255, 255],  # 8 traffic cone  Red
        [148, 22, 81, 255],# 9 trailer  Slategrey
        [0, 215, 255, 255],# 10 truck Burlywood
        [240, 240, 240, 255],    # 11 drive sur  Green    [0, 207, 191, 255]
        [132, 151, 176, 255],  # 12 other lat  nuTonomy green
        [255, 52, 179, 255],  # 13 sidewalk
        [23, 135, 31, 255],    # 14 terrain
        [72, 118, 255, 255],    # 15 manmade
        [84, 255, 159, 255],   # 16 vegeyation
], dtype=np.float32)

def visualize_planemap(plane_map, show=True):
    # bev_map = num_cls x H x W   [0,1]mask
    plane_map = plane_map.cpu().numpy() if isinstance(plane_map, torch.Tensor) else plane_map
    vis_plane_map = np.zeros_like(plane_map[0])
    
    # first stuff
    for id, map in enumerate(plane_map[10:]):
        vis_plane_map[map == 1] = id + 11
    # second things: to make sure things visible on top of stuff
    for id, map in enumerate(plane_map[:10]):
        vis_plane_map[map == 1] = id + 1
    
    colors = colormap_to_colors / 255
    vis_plane_map = vis_plane_map.astype(np.uint8)
    vis_plane_map = colors[vis_plane_map]

    fig, axes = plt.subplots(1, 1, figsize=(6, 6))
    axes.imshow(vis_plane_map)
    plt.tight_layout()

    if show:
        plt.show()
    else:
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return img_array

def voxel2points(voxel, occ_show, voxelSize, pc_range):
    """
    Args:
        voxel: (Dx, Dy, Dz)
        occ_show: (Dx, Dy, Dz)
        voxelSize: (dx, dy, dz)

    Returns:
        points: (N, 3) 3: (x, y, z)
        voxel: (N, ) cls_id
        occIdx: (x_idx, y_idx, z_idx)
    """
    points = torch.cat((occ_show[:, 0][:, None] * voxelSize[0] + pc_range[0], \
                        occ_show[:, 1][:, None] * voxelSize[1] + pc_range[1], \
                        occ_show[:, 2][:, None] * voxelSize[2] + pc_range[2]),
                       dim=1)      # (N, 3) 3: (x, y, z)
    return points, voxel[:, -1]

def voxel_profile(voxel, voxel_size):
    """
    Args:
        voxel: (N, 3)  3:(x, y, z)
        voxel_size: (vx, vy, vz)

    Returns:
        box: (N, 7) (x, y, z - dz/2, vx, vy, vz, 0)
    """
    centers = torch.cat((voxel[:, :2], voxel[:, 2][:, None] - voxel_size[2] / 2), dim=1)     # (x, y, z - dz/2)
    # centers = voxel
    wlh = torch.cat((torch.tensor(voxel_size[0]).repeat(centers.shape[0])[:, None],
                     torch.tensor(voxel_size[1]).repeat(centers.shape[0])[:, None],
                     torch.tensor(voxel_size[2]).repeat(centers.shape[0])[:, None]), dim=1)
    yaw = torch.full_like(centers[:, 0:1], 0)
    return torch.cat((centers, wlh, yaw), dim=1)


def rotz(t):
    """Rotation about the z-axis."""
    c = torch.cos(t)
    s = torch.sin(t)
    return torch.tensor([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])

def my_compute_box_3d(center, size, heading_angle):
    """
    Args:
        center: (N, 3)  3: (x, y, z - dz/2)
        size: (N, 3)    3: (vx, vy, vz)
        heading_angle: (N, 1)
    Returns:
        corners_3d: (N, 8, 3)
    """
    h, w, l = size[:, 2], size[:, 0], size[:, 1]
    center[:, 2] = center[:, 2] + h / 2
    l, w, h = (l / 2).unsqueeze(1), (w / 2).unsqueeze(1), (h / 2).unsqueeze(1)
    x_corners = torch.cat([-l, l, l, -l, -l, l, l, -l], dim=1)[..., None]
    y_corners = torch.cat([w, w, -w, -w, w, w, -w, -w], dim=1)[..., None]
    z_corners = torch.cat([h, h, h, h, -h, -h, -h, -h], dim=1)[..., None]
    corners_3d = torch.cat([x_corners, y_corners, z_corners], dim=2)
    corners_3d[..., 0] += center[:, 0:1]
    corners_3d[..., 1] += center[:, 1:2]
    corners_3d[..., 2] += center[:, 2:3]
    return corners_3d

def show_point_cloud(points: np.ndarray, colors=True, points_colors=None, bbox3d=None, voxelize=False,
                     bbox_corners=None, linesets=None, vis=None, offset=[0,0,0], large_voxel=True, voxel_size=0.4):
    """
    :param points: (N, 3)  3:(x, y, z)
    :param colors: false
    :param points_colors: (N, 4）
    :param bbox3d: voxel grid (N, 7) 7: (center, wlh, yaw=0)
    :param voxelize: false
    :param bbox_corners: (N, 8, 3)
    :param linesets
    :return:
    """
    if vis is None:
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window()
    if isinstance(offset, list) or isinstance(offset, tuple):
        offset = np.array(offset)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points+offset)
    if colors:
        pcd.colors = o3d.utility.Vector3dVector(points_colors[:, :3])
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1, origin=[0, 0, 0])

    voxelGrid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    if large_voxel:
        vis.add_geometry(voxelGrid)
    else:
        vis.add_geometry(pcd)

    if voxelize:
        line_sets = o3d.geometry.LineSet()
        line_sets.points = o3d.open3d.utility.Vector3dVector(bbox_corners.reshape((-1, 3))+offset)
        line_sets.lines = o3d.open3d.utility.Vector2iVector(linesets.reshape((-1, 2)))
        line_sets.paint_uniform_color((0, 0, 0))
        vis.add_geometry(line_sets)

    vis.add_geometry(mesh_frame)
    return vis

def show_occ(occ_state, occ_show, voxel_size, pc_range, vis=None, offset=[0, 0, 0]):
    """
    Args:
        occ_state: (Dx, Dy, Dz), cls_id
        occ_show: (Dx, Dy, Dz), bool
        voxel_size: [0.4, 0.4, 0.4]
        vis: Visualizer
        offset:

    Returns:

    """
    colors = colormap_to_colors / 255
    pcd, labels = voxel2points(occ_state, occ_show, voxel_size, pc_range)
    # pcd: (N, 3)  3: (x, y, z)
    # labels: (N, )  cls_id
    _labels = labels % len(colors)
    pcds_colors = colors[_labels.int()]   # (N, 4)

    bboxes = voxel_profile(pcd, voxel_size)    # (N, 7)   7: (x, y, z - dz/2, dx, dy, dz, 0)
    bboxes_corners = my_compute_box_3d(bboxes[:, 0:3], bboxes[:, 3:6], bboxes[:, 6:7])      # (N, 8, 3)

    bases_ = torch.arange(0, bboxes_corners.shape[0] * 8, 8)
    edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]])  # lines along y-axis
    edges = edges.reshape((1, 12, 2)).repeat(bboxes_corners.shape[0], 1, 1)     # (N, 12, 2)
    # (N, 12, 2) + (N, 1, 1) --> (N, 12, 2)
    edges = edges + bases_[:, None, None]

    vis = show_point_cloud(
        points=pcd.numpy(),
        colors=True,
        points_colors=pcds_colors,
        voxelize=False,
        bbox3d=bboxes.numpy(),
        bbox_corners=bboxes_corners.numpy(),
        linesets=edges.numpy(),
        vis=vis,
        offset=offset,
        large_voxel=True,
        voxel_size=0.4
    )
    return vis

def visualize_occ(voxel_coord_label=None, voxel_coord=None, voxel_label=None, dataset_type="Occ3D-nuScenes"):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    view_control = vis.get_view_control()
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1])
    opt.line_width = 80

    if voxel_coord_label is not None:
        voxel_coord_label = voxel_coord_label.cpu().numpy() if isinstance(voxel_coord_label, torch.Tensor) else voxel_coord_label
    elif voxel_coord is not None and voxel_label is not None:
        voxel_coord_label = torch.cat([voxel_coord, voxel_label.unsqueeze(-1)], dim=-1).view(-1, 4)
        voxel_coord_label = voxel_coord_label.cpu().numpy()

    voxel_coord_label = voxel_coord_label[(voxel_coord_label[:, -1] != FREE_LABEL) & (voxel_coord_label[:, -1] != NOISE_LABEL)]
    voxel_show = voxel_coord_label[(voxel_coord_label[:, -1] != FREE_LABEL) & (voxel_coord_label[:, -1] != NOISE_LABEL)][:, :3]

    voxel_size = VOXEL_SIZE[dataset_type]
    pc_range = POINT_CLOUD_RANGE[dataset_type]
    vis = show_occ(torch.from_numpy(voxel_coord_label), torch.from_numpy(voxel_show), 
            voxel_size=voxel_size, pc_range=pc_range, vis=vis, offset=[0, voxel_coord_label.shape[0] * voxel_size[0] * 1.2 * 0, 0])

    vis.poll_events()
    vis.update_renderer()
    vis.run()

def draw_occ_triplane(occupancy, show=False):

    def voxel2plane(voxel_label, plane, num_classes=17):
        H, W, D = voxel_label.shape
        plane_size = {
            'xy': [H, W],
            'xz': [H, D],
            'yz': [W, D], 
        }
        plane_axis = {
            'xy': 2,
            'xz': 1,
            'yz': 0,
        }
        plane_map = np.zeros((num_classes-1, *plane_size[plane]), dtype=np.uint8)  # ignore unoccupied class

        for c in range(1, num_classes):
            mask = (voxel_label == c)
            plane_map[c-1] = np.any(mask, axis=plane_axis[plane]).astype(np.uint8)
        
        return plane_map
    
    def compose_featmaps(feat_xy, feat_xz, feat_yz, transpose=True):
        H, W = feat_xy.shape[1:]
        D = feat_xz.shape[-1]

        empty_block = np.zeros(list(feat_xy.shape[:-2]) + [D, D], dtype=feat_xy.dtype)
        if transpose:
            feat_yz = feat_yz.transpose(0, 2, 1)
        
        composed_map = np.concatenate(
            [np.concatenate([feat_xy, feat_xz], axis=-1),       # C,X,Y+Z
             np.concatenate([feat_yz, empty_block], axis=-1)],  # C,Z,Y+Z
            axis=-2                                             # C,X+Z,Y+Z
        )
        return composed_map
    
    gt_masks_plane_xy = voxel2plane(occupancy, 'xy') # num_cls, H,W
    gt_masks_plane_xz = voxel2plane(occupancy, 'xz') # num_cls, H,D
    gt_masks_plane_yz = voxel2plane(occupancy, 'yz') # num_cls, W,D

    gt_masks_triplane = compose_featmaps(gt_masks_plane_xy, gt_masks_plane_xz, gt_masks_plane_yz)
    if show:
        visualize_planemap(gt_masks_triplane, show=show)
    else:
        return visualize_planemap(gt_masks_triplane, show=show)

def draw_latent_field2D(latents, show=False, save=False, title=None):
    """
    latents: (16, H, W)
    """
    feat_dim, H, W = latents.shape
    selected_latents = latents[::4]
    num_plots = selected_latents.shape[0]

    fig, axes = plt.subplots(1, num_plots, figsize=(16, 4))

    for i in range(num_plots):
        cax = axes[i].imshow(selected_latents[i], cmap='viridis')
        axes[i].set_title(f'{title} -- Channel {i * 4}')

    plt.tight_layout()

    if save:
        plt.savefig('latent_field.png')
    if show:
        plt.show()
    else:
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return img_array


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize the predicted '
                                     'result of nuScenes')
    parser.add_argument(
        'res', help='Path to the predicted result')
    parser.add_argument(
        '--canva-size', type=int, default=1000, help='Size of canva in pixel')
    parser.add_argument(
        '--vis-frames',
        type=int,
        default=500,
        help='Number of frames for visualization')
    parser.add_argument(
        '--scale-factor',
        type=int,
        default=4,
        help='Trade-off between image-view and bev in size of '
        'the visualized canvas')
    parser.add_argument(
        '--version',
        type=str,
        default='val',
        help='Version of nuScenes dataset')
    parser.add_argument('--draw-gt', action='store_true')
    parser.add_argument(
        '--root_path',
        type=str,
        default='./data/nuscenes',
        help='Path to nuScenes dataset')
    parser.add_argument(
        '--save_path',
        type=str,
        default='./vis',
        help='Path to save visualization results')
    parser.add_argument(
        '--format',
        type=str,
        default='image',
        choices=['video', 'image'],
        help='The desired format of the visualization result')
    parser.add_argument(
        '--fps', type=int, default=10, help='Frame rate of video')
    parser.add_argument(
        '--video-prefix', type=str, default='vis', help='name of video')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # load predicted results
    results_dir = args.res

    # load dataset information
    info_path = \
        args.root_path + '/bevdetv2-nuscenes_infos_%s.pkl' % args.version
    dataset = pickle.load(open(info_path, 'rb'))
    # prepare save path and medium
    vis_dir = args.save_path
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    print('saving visualized result to %s' % vis_dir)
    scale_factor = args.scale_factor
    canva_size = args.canva_size
    if args.format == 'video':
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        vout = cv2.VideoWriter(
            os.path.join(vis_dir, '%s.mp4' % args.video_prefix), fourcc,
            args.fps, (int(1600 / scale_factor * 3),
                       int(900 / scale_factor * 2 + canva_size)))

    views = [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ]
    print('start visualizing results')

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    for cnt, info in enumerate(
            dataset['infos'][:min(args.vis_frames, len(dataset['infos']))]):
        if cnt % 10 == 0:
            print('%d/%d' % (cnt, min(args.vis_frames, len(dataset['infos']))))

        scene_name = info['scene_name']
        sample_token = info['token']

        pred_occ_path = os.path.join(results_dir, scene_name, sample_token, 'pred.npz')
        gt_occ_path = info['occ_path']

        pred_occ = np.load(pred_occ_path)['pred']
        gt_data = np.load(os.path.join(args.root_path, gt_occ_path, 'labels.npz'))
        voxel_label = gt_data['semantics']
        lidar_mask = gt_data['mask_lidar']
        camera_mask = gt_data['mask_camera']

        # load imgs
        imgs = []
        for view in views:
            img = cv2.imread(info['cams'][view]['data_path'])
            imgs.append(img)

        # occ_canvas
        voxel_show = np.logical_and(pred_occ != FREE_LABEL, camera_mask)
        # voxel_show = pred_occ != FREE_LABEL
        voxel_size = VOXEL_SIZE
        vis = show_occ(torch.from_numpy(pred_occ), torch.from_numpy(voxel_show), voxel_size=voxel_size, vis=vis,
                       offset=[0, pred_occ.shape[0] * voxel_size[0] * 1.2 * 0, 0])

        if args.draw_gt:
            voxel_show = np.logical_and(voxel_label != FREE_LABEL, camera_mask)
            vis = show_occ(torch.from_numpy(voxel_label), torch.from_numpy(voxel_show), voxel_size=voxel_size, vis=vis,
                           offset=[0, voxel_label.shape[0] * voxel_size[0] * 1.2 * 1, 0])

        view_control = vis.get_view_control()

        look_at = np.array([-0.185, 0.513, 3.485])
        front = np.array([-0.974, -0.055, 0.221])
        up = np.array([0.221, 0.014, 0.975])
        zoom = np.array([0.08])

        view_control.set_lookat(look_at)
        view_control.set_front(front)
        view_control.set_up(up)
        view_control.set_zoom(zoom)

        opt = vis.get_render_option()
        opt.background_color = np.asarray([1, 1, 1])
        opt.line_width = 5

        vis.poll_events()
        vis.update_renderer()
        vis.run()

        occ_canvas = vis.capture_screen_float_buffer(do_render=True)
        occ_canvas = np.asarray(occ_canvas)
        occ_canvas = (occ_canvas * 255).astype(np.uint8)
        occ_canvas = occ_canvas[..., [2, 1, 0]]
        occ_canvas_resize = cv2.resize(occ_canvas, (canva_size, canva_size), interpolation=cv2.INTER_CUBIC)

        vis.clear_geometries()

        big_img = np.zeros((900 * 2 + canva_size * scale_factor, 1600 * 3, 3),
                       dtype=np.uint8)
        big_img[:900, :, :] = np.concatenate(imgs[:3], axis=1)
        img_back = np.concatenate(
            [imgs[3][:, ::-1, :], imgs[4][:, ::-1, :], imgs[5][:, ::-1, :]],
            axis=1)
        big_img[900 + canva_size * scale_factor:, :, :] = img_back
        big_img = cv2.resize(big_img, (int(1600 / scale_factor * 3),
                                       int(900 / scale_factor * 2 + canva_size)))
        w_begin = int((1600 * 3 / scale_factor - canva_size) // 2)
        big_img[int(900 / scale_factor):int(900 / scale_factor) + canva_size,
                w_begin:w_begin + canva_size, :] = occ_canvas_resize

        if args.format == 'image':
            out_dir = os.path.join(vis_dir, f'{scene_name}', f'{sample_token}')
            mmcv.mkdir_or_exist(out_dir)
            for i, img in enumerate(imgs):
                cv2.imwrite(os.path.join(out_dir, f'img{i}.png'), img)
            cv2.imwrite(os.path.join(out_dir, 'occ.png'), occ_canvas)
            cv2.imwrite(os.path.join(out_dir, 'overall.png'), big_img)
        elif args.format == 'video':
            cv2.putText(big_img, f'{cnt:{cnt}}', (5, 15), fontFace=cv2.FONT_HERSHEY_COMPLEX, color=(0, 0, 0),
                        fontScale=0.5)
            cv2.putText(big_img, f'{scene_name}', (5, 35), fontFace=cv2.FONT_HERSHEY_COMPLEX, color=(0, 0, 0),
                        fontScale=0.5)
            cv2.putText(big_img, f'{sample_token[:5]}', (5, 55), fontFace=cv2.FONT_HERSHEY_COMPLEX, color=(0, 0, 0),
                        fontScale=0.5)
            vout.write(big_img)

    if args.format == 'video':
        vout.release()
    vis.destroy_window()

if __name__ == '__main__':
    main()