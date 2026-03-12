import os
import numpy as np
import numba as nb
from torch.utils import data
import pickle
import pathlib
import torch
import tqdm
from pyquaternion import Quaternion
from scipy.ndimage import distance_transform_edt
from tools.vis_occ import visualize_occ

class NuScenesDatasetOcc(data.Dataset):
    def __init__(self, args, imageset='train', get_query=True):
        if imageset == 'train':
            info_path = args.train_info_path
        elif imageset == 'val':
            info_path = args.val_info_path
        
        with open(info_path, 'rb') as f:
            data = pickle.load(f)
        self.nusc_infos = data['infos']
            
        self.args = args
        self.get_query = get_query
        self.imageset = imageset
        self.dataset = args.dataset
        self.grid_size = np.array(args.grid_size)
        self.pc_range = np.array(args.pc_range)
        self.voxel_size = (self.pc_range[3:] - self.pc_range[:3]) / self.grid_size
        self.data_path = args.data_path
        self.num_class = args.num_class
        self.unoccupied = 0

        if self.dataset == 'Occ3D-nuScenes':
            self.label_mapping = {
                0: 17,  # GO
                1: 1,   # Barrier
                2: 2,   # Bicycle
                3: 3,   # Bus
                4: 4,   # Car
                5: 5,   # Construction vehicle
                6: 6,   # Motorcycle
                7: 7,   # Pedestrian
                8: 8,   # Traffic cone
                9: 9,   # Trailer
                10: 10, # Truck
                11: 11, # Drivable surface
                12: 12, # Other flat
                13: 13, # Sidewalk
                14: 14, # Terrain
                15: 15, # Manmade
                16: 16, # Vegetation
                17: 0,  # FREE
            }
            self.learning_map = np.array(list(self.label_mapping.values()))
            self.max_points = 200000
        elif self.dataset == 'nuScenes-Occupancy':
            self.max_points = 400000
        
        if imageset == 'train':
            if self.dataset == 'Occ3D-nuScenes':
                complt_num_per_class = np.asarray([1.71263908e+10, 3.01297000e+06, 2.34046000e+05, 5.38540200e+06, 3.41464940e+07, 2.04412400e+06, 3.25765000e+05, 3.33025300e+06, 5.43815000e+05, 5.78507900e+06, 1.35211120e+07, 1.98278651e+08, 4.89589500e+06, 5.65404710e+07, 6.65046170e+07, 2.27803562e+08, 2.52374615e+08, 2.08234900e+06])
            elif self.dataset == 'nuScenes-Occupancy':
                complt_num_per_class = np.asarray([2.87652912e+11, 9.26721500e+06, 7.00130000e+05, 2.13368000e+07, 1.25642499e+08, 1.03279170e+07, 1.15663200e+06, 9.77423800e+06, 4.00548500e+06, 2.58165320e+07, 5.51231480e+07, 2.25471591e+09, 5.09567950e+07, 6.45964706e+08, 8.68889964e+08, 1.44610348e+09, 1.72439486e+09])
            compl_labelweights = complt_num_per_class / np.sum(complt_num_per_class)
            self.weights = torch.Tensor(np.power(np.amax(compl_labelweights) / compl_labelweights, 1 / 3.0)).cuda()
            
        elif imageset == 'val':
            self.weights = torch.Tensor(np.ones(self.num_class) * 3).cuda()
            self.weights[0] = 1
            
        elif imageset == 'test':
            self.weights = torch.Tensor(np.ones(self.num_class) * 3).cuda()
            self.weights[0] = 1
        else:
            raise Exception('Split must be train/val/test')
        
        self.im_idx=[]
        if self.dataset == 'Occ3D-nuScenes':
            self.lidar2ego = []
            for scene_name in list(self.nusc_infos.keys()):
                for frame in self.nusc_infos[scene_name]:
                    token = frame['token']
                    label_file = os.path.join(self.data_path, f'{scene_name}/{token}/labels.npz')
                    self.im_idx.append(label_file)
                    # lidar to ego transform
                    lidar2ego = np.eye(4).astype(np.float32)
                    lidar2ego[:3, :3] = Quaternion(
                        frame["lidar2ego_rotation"]).rotation_matrix
                    lidar2ego[:3, 3] = frame["lidar2ego_translation"]
                    self.lidar2ego.append(lidar2ego)
        elif self.dataset == 'nuScenes-Occupancy':
            for frame in self.nusc_infos:
                scene_token = frame['scene_token']
                lidar_token = frame['lidar_token']
                label_file = os.path.join(self.data_path, 'scene_'+scene_token, 'occupancy', lidar_token+'.npy')
                self.im_idx.append(label_file)
        print(('Successfully loaded {} dataset with {} samples.').format(imageset, len(self.im_idx)))

    def voxel2world(self, voxel):
        """
        voxel: [N, 3]
        """
        return voxel * self.voxel_size[None, :] + self.pc_range[:3][None, :]

    def world2voxel(self, world):
        """
        world: [N, 3]
        """
        return (world - self.pc_range[:3][None, :]) / self.voxel_size[None, :]
    
    def world2voxel_with_pcrange(self, world, pc_range):
        """
        world: [N, 3]
        """
        return (world - pc_range[:3][None, :]) / self.voxel_size[None, :]
    
    def ego2lidar(self, xyz_ego, lidar2ego):
        """
        xyz_ego: [N, 3]
        lidar2ego: [4, 4]
        """
        ego2lidar = np.linalg.inv(lidar2ego)
        xyz_ego = np.concatenate([xyz_ego, np.ones((len(xyz_ego), 1))], axis=-1)
        pc_range_ego = np.concatenate([self.pc_range[:3][None], self.pc_range[3:][None]], axis=0)   # self.pc_range under ego_vehicle coord
        pc_range_ego = np.concatenate([pc_range_ego, np.ones((len(pc_range_ego), 1))], axis=-1)
        xyz_lidar = (xyz_ego @ ego2lidar.T)[:, :3]  # transform xyz to lidar_coord
        pc_range_lidar = (pc_range_ego @ ego2lidar.T)[:, :3]    # transform pc_range to lidar_coord
        pc_range_lidar = np.concatenate([pc_range_lidar.min(0), pc_range_lidar.max(0)], axis=0)
        return xyz_lidar, pc_range_lidar

    def load_occ3d_nuscenes(self, index):
        path = self.im_idx[index]
        if self.imageset == 'test':
            voxel_label = np.zeros(self.grid_size, dtype=int)
        else:
            voxel_label = np.load(path)['semantics']    # H,W,D under ego_vehicle coord
        voxel_label = self.learning_map[voxel_label]  # map

        invalid = (voxel_label == 255).astype(np.int64)
        return voxel_label, invalid
    
    def load_nuscenes_occupancy(self, index):
        path = self.im_idx[index]
        if self.imageset == 'test':
            voxel_label = np.zeros(self.grid_size, dtype=int)
        else:
            voxel_label = np.load(path) #  [z y x cls]
        
        label = voxel_label[..., -1:]
        label[label==0] = 255   # map invalid to 255

        # occupied -> voxel
        occupied_coord_label = voxel_label[..., [2,1,0,3]]   # [x y z cls]
        processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.unoccupied
        processed_label[occupied_coord_label[..., 0], occupied_coord_label[..., 1], occupied_coord_label[..., 2]] = occupied_coord_label[..., 3]
        invalid = (processed_label == 255).astype(np.int64)
        return processed_label, invalid, occupied_coord_label

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)
    
    def __getitem__(self, index):
        if self.dataset == 'Occ3D-nuScenes':
            voxel_label, invalid = self.load_occ3d_nuscenes(index)

            if self.get_query :
                if self.imageset == 'train' :
                    p = torch.randint(0, 6, (1,)).item()
                    if p == 0:
                        voxel_label, invalid = flip(voxel_label, invalid, flip_dim=0)
                    elif p == 1:
                        voxel_label, invalid = flip(voxel_label, invalid, flip_dim=1)
                    elif p == 2:
                        voxel_label, invalid = flip(voxel_label, invalid, flip_dim=0)
                        voxel_label, invalid = flip(voxel_label, invalid, flip_dim=1)
                query, xyz_label, xyz_center = get_query(voxel_label, self.num_class, self.grid_size, self.max_points)

            else : 
                query, xyz_label, xyz_center = torch.zeros(1), torch.zeros(1), torch.zeros(1)

        elif self.dataset == 'nuScenes-Occupancy':
            voxel_label, invalid, occupied_coord_label = self.load_nuscenes_occupancy(index)

            if self.get_query:
                if occupied_coord_label.shape[0] > self.max_points:
                    idx = np.random.choice(occupied_coord_label.shape[0], self.max_points, replace=False)
                    xyzl = occupied_coord_label[idx]
                else:
                    # free space near object
                    NEIGHBOR_OFFSETS = np.array([
                        [-1,0,0], [1,0,0],  # x
                        [0,-1,0], [0,1,0],  # y
                        [0,0,-1], [0,0,1]   # z
                    ], dtype=np.int32)
                    base_coords = occupied_coord_label[..., :3]
                    offsets = NEIGHBOR_OFFSETS[np.newaxis, ...]
                    all_candidates = base_coords[:, np.newaxis, :] + offsets
                    all_candidates = all_candidates.reshape(-1, 3)
                    # filter out of range
                    valid_mask = (all_candidates >= 0) & (all_candidates < np.array(self.grid_size))
                    valid_mask = np.all(valid_mask, axis=1)
                    all_candidates = all_candidates[valid_mask]
                    all_candidates_label = voxel_label[all_candidates[:, 0], all_candidates[:, 1], all_candidates[:, 2]]
                    valid_mask = all_candidates_label == 0
                    valid_candidates = np.concatenate([
                        all_candidates[valid_mask], all_candidates_label[valid_mask][:,None]], axis=1)
                    idx = torch.randperm(valid_candidates.shape[0])
                    select_candidates = valid_candidates[idx][:self.max_points-occupied_coord_label.shape[0]]
                    # query
                    xyzl = np.concatenate([occupied_coord_label, select_candidates], axis=0)
                    # padding to max_point
                    if xyzl.shape[0] < self.max_points:
                        padding_xyz = torch.nonzero(torch.Tensor(voxel_label) == 0, as_tuple=False)
                        padding_xyzlabel = torch.nn.functional.pad(padding_xyz, (0,1),'constant', value=0)
                        idx = torch.randperm(padding_xyzlabel.shape[0])
                        padding_xyzlabel = padding_xyzlabel[idx][:self.max_points-xyzl.shape[0]].numpy()
                        xyzl = np.concatenate([xyzl, padding_xyzlabel], axis=0)

                xyz_label = torch.from_numpy(xyzl[:, 3])
                xyz_center = torch.from_numpy(xyzl[:, :3])
                xyz = xyz_center.float()

                query = torch.zeros(xyz.shape, dtype=torch.float32, device=xyz.device)
                query[:,0] = 2*xyz[:,0].clamp(0,self.grid_size[0]-1)/float(self.grid_size[0]-1) -1
                query[:,1] = 2*xyz[:,1].clamp(0,self.grid_size[1]-1)/float(self.grid_size[1]-1) -1
                query[:,2] = 2*xyz[:,2].clamp(0,self.grid_size[2]-1)/float(self.grid_size[2]-1) -1
            else:
                query, xyz_label, xyz_center = torch.zeros(1), torch.zeros(1), torch.zeros(1)
        return voxel_label, query, xyz_label, xyz_center, self.im_idx[index], invalid
        # voxel_label=[256,256,3]   query=N,3 [-1,1] query_coord     xyz_label=N,1   xyz_center=N,3 query_voxcoord
    
def get_query(voxel_label, num_class=18, grid_size = (200,200,16), max_points = 200000):
    xyzl = []
    # object
    for i in range(1, num_class):
        xyz = torch.nonzero(torch.Tensor(voxel_label) == i, as_tuple=False)
        xyzlabel = torch.nn.functional.pad(xyz, (1,0),'constant', value=i)
        xyzl.append(xyzlabel)
    # free space near object
    tdf = compute_tdf(voxel_label, trunc_distance=2)
    xyz = torch.nonzero(torch.tensor(np.logical_and(tdf > 0, tdf <= 2)), as_tuple=False)
    idx = torch.randperm(xyz.shape[0])
    xyzlabel = torch.nn.functional.pad(xyz[idx], (1,0),'constant', value=0)
    xyzl.append(xyzlabel)
    
    num_far_free = int(max_points - len(torch.cat(xyzl, dim=0)))
    if num_far_free <= 0 :
        xyzl = torch.cat(xyzl, dim=0)
        xyzl = xyzl[:max_points]
    else : 
        xyz = torch.nonzero(torch.tensor(np.logical_and(voxel_label == 0, tdf == -1)), as_tuple=False)
        xyzlabel = torch.nn.functional.pad(xyz, (1, 0), 'constant', value=0)
        idx = torch.randperm(xyzlabel.shape[0])
        xyzlabel = xyzlabel[idx][:min(xyzlabel.shape[0], num_far_free)]
        xyzl.append(xyzlabel)
        while len(torch.cat(xyzl, dim=0)) < max_points:
            for i in range(1, num_class):
                xyz = torch.nonzero(torch.Tensor(voxel_label) == i, as_tuple=False)
                xyzlabel = torch.nn.functional.pad(xyz, (1,0),'constant', value=i)
                xyzl.append(xyzlabel)
        xyzl = torch.cat(xyzl, dim=0)
        xyzl = xyzl[:max_points]
        
    xyz_label = xyzl[:, 0]
    xyz_center = xyzl[:, 1:]
    xyz = xyz_center.float()

    query = torch.zeros(xyz.shape, dtype=torch.float32, device=xyz.device)
    query[:,0] = 2*xyz[:,0].clamp(0,grid_size[0]-1)/float(grid_size[0]-1) -1
    query[:,1] = 2*xyz[:,1].clamp(0,grid_size[1]-1)/float(grid_size[1]-1) -1
    query[:,2] = 2*xyz[:,2].clamp(0,grid_size[2]-1)/float(grid_size[2]-1) -1
    
    return query, xyz_label, xyz_center
    # N,3 [-1,1] query_coord
    # N,1
    # N,3 [0,grid_size] query_voxcoord

def compute_tdf(voxel_label: np.ndarray, trunc_distance: float = 3, trunc_value: float = -1) -> np.ndarray:
    """ Compute Truncated Distance Field (TDF). voxel_label -- [X, Y, Z] """
    # make TDF at free voxels.
    # distance is defined as Euclidean distance to nearest unfree voxel (occupied or unknown).
    if voxel_label.shape[0] == 512:
        # downsample to save time
        voxel_label = voxel_label[::2, ::2, ::2]    
        free = voxel_label == 0
        tdf_downsampled = distance_transform_edt(free)
        # upsample to original size
        tdf = np.repeat(np.repeat(np.repeat(tdf_downsampled, 2, axis=0), 2, axis=1), 2, axis=2)
    else:
        free = voxel_label == 0
        tdf = distance_transform_edt(free)

    # Set -1 if distance is greater than truncation_distance
    tdf[tdf > trunc_distance] = trunc_value
    return tdf  # [X, Y, Z]

def flip(voxel, invalid, flip_dim=0):
    voxel = np.flip(voxel, axis=flip_dim).copy()
    invalid = np.flip(invalid, axis=flip_dim).copy()
    return voxel, invalid

# u1: uint8, u8: uint16, i8: int64
@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True, parallel=False)
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

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.dataset = 'Occ3D-nuScenes' # ['Occ3D-nuScenes', 'nuScenes-Occupancy']
    
    if args.dataset == 'Occ3D-nuScenes':
        args.data_path='data/nuscenes/gts'
        args.train_info_path='data/nuscenes/nuscenes_infos_train_temporal_v3_scene.pkl' # generated by OccWorld which use Occ3D dataset
        args.val_info_path='data/nuscenes/nuscenes_infos_val_temporal_v3_scene.pkl'
        args.num_class = 18
        args.grid_size = [200, 200, 16]
        args.pc_range = [-40, -40, -1, 40, 40, 5.4]

        dataset = NuScenesDatasetOcc(args, 'train')
        val_dataset = NuScenesDatasetOcc(args, 'val')

    elif args.dataset == 'nuScenes-Occupancy':
        args.data_path='data/nuscenes/nuScenes-Occupancy'
        args.train_info_path='data/nuscenes/nuscenes_infos_temporal_train_new.pkl'      # generated by Drive-OccWorld which use nuScenes-Occupancy dataset
        args.val_info_path='data/nuscenes/nuscenes_infos_temporal_train_new.pkl'
        args.num_class = 18
        args.grid_size = [512, 512, 40]
        args.pc_range = [-51.2, -51.2, -5, 51.2, 51.2, 3]

        dataset = NuScenesDatasetOcc(args, 'train', get_query=True)
        val_dataset = NuScenesDatasetOcc(args, 'val', get_query=True)

    voxel_num_per_class = np.zeros(args.num_class)
    for voxel_label, query, xyz_label, xyz_center, path, invalid in tqdm.tqdm(dataset):
        label_num = []
        for idx in range(args.num_class):
            voxel_num_per_class[idx] += (voxel_label == idx).sum()

    print(voxel_num_per_class)
