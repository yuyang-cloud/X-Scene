import logging
import numpy as np
import random
import copy
import mmcv
from mmcv.parallel import DataContainer as DC
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
from mmdet.datasets.pipelines import to_tensor
from nuscenes.eval.common.utils import Quaternion, quaternion_yaw

from .map_utils import LiDARInstanceLines, VectorizedLocalMap, visualize_bev_hdmap, project_map_to_image, project_box_to_image


@DATASETS.register_module()
class NuScenesTDataset(NuScenesDataset):
    def __init__(
        self,
        ann_file,
        pipeline=None,
        dataset_root=None,
        object_classes=None,
        map_classes=None,
        load_interval=1,
        with_velocity=True,
        modality=None,
        box_type_3d="LiDAR",
        filter_empty_gt=True,
        test_mode=False,
        eval_version="detection_cvpr_2019",
        use_valid_flag=False,
        force_all_boxes=False,
        video_length=None,
        ref_length=None,
        candidate_length=None,
        start_on_keyframe=True,
        fixed_ptsnum_per_line=-1,
        padding_value=-10000,
        temporal=True,
        use_2Hz=False
    ) -> None:
        self.video_length = video_length
        self.ref_length = ref_length
        self.candidate_length = candidate_length
        self.start_on_keyframe = start_on_keyframe
        self.temporal = temporal
        self.use_2Hz = use_2Hz
        super().__init__(
            ann_file, pipeline, dataset_root, object_classes, map_classes,
            load_interval, with_velocity, modality, box_type_3d,
            filter_empty_gt, test_mode, eval_version, use_valid_flag,
            force_all_boxes)
        self.padding_value = padding_value
        self.fixed_num = fixed_ptsnum_per_line
        self.object_classes = object_classes
        self.map_classes = map_classes

    def __len__(self):
        return len(self.clip_infos)

    def build_2Hz_clips(self, data_infos, scene_tokens):
        """Since the order in self.data_infos may change on loading, we
        calculate the index for clips after loading.

        Args:
            data_infos (list of dict): loaded data_infos
            scene_tokens (2-dim list of str): 2-dim list for tokens to each
            scene 

        Returns:
            2-dim list of int: int is the index in self.data_infos
        """
        self.token_data_dict = {
            item['token']: idx for idx, item in enumerate(data_infos)}
        all_clips = []
        assert self.temporal
        self.scene_key_frame = {}
        num_clip = 0
        for si, scene in enumerate(scene_tokens):
            scene_2Hz = []
            for x in scene:
                if ";" not in x and len(x)<33:
                    scene_2Hz.append(x)
            scene = scene_2Hz

            for start in range(0, len(scene), self.video_length):
                clip = [self.token_data_dict[token]
                        for token in scene[start: min(start + self.video_length, len(scene))]]
                if len(clip) < self.video_length:
                    clip += [clip[-1]]*(self.video_length-len(clip))

                ref_idx = [0, 0] if start == 0 else sorted(random.sample(range(max(start-self.candidate_length, 0), start), self.ref_length))
                ref = [self.token_data_dict[scene[idx]] for idx in ref_idx]
                clip = ref + clip

                if f'{si}' not in self.scene_key_frame:
                    self.scene_key_frame[f'{si}'] = []
                self.scene_key_frame[f'{si}'].append(num_clip)

                num_clip += 1

                all_clips.append(clip)
        logging.info(f"[{self.__class__.__name__}] Got {len(scene_tokens)} "
                     f"continuous scenes. Cut into {self.video_length + self.ref_length}-clip ({self.ref_length}/{self.video_length}), "
                     f"which has {len(all_clips)} in total.")
        return all_clips

    def build_clips(self, data_infos, scene_tokens):
        """Since the order in self.data_infos may change on loading, we
        calculate the index for clips after loading.

        Args:
            data_infos (list of dict): loaded data_infos
            scene_tokens (2-dim list of str): 2-dim list for tokens to each
            scene 

        Returns:
            2-dim list of int: int is the index in self.data_infos
        """
        self.token_data_dict = {
            item['token']: idx for idx, item in enumerate(data_infos)}
        all_clips = []
        if not self.temporal:
            self.ref_length = 0
            self.video_length = 1
        self.scene_key_frame = {}
        num_clip = 0
        for si, scene in enumerate(scene_tokens):
            for start in range(0, len(scene) - self.video_length + 1):
                if self.start_on_keyframe and ";" in scene[start]:
                    continue  # this is not a keyframe
                if self.start_on_keyframe and len(scene[start]) >= 33:
                    continue  # this is not a keyframe
                clip = [self.token_data_dict[token]
                        for token in scene[start: start + self.video_length]]

                if self.temporal:
                    ref_idx = [0, 0] if start == 0 else sorted(random.sample(range(max(start-self.candidate_length, 0), start), self.ref_length))
                    ref = [self.token_data_dict[scene[idx]] for idx in ref_idx]
                    clip = ref + clip

                if not (";" in scene[start + self.video_length-1] or len(scene[start + self.video_length-1]) >= 33):
                    if f'{si}' not in self.scene_key_frame:
                        self.scene_key_frame[f'{si}'] = []
                    self.scene_key_frame[f'{si}'].append(num_clip)

                num_clip += 1

                all_clips.append(clip)
        logging.info(f"[{self.__class__.__name__}] Got {len(scene_tokens)} "
                     f"continuous scenes. Cut into {self.video_length + self.ref_length}-clip ({self.ref_length}/{self.video_length}), "
                     f"which has {len(all_clips)} in total.")
        return all_clips

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file)
        data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
        data_infos = data_infos[:: self.load_interval]
        self.metadata = data["metadata"]
        self.version = self.metadata["version"]
        self.clip_infos = self.build_clips(data_infos, data['scene_tokens']) if not self.use_2Hz else self.build_2Hz_clips(data_infos, data['scene_tokens'])
        return data_infos

    def get_can_bus(self, frame):
        ego2global = frame['ego2global']
        translation = ego2global[:3, 3]
        rotation = Quaternion(matrix=ego2global[:3, :3].astype(np.float64))
        can_bus = np.zeros(9)
        can_bus[:3] = translation
        can_bus[3:7] = rotation
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle

        return can_bus
    
    def get_data_info_single(self, index: int):
        info = self.data_infos[index]

        data = dict(
            token=info["token"],
            sample_idx=info['token'],
            scene_name=info["scene_name"],
            scene_token=info["scene_token"],
            lidar_token=info['lidar_token'],
            lidar_path=info["lidar_path"],
            sweeps=info["sweeps"],
            timestamp=info["timestamp"],
            location=info["location"],
            nearest_keyframe_token=info["nearest_keyframe_token"],
        )
        add_key = [
            "description",
            "timeofday",
            "visibility",
            "flip_gt",
        ]
        for key in add_key:
            if key in info:
                data[key] = info[key]

        # ego to global transform
        ego2global = np.eye(4).astype(np.float64)
        ego2global[:3, :3] = Quaternion(
            info["ego2global_rotation"]).rotation_matrix
        ego2global[:3, 3] = info["ego2global_translation"]
        data["ego2global"] = ego2global

        # lidar to ego transform
        lidar2ego = np.eye(4).astype(np.float64)
        lidar2ego[:3, :3] = Quaternion(
            info["lidar2ego_rotation"]).rotation_matrix
        lidar2ego[:3, 3] = info["lidar2ego_translation"]
        data["lidar2ego"] = lidar2ego

        if self.modality["use_camera"]:
            data["image_paths"] = []
            data["lidar2camera"] = []
            data["lidar2image"] = []
            data["camera2ego"] = []
            data["camera_intrinsics"] = []
            data["camera2lidar"] = []

            for _, camera_info in info["cams"].items():
                data["image_paths"].append(camera_info["data_path"])

                # lidar to camera transform
                lidar2camera_r = np.linalg.inv(
                    camera_info["sensor2lidar_rotation"])
                lidar2camera_t = (
                    camera_info["sensor2lidar_translation"] @ lidar2camera_r.T
                )
                lidar2camera_rt = np.eye(4).astype(np.float32)
                lidar2camera_rt[:3, :3] = lidar2camera_r.T
                lidar2camera_rt[3, :3] = -lidar2camera_t
                data["lidar2camera"].append(lidar2camera_rt.T)

                # camera intrinsics
                camera_intrinsics = np.eye(4).astype(np.float32)
                camera_intrinsics[:3, :3] = camera_info["camera_intrinsics"]
                data["camera_intrinsics"].append(camera_intrinsics)

                # lidar to image transform
                lidar2image = camera_intrinsics @ lidar2camera_rt.T
                data["lidar2image"].append(lidar2image)

                # camera to ego transform
                camera2ego = np.eye(4).astype(np.float32)
                camera2ego[:3, :3] = Quaternion(
                    camera_info["sensor2ego_rotation"]
                ).rotation_matrix
                camera2ego[:3, 3] = camera_info["sensor2ego_translation"]
                data["camera2ego"].append(camera2ego)

                # camera to lidar transform
                camera2lidar = np.eye(4).astype(np.float32)
                camera2lidar[:3, :3] = camera_info["sensor2lidar_rotation"]
                camera2lidar[:3, 3] = camera_info["sensor2lidar_translation"]
                data["camera2lidar"].append(camera2lidar)

        annos, mask = self.get_ann_info(index)
        if "visibility" in data:
            data["visibility"] = data["visibility"][mask]
        data["ann_info"] = annos
        return data
    
    def get_data_info(self, index):
        """We should sample from clip_infos
        """
        clip = self.clip_infos[index]
        frames = []
        for frame in clip:
            frame_info = self.get_data_info_single(frame)
            frames.append(frame_info)

        return frames

    def prepare_train_data(self, index):
        """This is called by `__getitem__`
            index is the index of the clip_infos
        """
        frames = self.get_data_info(index)
        if None in frames:
            return None
        examples = []
        for i, frame in enumerate(frames):
            self.pre_pipeline(frame)
            example = self.pipeline(frame)

            # process can bus information
            can_bus = self.get_can_bus(frame)
            if i == 0:
                prev_pos = copy.deepcopy(can_bus[:3])
                prev_angle = copy.deepcopy(can_bus[-1])
                can_bus[:3] = 0
                can_bus[-1] = 0
            else:
                tmp_pos = copy.deepcopy(can_bus[:3])
                tmp_angle = copy.deepcopy(can_bus[-1])
                can_bus[:3] -= prev_pos
                can_bus[-1] -= prev_angle
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)
            example['can_bus'] = DC(to_tensor(can_bus), cpu_only=False)

            if self.filter_empty_gt and frame['is_key_frame'] and (
                example is None or ~(example["gt_labels_3d"]._data != -1).any()
            ):
                return None
            examples.append(example)

        if not self.temporal:
            return examples[0]
 
        return examples

