import numpy as np
from copy import deepcopy
import torch
from pathlib import Path
from typing import Tuple, List
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.utils.colmap_parsing_utils import read_points3D_binary

class DatasetMerger(InputDataset):
    def __init__(self,dataset_groups:list, *args, **kwargs):
        #super().__init__(*args,**kwargs)
        self.group=dataset_groups
        self.group_id=[]
        self.group_len=[]
        for i in range(len(self.group)):
            self.group_id.extend([i]*len(self.group[i]))
            self.group_len.append(len(self.group[i]))
        self.scene_box=self.group[0].scene_box#should be extend to all group
        cameras=None
        for dataset in dataset_groups:
            cam=dataset.cameras

            if cameras is None:
                cameras=cam
            else:
                cameras = Cameras(
                    camera_to_worlds=torch.cat([cameras.camera_to_worlds,cam.camera_to_worlds],dim=0),
                    fx=torch.cat([cameras.fx,cam.fx],dim=0),
                    fy=torch.cat([cameras.fy,cam.fy],dim=0),
                    cx=torch.cat([cameras.cx,cam.cx],dim=0),
                    cy=torch.cat([cameras.cy,cam.cy],dim=0),
                    height=torch.cat([cameras.height,cam.height],dim=0),
                    width=torch.cat([cameras.width,cam.width],dim=0),
                    camera_type=CameraType.PERSPECTIVE,
                    distortion_params=torch.cat([cameras.distortion_params,cam.distortion_params],dim=0),
                )
        self.cameras=deepcopy(cameras)

        self.metadata=dataset_groups[0].metadata #todo fix it

    def __len__(self):
        return len(self.group_id)
    def id2group_id(self,image_idx:int):
        group_id=self.group_id[image_idx]
        offset=sum(self.group_len[:group_id])
        image_idx-=offset
        return group_id,image_idx, offset
    def get_data(self,image_idx:int):
        group_id, image_idx, offset=self.id2group_id(image_idx)
        data=self.group[group_id].get_data(image_idx)
        data['image_idx']+=offset
        return data
    def get_image(self,image_idx:int):
        group_id, image_idx=self.id2group_id(image_idx)
        return self.group[group_id].get_image(image_idx)
        