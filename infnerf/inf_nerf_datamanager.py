"""
Infnerf datamanager.
"""
import os
import numpy as np
import json
from dataclasses import dataclass, field
from typing import Type
import math

from typing import Any, Dict, List, Optional, Tuple, Type, Union
from nerfstudio.data.datamanagers import base_datamanager
# from nerfstudio.data.datamanagers import variable_res_datamanager# import VariableResDataManagerConfig, VariableResDataManager
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from infnerf.datasetmerger import DatasetMerger

from nerfstudio.data.datasets.base_dataset import InputDataset

from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, List

from nerfstudio.data.utils.colmap_parsing_utils import (
    qvec2rotmat,
    read_next_bytes,
    read_cameras_binary,
    read_images_binary,
    read_points3D_binary,
)
from nerfstudio.process_data.colmap_utils import (
    parse_colmap_camera_params
)

@dataclass
class MixMultiResDataManagerConfig(base_datamanager.VanillaDataManagerConfig):
    
    _target: Type = field(default_factory=lambda: MixMultiResDataManager)
    train_scale: List[float] = field(
        default_factory=lambda:[
            0.25,
            0.5,
            1.0
        ]    
    ),
    eval_scale: List[float] = field(
        default_factory=lambda:[
            0.25,
            0.5,
            1.0
        ]
    )
       
class MixMultiResDataManager(base_datamanager.VanillaDataManager):  # pylint: disable=abstract-method
    """Data manager implementation for data that also requires processing depth data.
    Args:
        config: the DataManagerConfig used to instantiate class
    """
    config:MixMultiResDataManagerConfig
    train_dataset:DatasetMerger
    eval_dataset:DatasetMerger
    def __init__(
        self,
        config: MixMultiResDataManagerConfig,
        *args, **kwargs        
    ):
        self.output_new_trans(config=config) # output new transforms.json
        #config.camera_res_scale_factor=config.scale_factor_and_step[0][0]
        super().__init__(config,*args, **kwargs)
    def create_train_dataset(self) -> DatasetMerger:
        datasets=[]

        for scale in self.config.train_scale:
            scale_factor=scale
            datasets.append(InputDataset(
                    dataparser_outputs=self.train_dataparser_outputs,
                    scale_factor=scale_factor,
                ))
        self.config.collate_fn = base_datamanager.variable_res_collate
        """Sets up the data loaders for training"""
        return DatasetMerger(datasets)

    def create_eval_dataset(self) -> DatasetMerger:
        datasets=[]
        parser_output=self.dataparser.get_dataparser_outputs(split=self.test_split)
        for scale in self.config.eval_scale:
            scale_factor=scale
            datasets.append(InputDataset(
                    dataparser_outputs=parser_output,
                    scale_factor=scale_factor,
                ))
        self.config.collate_fn = base_datamanager.variable_res_collate
        return DatasetMerger(datasets)

    def output_new_trans(self, config):
        return
        data_path = config.data
        colmap_path = os.path.join(data_path, config.colmap_path)

        self.Rt = self.get_Rt_from_pt(colmap_path)
        self.trans_xyz_list = self.get_trans_pt()

        # get new translation based on pts in range 5% to 95%
        self.new_t, self.scale = self.get_new_trans_scale()

        # get new trans_xyz_list after new_t translation
        self.trans_xyz_list = self.get_new_trans_xyz_list()

        # get scale xyz_list
        norm_xyz_list = self.get_norm_xyz_list()
        self.norm_xyz_list = norm_xyz_list

        # output scaled transforms.json
        recon_path = Path(colmap_path)
        output_path = Path(data_path)
        self.colmap_to_json(recon_path, output_path)

        # calculate min scale for each xyz
        self.pt_scale_list = self.get_pt_scale()
    
    def get_Rt_from_pt(self, colmap_path):
        points3D = read_points3D_binary(os.path.join(colmap_path, 'points3D.bin'))
        xyz_list = []
        xyz_img_ids_list = []

        for im_id, pt_data in points3D.items():
            # remove points where obs < 3 (outlier)
            if len(pt_data.image_ids) < 3:
                continue
            xyz_list.append(pt_data.xyz)
            xyz_img_ids_list.append(pt_data.image_ids)

        xyz_list = np.array(xyz_list)
        self.xyz_list = xyz_list
        self.xyz_img_ids_list = xyz_img_ids_list

        origin = np.mean(xyz_list, axis=0)
        norm_vec, tan_vec1, tan_vec2 = self.get_ortho_basis(xyz_list)

        z = np.array(norm_vec)
        x = np.array(tan_vec1)
        y = np.array(tan_vec2)

        Xw = np.column_stack((x, y, z))
        R = np.linalg.inv(Xw)

        t = np.array(origin) * -1
        t01 = np.dot(R, t)
        R01 = R

        Rt = np.eye(4)
        Rt[:3, :3] = R01
        Rt[:3, 3] = t01

        return Rt

    def get_ortho_basis(self, xyz_list: List[List[float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Center the point cloud by subtracting its centroid
        centroid = np.mean(xyz_list, axis=0)
        centered_point_cloud = xyz_list - centroid

        # Compute the covariance matrix
        covariance_matrix = np.cov(centered_point_cloud, rowvar=False)

        # Perform eigenvalue decomposition to find the principal components
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # The eigenvector corresponding to the smallest eigenvalue is the normal vector of the fitted plane
        norm_vec = eigenvectors[:, 0]
        norm_vec /= np.linalg.norm(norm_vec)

        arbitrary_vector = np.array([1.0, 0.0, 0.0])

        tan_vec1 = np.cross(norm_vec, arbitrary_vector)
        tan_vec1 /= np.linalg.norm(tan_vec1)

        tan_vec2 = np.cross(norm_vec, tan_vec1)
        tan_vec2 /= np.linalg.norm(tan_vec2)

        return norm_vec, tan_vec1, tan_vec2

    def get_trans_pt(self):
        xyz_list = self.xyz_list

        Rt = self.Rt
        pt_homo = np.append(xyz_list, np.ones((xyz_list.shape[0], 1)), axis=1)
        trans_pt_homo = np.dot(Rt, pt_homo.T).T

        return trans_pt_homo[:, :3]

    def sort_with_axis(self, arr, axis):
        sorted_idx = np.argsort(arr[:, axis])
        sorted_arr = arr[sorted_idx]
        return sorted_arr

    def get_new_trans_scale(self):
        trans_xyz_list = self.trans_xyz_list
        idx = int(0.05 * self.trans_xyz_list.shape[0])
        
        x_sorted_xyz_list = self.sort_with_axis(trans_xyz_list, 0)
        y_sorted_xyz_list = self.sort_with_axis(trans_xyz_list, 1)
        z_sorted_xyz_list = self.sort_with_axis(trans_xyz_list, 2)

        x_min = x_sorted_xyz_list[idx][0]
        x_max = x_sorted_xyz_list[-idx][0]
        y_min = y_sorted_xyz_list[idx][1]
        y_max = y_sorted_xyz_list[-idx][1]
        z_min = z_sorted_xyz_list[idx][2]
        z_max = z_sorted_xyz_list[-idx][2]

        # pre_t = np.array(((x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2))

        l = x_max - x_min
        w = y_max - y_min
        h = z_max - z_min
        new_t = np.array((x_min, y_min, z_min)) + 0.5 * max(l, w, h)

        # x_scale = 2 / (x_max - x_min)
        # y_scale = 2 / (y_max - y_min)
        # z_scale = 2 / (z_max - z_min)

        # pre_min_scale = min(x_scale, y_scale, z_scale)
        min_scale = 2 / max(l, w, h)

        # import pdb; pdb.set_trace()

        return new_t, min_scale
    
    def get_new_trans_xyz_list(self):
        return self.trans_xyz_list - self.new_t

    def get_norm_xyz_list(self):
        norm_xyz_list = self.scale * self.trans_xyz_list
        return norm_xyz_list
    
    def colmap_to_json(
        self,
        recon_dir: Path,
        output_dir: Path,
        camera_mask_path: Optional[Path] = None,
        image_id_to_depth_path: Optional[Dict[int, Path]] = None,
        image_rename_map: Optional[Dict[str, str]] = None,
    ) -> int:
        """Converts COLMAP's cameras.bin and images.bin to a JSON file.

        Args:
            recon_dir: Path to the reconstruction directory, e.g. "sparse/0"
            output_dir: Path to the output directory.
            camera_model: Camera model used.
            camera_mask_path: Path to the camera mask.
            image_id_to_depth_path: When including sfm-based depth, embed these depth file paths in the exported json
            image_rename_map: Use these image names instead of the names embedded in the COLMAP db

        Returns:
            The number of registered images.
        """

        cam_id_to_camera = read_cameras_binary(recon_dir / "cameras.bin")
        im_id_to_image = read_images_binary(recon_dir / "images.bin")

        frames = []
        img_ids_w2c = {}
        for im_id, im_data in im_id_to_image.items():
            rotation = qvec2rotmat(im_data.qvec)
            translation = im_data.tvec.reshape(3, 1)

            w2c = np.concatenate([rotation, translation], 1)
            w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)

            # transform w2c to new world coord
            Rt = self.Rt
            w2c = np.dot(w2c, np.linalg.inv(Rt))

            # further apply new_t transformation
            # w2c[:, 3] = w2c[:, 3] - np.append(self.new_t, 0)
            new_Rt = np.eye(4)
            new_Rt[:3, 3][:3] = self.new_t

            w2c = np.dot(w2c, np.linalg.inv(new_Rt))

            # scale camera translation
            w2c[:, 3][:3] = self.scale * w2c[:, 3][:3]

            img_ids_w2c[im_id] = w2c

            c2w = np.linalg.inv(w2c)

            # Convert from COLMAP's camera coordinate system (OpenCV) to ours (OpenGL)
            c2w[0:3, 1:3] *= -1
            c2w = c2w[np.array([1, 0, 2, 3]), :]
            c2w[2, :] *= -1

            name = im_data.name
            if image_rename_map is not None:
                name = image_rename_map[name]
            name = Path(f"./images/{name}")

            frame = {
                "file_path": name.as_posix(),
                "transform_matrix": c2w.tolist(),
                "colmap_im_id": im_id,
            }
            if camera_mask_path is not None:
                frame["mask_path"] = camera_mask_path.relative_to(camera_mask_path.parent.parent).as_posix()
            if image_id_to_depth_path is not None:
                depth_path = image_id_to_depth_path[im_id]
                frame["depth_file_path"] = str(depth_path.relative_to(depth_path.parent.parent))
            frames.append(frame)

        if set(cam_id_to_camera.keys()) != {1}:
            raise RuntimeError("Only single camera shared for all images is supported.")
        out = parse_colmap_camera_params(cam_id_to_camera[1])
        self.f = math.sqrt(out["fl_x"]**2 + out["fl_y"]**2)

        out["frames"] = frames

        applied_transform = np.eye(4)[:3, :]
        applied_transform = applied_transform[np.array([1, 0, 2]), :]
        applied_transform[2, :] *= -1
        out["applied_transform"] = applied_transform.tolist()

        with open(output_dir / "transforms.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=4)
        
        self.img_ids_w2c = img_ids_w2c
    
    def get_pt_scale(self):
        xyz_img_ids_list = self.xyz_img_ids_list
        norm_xyz_list = self.norm_xyz_list
        pt_scale_list = []

        for idx, xyz in enumerate(norm_xyz_list):
            image_ids = xyz_img_ids_list[idx]
            pt_scale = []

            for image_id in image_ids:
                w2c = self.img_ids_w2c[image_id]
                xyz_homo = np.append(xyz, 1)

                xyz_c = np.dot(w2c, xyz_homo)
                z = xyz_c[2]

                cur_scale = z / self.f
                pt_scale.append(cur_scale)
            
            pt_scale_list.append(min(pt_scale))

        return pt_scale_list
