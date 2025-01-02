# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Data parser for nerfstudio datasets. """

from __future__ import annotations

import math
import sys
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Type, Tuple

import numpy as np
import torch
from PIL import Image
from rich.prompt import Confirm

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils import colmap_parsing_utils as colmap_utils
from nerfstudio.process_data.colmap_utils import parse_colmap_camera_params
from nerfstudio.process_data.process_data_utils import downscale_images
from nerfstudio.utils.rich_utils import CONSOLE
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
MAX_AUTO_RESOLUTION = 1600


@dataclass
class InfNerfDataParserConfig(DataParserConfig):
    """Nerfstudio dataset config"""

    _target: Type = field(default_factory=lambda: InfNerfDataParser)
    """target class to instantiate"""
    data: Path = Path()
    """Directory or explicit json file path specifying location of data."""
    
    downscale_factor: Optional[int] = None
    """How much to downscale images. If not set, images are chosen such that the max dimension is <1600px."""
    
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    train_split_fraction: float = 0.9
    """The fraction of images to use for training. The remaining images are for eval."""
    depth_unit_scale_factor: float = 1e-3
    """Scales the depth values to meters. Default value is 0.001 for a millimeter to meter conversion."""

    images_path: Path = Path("images")
    """Path to images directory relative to the data path."""
    masks_path: Optional[Path] = None#Path("mask") # Path("mask")
    eval_masks_path: Optional[Path] = None#Path("eval_mask") # Path("eval_mask")
    """Path to masks directory. If not set, masks are not loaded."""
    depths_path: Optional[Path] = None
    """Path to depth maps directory. If not set, depths are not loaded."""
    colmap_path: Path = Path("sparse/0")
    """Path to the colmap reconstruction directory relative to the data path."""
    eval_mode: Literal["fraction", "filename", "interval", "all"] = "all"
    '''Use filename to split train, val and test'''

    largest_downscale: Optional[int] = None

class InfNerfDataParser(DataParser):
    """COLMAP DatasetParser.
    Expects a folder with the following structure:
        images/ # folder containing images used to create the COLMAP model
        sparse/0 # folder containing the COLMAP reconstruction (either TEXT or BINARY format)
        masks/ # (OPTIONAL) folder containing masks for each image
        depths/ # (OPTIONAL) folder containing depth maps for each image
    The paths can be different and can be specified in the config. (e.g., sparse/0 -> sparse)
    Currently, most COLMAP camera models are supported except for the FULL_OPENCV and THIN_PRISM_FISHEYE models.

    The dataparser loads the downscaled images from folders with `_{downscale_factor}` suffix.
    If these folders do not exist, the user can choose to automatically downscale the images and
    create these folders.

    The loader is compatible with the datasets processed using the ns-process-data script and
    can be used as a drop-in replacement. It further supports datasets like Mip-NeRF 360 (although
    in the case of Mip-NeRF 360 the downsampled images may have a different resolution because they
    use different rounding when computing the image resolution).
    """

    config: InfNerfDataParserConfig

    def __init__(self, config: InfNerfDataParserConfig):
        super().__init__(config)
        self.config = config
        self._downscale_factor = None
        self._cache={}

    def _get_all_images_and_cameras(self, recon_dir: Path):
        if (recon_dir / "cameras.txt").exists():
            cam_id_to_camera = colmap_utils.read_cameras_text(recon_dir / "cameras.txt")
            im_id_to_image = colmap_utils.read_images_text(recon_dir / "images.txt")
        elif (recon_dir / "cameras.bin").exists():
            cam_id_to_camera = colmap_utils.read_cameras_binary(recon_dir / "cameras.bin")
            im_id_to_image = colmap_utils.read_images_binary(recon_dir / "images.bin")
        else:
            raise ValueError(f"Could not find cameras.txt or cameras.bin in {recon_dir}")

        cameras = {}
        frames = []
        camera_model = None

        # Parse cameras
        for cam_id, cam_data in cam_id_to_camera.items():
            cameras[cam_id] = parse_colmap_camera_params(cam_data)

        # Parse frames
        for im_id, im_data in im_id_to_image.items():
            # NB: COLMAP uses Eigen / scalar-first quaternions
            # * https://colmap.github.io/format.html
            # * https://github.com/colmap/colmap/blob/bf3e19140f491c3042bfd85b7192ef7d249808ec/src/base/pose.cc#L75
            # the `rotation_matrix()` handles that format for us.
            rotation = colmap_utils.qvec2rotmat(im_data.qvec)
            translation = im_data.tvec.reshape(3, 1)
            w2c = np.concatenate([rotation, translation], 1)
            w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
            c2w = np.linalg.inv(w2c)
            # Convert from COLMAP's camera coordinate system (OpenCV) to ours (OpenGL)
            c2w[0:3, 1:3] *= -1
            c2w = c2w[np.array([1, 0, 2, 3]), :]
            c2w[2, :] *= -1

            frame = {
                "file_path": (self.config.images_path / im_data.name).as_posix(),
                "transform_matrix": c2w,
                "colmap_im_id": im_id,
            }
            frame.update(cameras[im_data.camera_id])
            if self.config.masks_path is not None:
                frame["mask_path"] = ((self.config.masks_path / im_data.name).with_suffix(".png").as_posix(),)
            if self.config.depths_path is not None:
                frame["depth_path"] = ((self.config.depths_path / im_data.name).with_suffix(".png").as_posix(),)
            frames.append(frame)
            if camera_model is not None:
                assert camera_model == frame["camera_model"], "Multiple camera models are not supported"
            else:
                camera_model = frame["camera_model"]

        out = {}
        out["frames"] = frames
        applied_transform = np.eye(4)[:3, :]
        applied_transform = applied_transform[np.array([1, 0, 2]), :]
        applied_transform[2, :] *= -1
        out["applied_transform"] = applied_transform.tolist()
        out["camera_model"] = camera_model
        assert len(frames) > 0, "No images found in the colmap model"
        return out

    def _get_image_indices(self, image_filenames, split):
        has_split_files_spec = (
            (self.config.data / "train_list.txt").exists()
            or (self.config.data / "test_list.txt").exists()
            or (self.config.data / "validation_list.txt").exists()
        )
        if (self.config.data / f"{split}_list.txt").exists():
            CONSOLE.log(f"Using {split}_list.txt to get indices for split {split}.")
            with (self.config.data / f"{split}_list.txt").open("r", encoding="utf8") as f:
                filenames = f.read().splitlines()
            #if split!="train":
            #    filenames=["C/DJI_0362.JPG"]
            # Validate split first
            split_filenames = set(self.config.images_path / x for x in filenames)
            unmatched_filenames = split_filenames.difference(image_filenames)
            if unmatched_filenames:
                raise RuntimeError(
                    f"Some filenames for split {split} were not found: {set(map(str, unmatched_filenames))}."
                )

            indices = [i for i, path in enumerate(image_filenames) if path in split_filenames]
            CONSOLE.log(f"[yellow] Dataset is overriding {split}_indices to {indices}")
            indices = np.array(indices, dtype=np.int32)
        elif has_split_files_spec:
            raise RuntimeError(f"The dataset's list of filenames for split {split} is missing.")
        elif self.config.eval_mode == "filename":
            i_train, i_eval = self.get_train_eval_split_filename(image_filenames)

            if split == "train":
                indices = i_train
            elif split in ["val", "test"]:
                indices = i_eval
        else:
            # filter image_filenames and poses based on train/eval split percentage
            num_images = len(image_filenames)
            num_train_images = math.ceil(num_images * self.config.train_split_fraction)
            num_eval_images = num_images - num_train_images
            i_all = np.arange(num_images)
            i_train = np.linspace(
                0, num_images - 1, num_train_images, dtype=int
            )  # equally spaced training images starting and ending at 0 and num_images-1
            i_eval = np.setdiff1d(i_all, i_train)  # eval images are the remaining images
            assert len(i_eval) == num_eval_images
            if split == "train":
                indices = i_train
            elif split in ["val", "test"]:
                indices = i_eval
            else:
                raise ValueError(f"Unknown dataparser split {split}")
        return indices
    def get_train_eval_split_filename(self, image_filenames: List) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the train/eval split based on the filename of the images.

        Args:
            image_filenames: list of image filenames
        """

        num_images = len(image_filenames)
        basenames = [os.path.basename(image_filename) for image_filename in image_filenames]
        i_all = np.arange(num_images)
        i_train = []
        i_eval = []
        for idx, basename in zip(i_all, basenames):
            # check the frame index
            if "train" in basename:
                i_train.append(idx)
            elif "eval" in basename:
                i_eval.append(idx)
                i_train.append(idx) # train left half of eval image
            else:
                raise ValueError("frame should contain train/eval in its name to use this eval-frame-index eval mode")

        return np.array(i_train), np.array(i_eval)

    def _generate_dataparser_outputs(self, split: str = "train"):
        #import pdb;pdb.set_trace();
        assert self.config.data.exists(), f"Data directory {self.config.data} does not exist."
        colmap_path = self.config.data / self.config.colmap_path
        assert colmap_path.exists(), f"Colmap path {colmap_path} does not exist."

        meta = self._get_all_images_and_cameras(colmap_path)
        camera_type = CAMERA_MODEL_TO_TYPE[meta["camera_model"]]

        image_filenames = []
        mask_filenames = []
        depth_filenames = []
        poses = []
        poses_dict = {}

        fx = []
        fy = []
        cx = []
        cy = []
        height = []
        width = []
        distort = []

        for frame in meta["frames"]:
            fx.append(float(frame["fl_x"]))
            fy.append(float(frame["fl_y"]))
            cx.append(float(frame["cx"]))
            cy.append(float(frame["cy"]))
            height.append(int(frame["h"]))
            width.append(int(frame["w"]))
            distort.append(
                camera_utils.get_distortion_params(
                    k1=float(frame["k1"]) if "k1" in frame else 0.0,
                    k2=float(frame["k2"]) if "k2" in frame else 0.0,
                    k3=float(frame["k3"]) if "k3" in frame else 0.0,
                    k4=float(frame["k4"]) if "k4" in frame else 0.0,
                    p1=float(frame["p1"]) if "p1" in frame else 0.0,
                    p2=float(frame["p2"]) if "p2" in frame else 0.0,
                )
            )

            image_filenames.append(Path(frame["file_path"]))
            poses.append(frame["transform_matrix"])
            poses_dict[frame["colmap_im_id"]] = frame["transform_matrix"]
            if "mask_path" in frame:
                if split == 'train':
                    mask_filenames.append(Path(frame["mask_path"][0]))
                else:
                    mask_filenames.append(Path(frame["mask_path"][0].replace(str(self.config.masks_path), str(self.config.eval_masks_path))))

            if "depth_path" in frame:
                depth_filenames.append(Path(frame["depth_path"]))

        assert len(mask_filenames) == 0 or (
            len(mask_filenames) == len(image_filenames)
        ), """
        Different number of image and mask filenames.
        You should check that mask_path is specified for every frame (or zero frames) in transforms.json.
        """
        assert len(depth_filenames) == 0 or (
            len(depth_filenames) == len(image_filenames)
        ), """
        Different number of image and depth filenames.
        You should check that depth_file_path is specified for every frame (or zero frames) in transforms.json.
        """
        poses=np.array(poses)
        max_img_id=max(poses_dict.keys())
        poses_ordered_by_id=np.array([poses_dict[i] if i in poses_dict else np.identity(4) for i in range(max_img_id+1)])#colmap img id start from 1
        assert(poses_ordered_by_id.shape==(max_img_id+1,4,4))
        #poses_s = np.array([poses_dict[key] for key in sorted(poses_dict)])
        if "pt" not in self._cache or "pt_s" not in self._cache:
            points3D = read_points3D_binary(colmap_path/'points3D.bin')
            pt_s=self.get_pt_scale(points3D, poses_ordered_by_id, fx[0], meta)
            obs_mask=self.filter_pt(points3D)
            pt=np.array([pt.xyz for _, pt in points3D.items()])
            pt = pt[obs_mask]
            pt_s=pt_s[obs_mask]
            self._cache["pt"]=pt #cache sparse point up, so eval no need to do it again
            self._cache["pt_s"]=pt_s
        else:
            pt=self._cache["pt"]
            pt_s=self._cache["pt_s"]

        pt = np.append(pt,np.ones((pt.shape[0],1)),axis=1)
        applied_transform=np.array(meta['applied_transform'])#todo applied scaling
        pt=(applied_transform@pt.T).T
        #self._write_pts("pt.xyz",pt)
        #self._write_pts("c.xyz",poses[:,:3,3])
        
        #find rota translation scale from pt
        r = self.orient_pca(pt)
        pt = np.append(pt,np.ones((pt.shape[0],1)),axis=1)#hormalized
        pt_rota=(r@pt.T).T
        t,s= self.get_ts(self.get_th_aabb(pt_rota,0.01)) # 0.02
        pt=((s@t@r)@pt.T).T #r, t, s applied on pt in order
        bbox=self.get_th_aabb(pt,0.0001)
        #self._write_pts("pt_rts.xyz",pt[:,:3])
        scale_factor=s[0,0]
        pt_s=scale_factor*pt_s
        transform_matrix=torch.from_numpy(t@r)

        poses = (t@r)@poses
        poses[:,:,3]=(s@poses[:,:,3].T).T#apply scale on c
        #self._write_pts("c_rts.xyz", poses[:,:3,3])

        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        #import pdb;pdb.set_trace()
        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        total_num_photo=len(image_filenames)
        indices = self._get_image_indices(image_filenames, split)
        image_filenames, mask_filenames, depth_filenames, downscale_factor = self._setup_downscale_factor(
            image_filenames, mask_filenames, depth_filenames
        )

        image_filenames = [image_filenames[i] for i in indices]
        mask_filenames = [mask_filenames[i] for i in indices] if len(mask_filenames) > 0 else []
        depth_filenames = [depth_filenames[i] for i in indices] if len(depth_filenames) > 0 else []

        idx_tensor = torch.tensor(indices, dtype=torch.long)
        poses = poses[idx_tensor]

        # in x,y,z order
        
        scene_box = SceneBox(torch.from_numpy(bbox))
        fg_box=torch.tensor([[-1, -1, -1], [1, 1, 1]], dtype=torch.float32)

        fx = torch.tensor(fx, dtype=torch.float32)[idx_tensor]
        fy = torch.tensor(fy, dtype=torch.float32)[idx_tensor]
        cx = torch.tensor(cx, dtype=torch.float32)[idx_tensor]
        cy = torch.tensor(cy, dtype=torch.float32)[idx_tensor]
        height = torch.tensor(height, dtype=torch.int32)[idx_tensor]
        width = torch.tensor(width, dtype=torch.int32)[idx_tensor]
        distortion_params = torch.stack(distort, dim=0)[idx_tensor]

        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_params=distortion_params,
            height=height,
            width=width,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=camera_type,
        )

        cameras.rescale_output_resolution(scaling_factor=1.0 / downscale_factor)

        pt_s*=downscale_factor #got a bigger pixel
        
        if "applied_transform" in meta:
            applied_transform = torch.tensor(meta["applied_transform"], dtype=transform_matrix.dtype)
            transform_matrix = transform_matrix @ torch.cat(
                [applied_transform, torch.tensor([[0, 0, 0, 1]], dtype=transform_matrix.dtype)], 0
            )
        if "applied_scale" in meta:
            applied_scale = float(meta["applied_scale"])
            scale_factor *= applied_scale
        # give eval to ori to model
        mapper_prefix="train" if split=="train" else "val"
        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames if len(mask_filenames) > 0 else None,
            dataparser_scale=scale_factor,
            dataparser_transform=transform_matrix,
            metadata={
                "depth_filenames": depth_filenames if len(depth_filenames) > 0 else None,
                "depth_unit_scale_factor": self.config.depth_unit_scale_factor,
                "sparse_pt": pt,
                "sparse_pt_scale": pt_s,
                "num_photo": total_num_photo,
                mapper_prefix+"_photo_map":idx_tensor,
                "foreground_box":fg_box,
            },
        )
        return dataparser_outputs
    def _write_pts(self, filename, pts):
        np.savetxt(filename, pts)

    def filter_pt(self, pts):
        ret = []
        # remove points where obs < 3 (outlier)
        for im_id, pt_data in pts.items():
            ret.append(len(pt_data.image_ids) >= 3)
        return np.array(ret,dtype=bool)
    #
    def orient_pca(self, pts):
        # Center the point cloud by subtracting its centroid
        centroid = np.mean(pts, axis=0)
        centered_point_cloud = pts - centroid

        # Compute the covariance matrix
        covariance_matrix = np.cov(centered_point_cloud, rowvar=False)

        # Perform eigenvalue decomposition to find the principal components
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        eigenvectors=np.flip(eigenvectors,axis=-1)#short axis align to z

        if np.linalg.det(eigenvectors) < 0:
            eigenvectors[:, 2] = -eigenvectors[:, 2]
        transform = np.eye(4)
        transform[:3,:3]=eigenvectors.T
        return transform

    # get a aabb include most of the pts
    # th: +-% pts will be excluded.
    def get_th_aabb(self, pts, th):
        idx = int(th * pts.shape[0])
        #partital sort to get min,max 5%
        front_sorted=np.partition(pts,idx,axis=0)
        end_sorted=np.partition(pts,-idx-1,axis=0)
        aabb_min=front_sorted[idx,:3]
        aabb_max=end_sorted[-idx-1,:3]
        dia=aabb_max-aabb_min
        #expend a little bit
        aabb_min-=th*dia*4
        aabb_max+=th*dia*4
        return np.stack(
                [aabb_min,
                 aabb_max], axis=0)
        
    #get a translate + scale to move aabb into [-1,1]^3 
    def get_ts(self,aabb):
        cube_box_halfsize=np.max((aabb[1,:]-aabb[0,:])/2)
        ori=aabb[0,:]+cube_box_halfsize
        translate=np.eye(4)
        translate[:3,3]=-ori
        scale=np.eye(4)
        scale[:3,:3]*=1/cube_box_halfsize
        return translate,scale
    def get_pt_scale(self, points3D,poses,fx,meta):
        #obs's image id
        obs_i=[iid for _,data in points3D.items() for iid in data.image_ids]
        obs_i=np.array(obs_i)
        #obs's pt id
        obs_p=[pid for pid,kv in enumerate(points3D.items()) for _ in kv[1].image_ids]
        obs_p=np.array(obs_p)

        w2c=np.linalg.inv(poses)#ns'w to ns'c M,4,4
        pt=np.expand_dims(np.array([pt.xyz for _, pt in points3D.items()]),-1)
        n_pt=pt.shape[0]
        pt = np.append(pt,np.ones((n_pt,1,1)),axis=1)#pt in colmap N,4,1
        tf=np.array(meta['applied_transform'])#from colmap to ns
        tf=np.append(tf,[[0,0,0,1]],axis=0)# 4*4
        pt_c=np.matmul(np.matmul(w2c[obs_i],tf),pt[obs_p]) #pt in camera= w2c*(tf*x_w), N_obs*4
        r=pt_c[:,2]**2/(fx*np.linalg.norm(pt_c[:,:3],axis=1))# radius of a ball who is about 1 pixel in camera
        pt_s=np.ones(n_pt)*np.inf
        np.minimum.at(pt_s,obs_p,r.squeeze())#reduce by pt, find the minimum radius
        return pt_s

    def _setup_downscale_factor(
        self, image_filenames: List[Path], mask_filenames: List[Path], depth_filenames: List[Path]
    ):
        """
        Setup the downscale factor for the dataset. This is used to downscale the images and cameras.
        """

        def get_fname(filepath: Path) -> Path:
            """Returns transformed file name when downscale factor is applied"""
            parts = list(filepath.parts)
            if self._downscale_factor > 1:
                parts[-2] += f"_{self._downscale_factor}"
            filepath = Path(*parts)
            return self.config.data / filepath

        filepath = next(iter(image_filenames))
        if self._downscale_factor is None:
            if self.config.downscale_factor is None:
                test_img = Image.open(self.config.data / filepath)
                h, w = test_img.size
                max_res = max(h, w)
                df = 0
                while True:
                    if (max_res / 2 ** (df)) < MAX_AUTO_RESOLUTION:
                        break
                    df += 1

                self._downscale_factor = 2**df
                CONSOLE.log(f"Using image downscale factor of {self._downscale_factor}")
            else:
                self._downscale_factor = self.config.downscale_factor

        # For downscale current images
        if self.config.largest_downscale > 1:
            image_dir = self.config.data / f"{image_filenames[0].parent}_{self.config.largest_downscale}"

            if not image_dir.exists():
                # Downscaled images not found
                # Ask if user wants to downscale the images automatically here
                CONSOLE.print(
                    f"[bold red]Downscaled images do not exist for factor of {self.config.largest_downscale}.[/bold red]"
                )
                if Confirm.ask("\nWould you like to downscale the images now?", default=False, console=CONSOLE):
                    # Install the method
                    image_dir = self.config.data / image_filenames[0].parent
                    num_downscales = int(math.log2(self.config.largest_downscale))
                    assert 2**num_downscales == self.config.largest_downscale, "Downscale factor must be a power of 2"
                    downscale_images(image_dir, num_downscales, folder_name=image_dir.name, nearest_neighbor=False)
                    if len(mask_filenames) > 0:
                        mask_dir = mask_filenames[0].parent
                        downscale_images(mask_dir, num_downscales, folder_name=mask_dir.name, nearest_neighbor=True)
                    if len(depth_filenames) > 0:
                        depth_dir = depth_filenames[0].parent
                        downscale_images(depth_dir, num_downscales, folder_name=depth_dir.name, nearest_neighbor=False)
                else:
                    sys.exit(1)

        # Return transformed filenames
        #if self._downscale_factor > 1:
        image_filenames = [get_fname(fp) for fp in image_filenames]
        mask_filenames = [get_fname(fp) for fp in mask_filenames]
        depth_filenames = [get_fname(fp) for fp in depth_filenames]
        assert isinstance(self._downscale_factor, int)
        return image_filenames, mask_filenames, depth_filenames, self._downscale_factor
