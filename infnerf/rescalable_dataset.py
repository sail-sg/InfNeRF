import numpy as np
import numpy.typing as npt
from copy import deepcopy
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, List, Dict
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.utils.colmap_parsing_utils import read_points3D_binary
from nerfstudio.data.utils.data_utils import get_image_mask_tensor_from_path
from jaxtyping import Float
from PIL import Image
from torch import Tensor
import os


# downscale_factor = int(1 / self.scale_factor)
# new_image_folder = str(self.images_path + "_" + str(downscale_factor))
# new_image_filename = str(image_filename).replace(self.images_path, new_image_folder)  # type: ignore
#
# if os.path.exists(new_image_filename):
#     pil_image = Image.open(new_image_filename)
# else:
#     pil_image = Image.open(image_filename)
#
#     if self.scale_factor != 1.0:
#         width, height = pil_image.size
#         newsize = int(width * self.scale_factor), int(height * self.scale_factor)
#         pil_image = pil_image.resize(newsize, resample=Image.BILINEAR)


class CustomInputDataset(Dataset):
    """Dataset that returns images.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs
    """

    exclude_batch_keys_from_device: List[str] = ["image", "mask"]
    cameras: Cameras

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__()
        self._dataparser_outputs = dataparser_outputs
        self.scale_factor = scale_factor
        self.scene_box = deepcopy(dataparser_outputs.scene_box)
        self.metadata = deepcopy(dataparser_outputs.metadata)
        self.cameras = deepcopy(dataparser_outputs.cameras)
        self.cameras.rescale_output_resolution(scaling_factor=scale_factor)
        self.images_path = "images"

    def __len__(self):
        return len(self._dataparser_outputs.image_filenames)

    def get_numpy_image(self, image_idx: int) -> npt.NDArray[np.uint8]:
        """Returns the image of shape (H, W, 3 or 4).

        Args:
            image_idx: The image index in the dataset.
        """
        image_filename = self._dataparser_outputs.image_filenames[image_idx]
        downscale_factor = int(1 / self.scale_factor)
        new_image_folder = str(self.images_path + "_" + str(downscale_factor))
        new_image_filename = str(image_filename).replace(self.images_path, new_image_folder)  # type: ignore

        if os.path.exists(new_image_filename):
            pil_image = Image.open(new_image_filename)
        else:
            pil_image = Image.open(image_filename)

            if self.scale_factor != 1.0:
                width, height = pil_image.size
                newsize = int(width * self.scale_factor), int(height * self.scale_factor)
                pil_image = pil_image.resize(newsize, resample=Image.BILINEAR)

        image = np.array(pil_image, dtype="uint8")  # shape is (h, w) or (h, w, 3 or 4)
        if len(image.shape) == 2:
            image = image[:, :, None].repeat(3, axis=2)
        assert len(image.shape) == 3
        assert image.dtype == np.uint8
        assert image.shape[2] in [3, 4], f"Image shape of {image.shape} is in correct."
        return image

    def get_image(self, image_idx: int) -> Float[Tensor, "image_height image_width num_channels"]:
        """Returns a 3 channel image.

        Args:
            image_idx: The image index in the dataset.
        """
        image = torch.from_numpy(self.get_numpy_image(image_idx).astype("float32") / 255.0)
        if self._dataparser_outputs.alpha_color is not None and image.shape[-1] == 4:
            image = image[:, :, :3] * image[:, :, -1:] + self._dataparser_outputs.alpha_color * (1.0 - image[:, :, -1:])
        return image

    def get_data(self, image_idx: int) -> Dict:
        """Returns the ImageDataset data as a dictionary.

        Args:
            image_idx: The image index in the dataset.
        """
        image = self.get_image(image_idx)
        data = {"image_idx": image_idx, "image": image}
        if self._dataparser_outputs.mask_filenames is not None:
            mask_filepath = self._dataparser_outputs.mask_filenames[image_idx]
            data["mask"] = get_image_mask_tensor_from_path(filepath=Path(mask_filepath), scale_factor=self.scale_factor)
            assert (
                data["mask"].shape[:2] == data["image"].shape[:2]
            ), f"Mask and image have different shapes. Got {data['mask'].shape[:2]} and {data['image'].shape[:2]}"
        metadata = self.get_metadata(data)
        data.update(metadata)
        return data

    def get_metadata(self, data: Dict) -> Dict:
        """Method that can be used to process any additional metadata that may be part of the model inputs.

        Args:
            image_idx: The image index in the dataset.
        """
        del data
        return {}

    def __getitem__(self, image_idx: int) -> Dict:
        data = self.get_data(image_idx)
        return data

    @property
    def image_filenames(self) -> List[Path]:
        """
        Returns image filenames for this dataset.
        The order of filenames is the same as in the Cameras object for easy mapping.
        """

        return self._dataparser_outputs.image_filenames




class RescalableDataset(CustomInputDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metadata
    def rescale(self, scale_factor: float = 1.0):
        if self.scale_factor==scale_factor:
            return
        self.cameras=deepcopy(self._dataparser_outputs.cameras)#don't rescale on current camera, rounding error accumulate
        self.cameras.rescale_output_resolution(scaling_factor=scale_factor)
        self.scale_factor=scale_factor

class DatasetMerger(CustomInputDataset):
    def __init__(self,dataset_groups:list, *args, **kwargs):
        #super().__init__(*args,**kwargs)
        #import pdb;pdb.set_trace()
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
        #import pdb;pdb.set_trace()
        group_id, image_idx, offset=self.id2group_id(image_idx)
        data=self.group[group_id].get_data(image_idx)
        data['image_idx']+=offset
        return data
    def get_image(self,image_idx:int):
        group_id, image_idx=self.id2group_id(image_idx)
        return self.group[group_id].get_image(image_idx)
        