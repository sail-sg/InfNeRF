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

"""
Code for sampling pixels.
"""

import random

import torch
from jaxtyping import Int
from torch import Tensor

import numpy as np

from dataclasses import dataclass, field
from nerfstudio.data.pixel_samplers import PixelSamplerConfig
from nerfstudio.data.utils.pixel_sampling_utils import erode_mask
from typing import (
    Dict,
    Optional,
    Type,
    Union,
)

from nerfstudio.configs.base_config import (
    InstantiateConfig,
)


@dataclass
class InfNerfPixelSamplerConfig(PixelSamplerConfig):
    """Configuration for pixel sampler instantiation."""

    _target: Type = field(default_factory=lambda: InfNerfPixelSampler)
    """Target class to instantiate."""
    num_rays_per_batch: int = 4096
    """Number of rays to sample per batch."""
    keep_full_image: bool = False
    """Whether or not to include a reference to the full image in returned batch."""
    is_equirectangular: bool = False
    """List of whether or not camera i is equirectangular."""


class InfNerfPixelSampler:
    """Samples 'pixel_batch's from 'image_batch's.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: InfNerfPixelSamplerConfig

    def __init__(self, config: InfNerfPixelSamplerConfig, **kwargs) -> None:
        self.kwargs = kwargs
        self.config = config
        # Possibly override some values if they are present in the kwargs dictionary
        self.config.num_rays_per_batch = self.kwargs.get("num_rays_per_batch", self.config.num_rays_per_batch)
        self.config.keep_full_image = self.kwargs.get("keep_full_image", self.config.keep_full_image)
        self.config.is_equirectangular = self.kwargs.get("is_equirectangular", self.config.is_equirectangular)
        self.set_num_rays_per_batch(self.config.num_rays_per_batch)

    def set_num_rays_per_batch(self, num_rays_per_batch: int):
        """Set the number of rays to sample per batch.

        Args:
            num_rays_per_batch: number of rays to sample per batch
        """
        self.num_rays_per_batch = num_rays_per_batch

    def sample_method(
        self,
        batch_size: int,
        num_images: int,
        image_height: int,
        image_width: int,
        mask: Optional[Tensor] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> Int[Tensor, "batch_size 3"]:
        """
        Naive pixel sampler, uniformly samples across all possible pixels of all possible images.

        Args:
            batch_size: number of samples in a batch
            num_images: number of images to sample over
            mask: mask of possible pixels in an image to sample from.
        """
        is_half_masked = False
        if mask is not None:
            top_left_mask_val = mask[0][0][0] # mask shape is (1, image_height, image_width, 1)
            top_right_mask_val = mask[0][0][image_width - 1]

            is_half_masked = top_left_mask_val == False or top_right_mask_val == False

        if isinstance(mask, torch.Tensor) and is_half_masked:
            mask = mask.to('cuda')
            nonzero_indices = torch.nonzero(mask[..., 0], as_tuple=False)
            chosen_indices = random.sample(range(len(nonzero_indices)), k=batch_size)
            indices = nonzero_indices[chosen_indices]
            indices = indices.cpu()
        else:
            indices = torch.floor(
                torch.rand((batch_size, 3), device=device)
                * torch.tensor([num_images, image_height, image_width], device=device)
            ).long()

        return indices

    def sample_method_equirectangular(
        self,
        batch_size: int,
        num_images: int,
        image_height: int,
        image_width: int,
        mask: Optional[Tensor] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> Int[Tensor, "batch_size 3"]:
        if isinstance(mask, torch.Tensor):
            # Note: if there is a mask, sampling reduces back to uniform sampling, which gives more
            # sampling weight to the poles of the image than the equators.
            # TODO(kevinddchen): implement the correct mask-sampling method.

            indices = self.sample_method(batch_size, num_images, image_height, image_width, mask=mask, device=device)
        else:
            # We sample theta uniformly in [0, 2*pi]
            # We sample phi in [0, pi] according to the PDF f(phi) = sin(phi) / 2.
            # This is done by inverse transform sampling.
            # http://corysimon.github.io/articles/uniformdistn-on-sphere/
            num_images_rand = torch.rand(batch_size, device=device)
            phi_rand = torch.acos(1 - 2 * torch.rand(batch_size, device=device)) / torch.pi
            theta_rand = torch.rand(batch_size, device=device)
            indices = torch.floor(
                torch.stack((num_images_rand, phi_rand, theta_rand), dim=-1)
                * torch.tensor([num_images, image_height, image_width], device=device)
            ).long()

        return indices

    def collate_image_dataset_batch(self, batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False):
        """
        Operates on a batch of images and samples pixels to use for generating rays.
        Returns a collated batch which is input to the Graph.
        It will sample only within the valid 'mask' if it's specified.

        Args:
            batch: batch of images to sample from
            num_rays_per_batch: number of rays to sample per batch
            keep_full_image: whether or not to include a reference to the full image in returned batch
        """

        device = batch["image"].device
        num_images, image_height, image_width, _ = batch["image"].shape

        if "mask" in batch:
            if self.config.is_equirectangular:
                indices = self.sample_method_equirectangular(
                    num_rays_per_batch, num_images, image_height, image_width, mask=batch["mask"], device=device
                )
            else:
                indices = self.sample_method(
                    num_rays_per_batch, num_images, image_height, image_width, mask=batch["mask"], device=device
                )
        else:
            if self.config.is_equirectangular:
                indices = self.sample_method_equirectangular(
                    num_rays_per_batch, num_images, image_height, image_width, device=device
                )
            else:
                indices = self.sample_method(num_rays_per_batch, num_images, image_height, image_width, device=device)

        c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
        c, y, x = c.cpu(), y.cpu(), x.cpu()
        collated_batch = {
            key: value[c, y, x] for key, value in batch.items() if key != "image_idx" and value is not None
        }

        assert collated_batch["image"].shape[0] == num_rays_per_batch

        # Needed to correct the random indices to their actual camera idx locations.
        indices[:, 0] = batch["image_idx"][c]
        collated_batch["indices"] = indices  # with the abs camera indices

        if keep_full_image:
            collated_batch["full_image"] = batch["image"]

        return collated_batch

    def collate_image_dataset_batch_list(self, batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False):
        """
        Does the same as collate_image_dataset_batch, except it will operate over a list of images / masks inside
        a list.

        We will use this with the intent of DEPRECIATING it as soon as we find a viable alternative.
        The intention will be to replace this with a more efficient implementation that doesn't require a for loop, but
        since pytorch's ragged tensors are still in beta (this would allow for some vectorization), this will do.

        Args:
            batch: batch of images to sample from
            num_rays_per_batch: number of rays to sample per batch
            keep_full_image: whether or not to include a reference to the full image in returned batch
        """

        device = batch["image"][0].device
        num_images = len(batch["image"])

        # only sample within the mask, if the mask is in the batch
        all_indices = []
        all_images = []

        if "mask" in batch:
            total_pixel=sum([img.shape[0]*img.shape[1] for img in batch["image"]])
            remain_pixel=num_rays_per_batch
            # num_rays_in_batch = num_rays_per_batch // num_images

            for i in range(num_images):
                image_height, image_width, _ = batch["image"][i].shape

                num_ray_in_img = num_rays_per_batch*image_height*image_width//total_pixel + 1
                num_ray_in_img = min(num_ray_in_img, remain_pixel)
                #num_ray_in_img = num_rays_per_batch // num_images
                if i == num_images - 1:
                    num_ray_in_img = remain_pixel

                if self.config.is_equirectangular:
                    indices = self.sample_method_equirectangular(
                        num_ray_in_img, 1, image_height, image_width, device=device
                    )
                else:
                    indices = self.sample_method(num_ray_in_img, 1, image_height, image_width, mask=batch["mask"][i].unsqueeze(0), device=device)

                remain_pixel-=num_ray_in_img

                # if i == num_images - 1:
                #     num_rays_in_batch = num_rays_per_batch - (num_images - 1) * num_rays_in_batch

                # indices = self.sample_method(
                #     num_rays_in_batch, 1, image_height, image_width, mask=batch["mask"][i].unsqueeze(0), device=device
                # )

                indices[:, 0] = i
                all_indices.append(indices)
                all_images.append(batch["image"][i][indices[:, 1], indices[:, 2]])
        else:
            total_pixel=sum([img.shape[0]*img.shape[1] for img in batch["image"]])
            remain_pixel=num_rays_per_batch
            #num_rays_in_batch = num_rays_per_batch // num_images
            for i in range(num_images):
                image_height, image_width, _ = batch["image"][i].shape
                num_ray_in_img = num_rays_per_batch*image_height*image_width//total_pixel + 1
                num_ray_in_img = min(num_ray_in_img, remain_pixel)
                #num_ray_in_img = num_rays_per_batch // num_images
                if i == num_images - 1:
                    num_ray_in_img = remain_pixel
                if self.config.is_equirectangular:
                    indices = self.sample_method_equirectangular(
                        num_ray_in_img, 1, image_height, image_width, device=device
                    )
                else:
                    indices = self.sample_method(num_ray_in_img, 1, image_height, image_width, device=device)
                remain_pixel-=num_ray_in_img
                indices[:, 0] = i
                all_indices.append(indices)
                all_images.append(batch["image"][i][indices[:, 1], indices[:, 2]])

        indices = torch.cat(all_indices, dim=0)
        c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
        collated_batch = {
            key: value[c, y, x]
            for key, value in batch.items()
            if key != "image_idx" and key != "image" and key != "mask" and value is not None
        }

        collated_batch["image"] = torch.cat(all_images, dim=0)

        assert collated_batch["image"].shape[0] == num_rays_per_batch

        # Needed to correct the random indices to their actual camera idx locations.
        indices[:, 0] = batch["image_idx"][c]
        collated_batch["indices"] = indices  # with the abs camera indices

        if keep_full_image:
            collated_batch["full_image"] = batch["image"]

        return collated_batch

    def sample(self, image_batch: Dict):
        """Sample an image batch and return a pixel batch.

        Args:
            image_batch: batch of images to sample from
        """
        if isinstance(image_batch["image"], list):
            image_batch = dict(image_batch.items())  # copy the dictionary so we don't modify the original
            pixel_batch = self.collate_image_dataset_batch_list(
                image_batch, self.num_rays_per_batch, keep_full_image=self.config.keep_full_image
            )
        elif isinstance(image_batch["image"], torch.Tensor):
            pixel_batch = self.collate_image_dataset_batch(
                image_batch, self.num_rays_per_batch, keep_full_image=self.config.keep_full_image
            )
        else:
            raise ValueError("image_batch['image'] must be a list or torch.Tensor")
        return pixel_batch

