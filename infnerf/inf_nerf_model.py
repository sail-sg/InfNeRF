from typing import Dict, List, Optional, Tuple, Type
from typing_extensions import Literal
from dataclasses import dataclass, field
from collections import defaultdict
import random
import numpy as np
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.functional.image import learned_perceptual_image_patch_similarity
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.model_components.scene_colliders import NearFarCollider, AABBBoxCollider
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
    UncertaintyRenderer,
)
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
)
from nerfstudio.model_components.ray_samplers import PDFSampler, UniformLinDispPiecewiseSampler, UniformSampler
from infnerf.octree_node import OctreeNode, OctreeNodeConfig
import torch
from torch.nn import Parameter
from nerfstudio.utils import colormaps

import numpy as np


@dataclass
class InfNerfModelConfig(ModelConfig):

    """Instant NGP Model Config"""

    _target: Type = field(
        default_factory=lambda: InfNerfModel
    )  # We can't write `NGPModel` directly, because `NGPModel` doesn't exist yet
    """target class to instantiate"""
    appearance_embedding_dim: int = 32
    near_plane: float = 0.03
    """How far along ray to start sampling."""
    far_plane: float = 5e0
    """How far along ray to stop sampling."""
    use_average_appearance_embedding: bool = False
    """Whether to use an appearance embedding."""
    background_color: Literal["random", "black", "white"] = "random"
    """The color that is given to untrained areas."""
    interlevel_node: int = 4
    "How many node apply interlevel loss in each step"
    num_interlevel_sample: int = 1024*2
    "number of interlevel loss sample in parent"
    transparency_loss_mult: float =0.001
    interlevel_loss_mult: float = 0.01
    """Proposal loss multiplier."""
    distortion_loss_mult: float = 0.002
    """Distortion loss multiplier."""
    num_samples: int = 48
    tree_config: OctreeNodeConfig=OctreeNodeConfig()
    device: str = 'cpu'
    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 96)
    """Number of samples per ray for each proposal network."""
   
class InfNerfModel(Model):
    config: InfNerfModelConfig
    def __init__(self, config: InfNerfModelConfig,**kwargs) -> None:
        super().__init__(config=config, **kwargs)

    def populate_modules(self):
        #import pdb;pdb.set_trace()
        
        """Set the fields and modules."""
        super().populate_modules()

        self.register_buffer("scene_aabb", self.scene_box.aabb)#Parameter(self.scene_box.aabb, requires_grad=False)
        # self.register_buffer("metadata", self.kwargs.metadata)
        #sparse_pt=self.metadata["sparse_pt"]

        #self.scene_aabb = self.scene_aabb

        metadata = self.kwargs["metadata"]
        if "sparse_pt" in metadata and "sparse_pt_scale" in metadata:
            norm_xyz_list = torch.tensor(metadata["sparse_pt"], dtype=float)
            pt_scale_lit = torch.tensor(metadata["sparse_pt_scale"], dtype=float).unsqueeze(-1)

            sparse_pt = torch.cat([norm_xyz_list, pt_scale_lit], dim=1) #xyz scale
        else:
            sparse_pt=self.scene_aabb[0,:] + torch.rand(1000,3) * (self.scene_aabb[1,:] - self.scene_aabb[0,:])
            sparse_pt=torch.cat([sparse_pt, torch.zeros(1000,1)],dim=1) #xyz scale
        if "num_photo" in metadata:
            self.num_photo=metadata["num_photo"]
        else:
            self.num_photo=0
        if "train_photo_map" in metadata:
            self.train_photo_map=metadata["train_photo_map"]#map camera indices in split to full, todo do it correctly in dataset
        else:
            self.train_photo_map=None
        if "val_photo_map" in metadata:
            self.val_photo_map=metadata["val_photo_map"]
        else:
            self.val_photo_map=None
        self.fg_box=metadata["foreground_box"]
        ##build up the octree from sparse hint recursively
        self.root=OctreeNode(
            config=self.config.tree_config,
            aabb=self.fg_box,
            sparse_pt=sparse_pt,
            depth= 0,
            num_photo=metadata["num_photo"],#photos in ori dataset, not including pyramid
            id='0'
        )

        # total_params = sum(p.numel() for p in self.root.parameters())
        # print('parameter amount: ', total_params)

        #aabbcollider?
        #self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)
        aabb=self.scene_box.aabb.clone() #tight bbox
        diagonal=aabb[1,:]-aabb[0,:]
        aabb[0,:2]-=diagonal[:2]/4
        aabb[1,:2]+=diagonal[:2]/4
        #aabb[0,2]-=diagonal[2]/16 #do not loose z so much, otherwise easy overfix
        #aabb[1,2]+=diagonal[2]/16
        #aabb[:,2]=self.scene_box.aabb[:,2] 
        bg_scene=SceneBox(aabb)
        self.collider = AABBBoxCollider(bg_scene,near_plane=self.config.near_plane)
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="expected")
        #self.renderer_uncertainty = UncertaintyRenderer()
        #samplers
        self.disp_sampler=UniformLinDispPiecewiseSampler(num_samples=self.config.num_samples*4)
        self.uni_sampler=UniformSampler(num_samples=self.config.num_samples*4)
        self.sampler_pdf = [
            PDFSampler(num_samples=self.config.num_samples*2, include_original=False),
            PDFSampler(num_samples=self.config.num_samples, include_original=False)
        ]

        self.embedding_appearance = Embedding(int(metadata["num_photo"]), self.config.appearance_embedding_dim)

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = learned_perceptual_image_patch_similarity

        # interlevel param
        self._steps_since_update = 0
        self._step = 0

        self.interlevel_loss = 0
        self.transparancy_loss = 0

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        #return []#split octree maybe?
        def update_occupancy_grid(step: int):
            # TODO: needs to get access to the sampler, on how the step size is determinated at each x. See
            # https://github.com/KAIR-BAIR/nerfacc/blob/127223b11401125a9fce5ce269bb0546ee4de6e8/examples/train_ngp_nerf.py#L190-L213
            #self.occupancy_grid.every_n_step(
            #    step=step,
            #    occ_eval_fn=lambda x: self.field.get_opacity(x, self.config.render_step_size),
            #)
            self.step=step

        return [
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=update_occupancy_grid,
            ),
        ]
    def to(self, *args, **kwargs):
        #
        #import pdb;pdb.set_trace()
        super().to(*args,**kwargs)
    # what is it for? return all octree?
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        #import pdb;pdb.set_trace()
        param_groups = {}
        # if self.field is None:
        #     raise ValueError("populate_fields() must be called before get_param_groups")
        param_groups["root"] = list(self.parameters())
        return param_groups
    def fix_ray_samples(self, ray_samples):
        ray_samples.frustums.ends=torch.maximum(ray_samples.frustums.ends,ray_samples.frustums.starts)
        ray_samples.deltas= ray_samples.frustums.ends-ray_samples.frustums.starts
        return ray_samples
    def get_density(self, ray_samples):
        flatten_ray_samples=ray_samples.reshape(-1)
        density=torch.zeros(*flatten_ray_samples.shape,1,dtype=torch.float32, device=ray_samples.frustums.pixel_area.device)
        idx=torch.tensor(np.arange(len(flatten_ray_samples))).to(ray_samples.frustums.pixel_area.device)
        self.root.get_density(flatten_ray_samples, density, None, idx)
        density = density.reshape(ray_samples.shape).unsqueeze(-1)
        return density
    def get_outputs(self, ray_bundle: RayBundle):
        with torch.no_grad():
            ray_samples = self.disp_sampler(
                ray_bundle=ray_bundle,
                #near_plane=self.config.near_plane,
                #far_plane=self.config.far_plane,
                #render_step_size=self.config.render_step_size,
                #cone_angle=self.config.cone_angle,
            )
            #ray_samples=self.fix_ray_samples(self.uni_sampler(ray_bundle))
            for i in range(len(self.sampler_pdf)):
                density  = self.get_density(ray_samples)
                weights = ray_samples.get_weights(density)
                anneal=0.9
                annealed_weights = torch.pow(weights, anneal)
                # pdf sampling
                ray_samples = self.fix_ray_samples(self.sampler_pdf[i](ray_bundle, ray_samples, annealed_weights))
        input_device=ray_bundle.origins.device #should be self.device
        #import pdb;pdb.set_trace()
        #map the camera indices in split to full
        if self.training: 
            if self.train_photo_map is not None:
                num_img_each_scale=self.train_photo_map.shape[0]
                ray_samples.camera_indices=self.train_photo_map.to(input_device)[ray_samples.camera_indices%num_img_each_scale] 
        else:
            if self.val_photo_map is not None:
                #import pdb;pdb.set_trace()
                mapper=self.val_photo_map
                num_img_each_scale=mapper.shape[0]
                ray_samples.camera_indices=mapper.to(input_device)[ray_samples.camera_indices%num_img_each_scale]    

        flatten_ray_samples=ray_samples.reshape(-1) #[num_ray*num_sample_per_ray]
        # appearance
        if self.training:
            # not predict camera appearance (例如：相机光圈，白平衡等)
            if self.num_photo == 0:
                embedded_appearance = torch.zeros(
                    (*flatten_ray_samples.shape, self.config.appearance_embedding_dim), device=input_device
                )
            else:
                embedded_appearance = self.embedding_appearance(flatten_ray_samples.camera_indices.squeeze(-1))
        else:
            if self.config.use_average_appearance_embedding:
                embedded_appearance = torch.ones(
                    (*flatten_ray_samples.shape, self.config.appearance_embedding_dim), device=input_device
                ) * self.embedding_appearance.mean(dim=0)
            else:
                embedded_appearance = self.embedding_appearance(flatten_ray_samples.camera_indices.squeeze(-1))

        tree_outputs = self.root.empty_outputs(flatten_ray_samples.shape, ray_bundle.origins.device)#todo move from node to model
        idx=torch.tensor(np.arange(len(flatten_ray_samples))).to(ray_bundle.origins.device)
        self.root.get_outputs(flatten_ray_samples, embedded_appearance, idx, tree_outputs)
        for head in tree_outputs:
            tree_outputs[head] = tree_outputs[head].reshape((*ray_samples.shape,-1))
        weights = ray_samples.get_weights(tree_outputs[FieldHeadNames.DENSITY])
        weights_list=[]
        weights_list.append(weights)
        ray_samples_list=[]
        ray_samples_list.append(ray_samples)
        if not self.training:    
            rgb = self.renderer_rgb(
                rgb=tree_outputs[FieldHeadNames.RGB],
                weights=weights,
                background_color=torch.Tensor([1.,0,0]).to(tree_outputs[FieldHeadNames.RGB].device),
                #ray_indices=ray_indices,
                #num_rays=num_rays,
            )
        else:
            rgb = self.renderer_rgb(
                rgb=tree_outputs[FieldHeadNames.RGB],
                weights=weights,
                background_color=torch.rand(3, device=tree_outputs[FieldHeadNames.RGB].device),
                #ray_indices=ray_indices,
                #num_rays=num_rays,
            )
        with torch.no_grad():
            semantics=self.renderer_rgb(rgb=tree_outputs[FieldHeadNames.SEMANTICS], weights=weights)
            depth = self.renderer_depth(
                weights=weights, 
                ray_samples=ray_samples, 
                #ray_indices=ray_indices, num_rays=num_rays
            )
            accumulation = self.renderer_accumulation(
                weights=weights, 
                #ray_indices=ray_indices, num_rays=num_rays
                )
        #alive_ray_mask = accumulation.squeeze(-1) > 0

        # if self.training and self.config.use_transient_embedding:
        #     weights_transient = ray_samples.get_weights(tree_outputs[FieldHeadNames.TRANSIENT_DENSITY])
        #     uncertainty = self.renderer_uncertainty(tree_outputs[FieldHeadNames.UNCERTAINTY], weights_transient)
        #     uncertainty = uncertainty + 0.03  # NOTE(ethan): this is the uncertainty min
        #     density_transient = tree_outputs[FieldHeadNames.TRANSIENT_DENSITY]

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "semantics": semantics,
            "weights_list":weights_list,
            "ray_samples_list":ray_samples_list,
            "level":tree_outputs["level"],
            # "uncertainty": uncertainty,
            # "density_transient": density_transient,
            #"alive_ray_mask": alive_ray_mask,  # the rays we kept from sampler
            #"num_samples_per_ray": packed_info[:, 1],
        }

        if self.training and self.config.interlevel_node > 0:
            interlevel_loss = self.get_interlevel_loss()
            transparency_loss = self.get_transparency_loss()

            outputs["interlevel_loss"] = interlevel_loss
            outputs["transparency_loss"] = transparency_loss

        return outputs

    def get_interlevel_loss(self):
        #import pdb;pdb.set_trace()
        nodes=self.root.collect_tree_node(True)
        selected_nodes=random.sample(nodes,min(len(nodes),self.config.interlevel_node))
        return torch.stack([node.get_interlevel_loss(self.config.num_interlevel_sample, self.device) for node in selected_nodes]).mean()
    def get_transparency_loss(self):
        #import pdb;pdb.set_trace()
        nodes=self.root.collect_tree_node(False)
        selected_nodes=random.sample(nodes,min(len(nodes),self.config.interlevel_node))
        return torch.stack([node.get_transparency_loss(self.config.num_interlevel_sample, self.device) for node in selected_nodes]).mean()
    def get_metrics_dict(self, outputs, batch, split="train", mask=None) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        metrics_dict = {}
        image = batch["image"].to(self.device)
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)

        if False and split == "train":
            cam_idx = batch["indices"][:, 0] % 2581
            idx = torch.isin(cam_idx, torch.tensor([50, 83, 179, 369, 567, 649, 655, 814, 840, 894, 1022, 1227, 1761, 1876, 1921, 2079, 2143, 2165, 2214, 2279, 2495]))
            metrics_dict["eval_psnr"] = self.psnr(outputs["rgb"][idx], image[idx])
            metrics_dict["eval_idx_len"] = len(image[idx])

        if self.training:
            metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])
            # if (not self.config.interlevel_node == 0):
            #     metrics_dict["interlevel_diff"] = self.get_interlevel_loss()
            # metrics_dict["transparancy"]=self.get_transparency_loss()

            # should use module in forward stage, not loss (ddp)
            if (not self.config.interlevel_node == 0):
                metrics_dict["interlevel_diff"] = outputs["interlevel_loss"]
            metrics_dict["transparancy"]=outputs["transparency_loss"]
        for d in range(self.root.get_tree_depth()+1):
            metrics_dict["lvl"+str(d)+"%"]=(outputs["level"]==d).count_nonzero()*100./(image.shape[0]*self.config.num_samples)
        return metrics_dict
    
    def get_loss_dict(self, outputs, batch, metrics_dict=None, mask=None) -> Dict[str, torch.Tensor]:
        loss_dict = {}
        image = batch["image"].to(self.device)
        # loss_dict["rgb_loss"] = self.rgb_loss(image, outputs["rgb"])
        if self.training:
            # loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
            #    outputs["weights_list"], outputs["ray_samples_list"]
            # )
            #assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]
            # if self.config.tree_config.predict_normals:
            #     # orientation loss for computed normals
            #     loss_dict["orientation_loss"] = self.config.orientation_loss_mult * torch.mean(
            #         outputs["rendered_orientation_loss"]
            #     )

            #     # ground truth supervision for normals
            #     loss_dict["pred_normal_loss"] = self.config.pred_normal_loss_mult * torch.mean(
            #         outputs["rendered_pred_normal_loss"]
            #     )
            
            if (not self.config.interlevel_node == 0) and (not self.config.interlevel_loss_mult==0):
                loss_dict["interlevel_loss"]=self.config.interlevel_loss_mult*metrics_dict["interlevel_diff"]
            loss_dict["transparency_loss"]=self.config.transparency_loss_mult*metrics_dict["transparancy"]
        
        # # transient loss
        # if self.training and self.config.use_transient_embedding:
        #     betas = outputs["uncertainty"]
        #     loss_dict["uncertainty_loss"] = 3 + torch.log(betas).mean()
        #     loss_dict["density_loss"] = 0.01 * outputs["density_transient"].mean()
        #     loss_dict["rgb_loss"] = (((image - outputs["rgb"]) ** 2).sum(-1) / (betas[..., 0] ** 2)).mean()
        # else:
        loss_dict["rgb_loss"] = self.rgb_loss(image, outputs["rgb"])

        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(self.device)
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )
        semantic = outputs["semantics"]

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        # lpips = self.lpips(image.cpu(), rgb.cpu(), normalize=True)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        # metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth, "semantic": semantic}

        # for i in range(self.config.num_proposal_iterations):
        #     key = f"prop_depth_{i}"
        #     prop_depth_i = colormaps.apply_depth_colormap(
        #         outputs[key],
        #         accumulation=outputs["accumulation"],
        #     )
        #     images_dict[key] = prop_depth_i

        return metrics_dict, images_dict

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            outputs = self.forward(ray_bundle=ray_bundle)
            for output_name, output in outputs.items():  # type: ignore
                if not torch.is_tensor(output):
                    # TODO: handle lists of tensors as well
                    continue
                outputs_lists[output_name].append(output)
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            if len(outputs_list)>0 and outputs_list[0].dim!=0: # fix for interlevel loss
                outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
        return outputs