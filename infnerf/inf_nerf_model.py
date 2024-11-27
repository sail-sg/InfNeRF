from typing import Dict, List, Optional, Tuple, Type
from typing_extensions import Literal
from dataclasses import dataclass, field
from collections import defaultdict
import random
import numpy as np
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.functional.image import learned_perceptual_image_patch_similarity
from nerfstudio.cameras.rays import Frustums, RayBundle, RaySamples
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
    )  
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
    interlevel_node: int = 2
    "How many node apply interlevel loss in each step"
    num_interlevel_sample: int = 1024*4
    "number of interlevel loss sample in parent"
    transparency_loss_mult: float =0.001
    interlevel_loss_mult: float = 0.01
    """octree interlevel loss"""
    prop_interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    distortion_loss_mult: float = 0.002
    """Distortion loss multiplier."""
    num_samples: int = 48
    tree_config: OctreeNodeConfig=OctreeNodeConfig()
    device: str = 'cpu'
    proposal_update_every: int = 5
    """Sample every n steps after the warmup"""
    proposal_warmup: int = 5000
    """Scales n from 1 to proposal_update_every over this many steps"""
    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 96)
    """Number of samples per ray for each proposal network."""
    proposal_weights_anneal_max_num_iters: int=3000
    """Max num iterations for the annealing function."""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
   
class InfNerfModel(Model):
    config: InfNerfModelConfig
    def __init__(self, config: InfNerfModelConfig,**kwargs) -> None:
        super().__init__(config=config, **kwargs)

    def populate_modules(self):
        
        """Set the fields and modules."""
        super().populate_modules()

        self.register_buffer("scene_aabb", self.scene_box.aabb)

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
        #samplers
        self.init_sampler=UniformLinDispPiecewiseSampler(num_samples=self.config.num_samples*4)
        self.pdf_sampler = PDFSampler(include_original=False, single_jitter=True)

        self.embedding_appearance = Embedding(int(metadata["num_photo"]), self.config.appearance_embedding_dim)

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = learned_perceptual_image_patch_similarity

        self._anneal = 1.0
        self._steps_since_update = 0
        self._step = 0

        self.timer_gen=0
        self.timer_output=0
    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        N = self.config.proposal_weights_anneal_max_num_iters

        def set_anneal(step):
            # https://arxiv.org/pdf/2111.12077.pdf eq. 18
            train_frac = np.clip(step / N, 0, 1)
            self.step = step

            def bias(x, b):
                return b * x / ((b - 1) * x + 1)

            anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
            self._anneal=(anneal)
        def step_cb(step):
            """Callback to register a training step has passed. This is used to keep track of the sampling schedule"""
            self._step = step
            self._steps_since_update += 1
        return [
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=set_anneal,
            ),
            TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=step_cb,
                )
        ]
    def to(self, *args, **kwargs):
        super().to(*args,**kwargs)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["root"] = list(self.parameters())
        return param_groups
    
    def fix_ray_samples(self, ray_samples):
        ray_samples.frustums.ends=torch.maximum(ray_samples.frustums.ends,ray_samples.frustums.starts)
        ray_samples.deltas= ray_samples.frustums.ends-ray_samples.frustums.starts
        return ray_samples
    
    def get_density(self, ray_samples, detail_level:Optional[int]):
        flatten_ray_samples=ray_samples.reshape(-1)
        density=torch.zeros(*flatten_ray_samples.shape,1,dtype=torch.float32, device=ray_samples.frustums.pixel_area.device)
        idx=torch.arange(len(flatten_ray_samples),device=ray_samples.frustums.pixel_area.device)#torch.tensor(np.arange(len(flatten_ray_samples))).to(ray_samples.frustums.pixel_area.device)
        self.root.get_density(flatten_ray_samples, density, None, idx, detail_level)
        density = density.reshape(ray_samples.shape).unsqueeze(-1)
        return density
    
    def generate_ray_samples(
        self,
        ray_bundle: Optional[RayBundle] = None,
    ) -> Tuple[RaySamples, List, List]:
        assert ray_bundle is not None
        weights_list = []
        ray_samples_list = []

        n = len(self.config.num_proposal_samples_per_ray)
        weights = None
        ray_samples = None
        update_sched=np.clip(
                np.interp(self._step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
                1,
                self.config.proposal_update_every,
            )
        updated = self._steps_since_update > update_sched or self._step < 10
        for i_level in range(n + 1):#n proposal + 1 real sample
            is_prop = i_level < n
            num_samples = self.config.num_proposal_samples_per_ray[i_level] if is_prop else self.config.num_samples
            if i_level == 0:
                # Uniform sampling because we need to start with some samples
                ray_samples = self.init_sampler(ray_bundle, num_samples=num_samples)
            else:
                # PDF sampling based on the last samples and their weights
                # Perform annealing to the weights. This will be a no-op if self._anneal is 1.0.
                assert weights is not None
                annealed_weights = torch.pow(weights, self._anneal)
                ray_samples = self.pdf_sampler(ray_bundle, ray_samples, annealed_weights, num_samples=num_samples)
            if is_prop:
                if updated:
                    # always update on the first step or the inf check in grad scaling crashes
                    density=self.get_density(ray_samples, i_level)
                else:
                    with torch.no_grad():
                        density = self.get_density(ray_samples, i_level)
                weights = ray_samples.get_weights(density)
                weights_list.append(weights)  # (num_rays, num_samples)
                ray_samples_list.append(ray_samples)

        if updated:
            self._steps_since_update = 0
        assert ray_samples is not None
        return ray_samples, weights_list, ray_samples_list
    def get_outputs(self, ray_bundle: RayBundle):
       
        ray_samples, weights_list, ray_samples_list = self.generate_ray_samples(ray_bundle)
        input_device=ray_bundle.origins.device #should be self.device
        #map the camera indices in split to full
        if self.training: 
            if self.train_photo_map is not None:
                self.train_photo_map=self.train_photo_map.to(input_device)
                num_img_each_scale=self.train_photo_map.shape[0]
                ray_samples.camera_indices=self.train_photo_map[ray_samples.camera_indices%num_img_each_scale] 
        else:
            if self.val_photo_map is not None:
                mapper=self.val_photo_map
                mapper=mapper.to(input_device)
                num_img_each_scale=mapper.shape[0]
                ray_samples.camera_indices=mapper[ray_samples.camera_indices%num_img_each_scale]    
        
        flatten_ray_samples=ray_samples.reshape(-1) #[num_ray*num_sample_per_ray]
        # appearance
        if False:
            embedded_appearance = torch.zeros(
                    (*flatten_ray_samples.shape, self.config.appearance_embedding_dim), device=input_device
                )
        else:
            if self.training:
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
        idx=torch.arange(len(flatten_ray_samples),device=ray_bundle.origins.device)
        self.root.get_outputs(flatten_ray_samples, embedded_appearance, idx, tree_outputs)
        for head in tree_outputs:
            tree_outputs[head] = tree_outputs[head].reshape((*ray_samples.shape,-1))
        weights = ray_samples.get_weights(tree_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)
        if not self.training:    
            rgb = self.renderer_rgb(
                rgb=tree_outputs[FieldHeadNames.RGB],
                weights=weights,
                background_color=torch.Tensor([1.,0,0]).to(tree_outputs[FieldHeadNames.RGB].device),
            )
        else:
            rgb = self.renderer_rgb(
                rgb=tree_outputs[FieldHeadNames.RGB],
                weights=weights,
                background_color=torch.rand(3, device=tree_outputs[FieldHeadNames.RGB].device),
            )
        with torch.no_grad():
            semantics=self.renderer_rgb(rgb=tree_outputs[FieldHeadNames.SEMANTICS], weights=weights)
            depth = self.renderer_depth(
                weights=weights, 
                ray_samples=ray_samples, 
            )
            accumulation = self.renderer_accumulation(
                weights=weights, 
                )

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "semantics": semantics,
        }
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list
        if self.training:
            tree_interlevel_loss = self.get_interlevel_loss()
            transparency_loss = self.get_transparency_loss()

            outputs["tree_interlevel_loss"] = tree_interlevel_loss
            outputs["transparency_loss"] = transparency_loss
        for i in range(len(self.config.num_proposal_samples_per_ray)):
            with torch.no_grad():
                outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        return outputs

    def get_interlevel_loss(self):
        nodes=self.root.collect_tree_node(True)
        num_samples=min(len(nodes),self.config.interlevel_node)
        if num_samples==0:
            return torch.zeros((1),device = self.device)
        selected_nodes=random.sample(nodes,min(len(nodes),self.config.interlevel_node))
        return torch.stack([node.get_interlevel_loss(self.config.num_interlevel_sample, self.device) for node in selected_nodes]).mean()
    def get_transparency_loss(self):
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

        if self.training:
            if "weights_list" in outputs:
                metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])
            
            # should use module in forward stage, not loss (ddp)
            if "tree_interlevel_loss" in outputs:
                metrics_dict["interlevel_diff"] = outputs["tree_interlevel_loss"]
            if "transparency_loss" in outputs:
                metrics_dict["transparancy"]=outputs["transparency_loss"]
            
        return metrics_dict
    
    def get_loss_dict(self, outputs, batch, metrics_dict=None, mask=None) -> Dict[str, torch.Tensor]:
        loss_dict = {}
        image = batch["image"].to(self.device)
        if self.training:
            loss_dict["prop_interlevel_loss"] = self.config.prop_interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            #assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]
                      
            if "interlevel_diff" in metrics_dict:
                loss_dict["interlevel_loss"]=self.config.interlevel_loss_mult*metrics_dict["interlevel_diff"]
            if "transparancy" in metrics_dict:
                loss_dict["transparency_loss"]=self.config.transparency_loss_mult*metrics_dict["transparancy"]
        
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

        for i in range(len(self.config.num_proposal_samples_per_ray)):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        return metrics_dict, images_dict

    #rendering one image. call for evaluation and rendering
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
            for output_name, output in outputs.items():  
                if not torch.is_tensor(output):
                    # TODO: handle lists of tensors as well
                    continue
                outputs_lists[output_name].append(output)
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            if output_name in ["level"]:
                continue
            try:
                if len(outputs_list)>0 and outputs_list[0].dim!=0: 
                    outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  
            except:
                import pdb;pdb.set_trace()
        return outputs