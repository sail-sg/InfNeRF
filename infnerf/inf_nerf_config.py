from pathlib import Path
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from infnerf.inf_nerf_model import InfNerfModelConfig
from infnerf.octree_node import OctreeNodeConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from infnerf.inf_nerf_datamanager import MixMultiResDataManagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from infnerf.inf_nerf_dataparser import InfNerfDataParserConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)

GPU_scale = 16 #64

print('##### GPU_scale: ', GPU_scale)
InfNerf=MethodSpecification(
    config=TrainerConfig(
        method_name="inf-nerf",
        steps_per_eval_batch=100, # evaluation step 100
        steps_per_eval_image=500,
        steps_per_save=4000, # 1000
        save_only_latest_checkpoint=False,
        max_num_iterations=5000*4000*2000//(GPU_scale*4096),# (total pixel=w*h*num_img)/ ray_per_batch
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager = MixMultiResDataManagerConfig(
                dataparser=InfNerfDataParserConfig(
                    #scene_scale=0.5
                    downscale_factor=1,
                    largest_downscale=1, # indicate the largest downscale factor for image preparation
                ),
                train_num_images_to_sample_from=1000, #3000
                train_num_times_to_repeat_images=100, # -1
                eval_num_images_to_sample_from=50,
                eval_num_times_to_repeat_images=50,
                train_num_rays_per_batch=4096*GPU_scale,
                eval_num_rays_per_batch=1024*GPU_scale,
                train_scale=[
                    #0.0625/8,
                    #0.0625/4,
                    #0.0625/2,
                    0.0625,
                    0.0625,
                    0.125,
                    0.0625,
                    0.125,
                    0.25,
                    0.5,
                    1.0,
                ],
                eval_scale=[
                    # 0.0625/16,
                    #0.0625/8,
                    #0.0625/4,
                    #0.0625/2,
                    0.0625,
                    0.125,
                    0.25,
                    0.5,
                    1.0
                ]
            ),
            model=InfNerfModelConfig(
                tree_config = OctreeNodeConfig(
                    max_depth=4, # 4
                ),
                eval_num_rays_per_chunk=(1<<10)*GPU_scale,
            ),
        ),
        optimizers={
            "root":{
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
                }
        },
        viewer=ViewerConfig(
            num_rays_per_chunk=1 << 15, # 1 << 15
            max_num_display_images=1),
        vis=
        "viewer+tensorboard",
    ),
    description="Infnerf description"
)
