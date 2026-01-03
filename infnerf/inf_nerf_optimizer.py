import nerfstudio.engine.optimizers
from dataclasses import dataclass
from nerfstudio.engine.optimizers import OptimizerConfig
#from inf_nerf_model import InfNerfModel
from typing import Type
import torch
@dataclass
class SGDOptimizerConfig(OptimizerConfig):
    """Basic optimizer config with Adam"""

    _target: Type = torch.optim.SGD
    weight_decay: float = 0
    """The weight decay to use."""
# @dataclass
# class DummyOptimizerConfig(OptimizerConfig):
#     _target: Type = DummyOptimizer

# class DummyOptimizer():
#     def __init__(self, model:InfNerfModel):
#        self.model=model
#        pass
#     def zero_grad(self):
#         self.model.zero_grad()
#     def step(self):
#         self.model.step() 
#     def param_groups(self):
#         pass