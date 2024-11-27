
from dataclasses import dataclass, field
import torch
from torch import Tensor, nn
from torch.nn.functional import l1_loss, mse_loss
from nerfstudio.models.base_model import Model
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.field_components.field_heads import (
    FieldHeadNames,
    # TransientDensityFieldHead,
    # TransientRGBFieldHead,
    # UncertaintyFieldHead,
)
# from nerfstudio.fields.nerfacto_field import TCNNNerfactoField
from nerfstudio.data.scene_box import SceneBox
from infnerf.inf_nerf_field import InfNerfField
from nerfstudio.fields.base_field import Field
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.field_components.embedding import Embedding
from typing import Dict, List, Tuple, Optional, Literal
from nerfstudio.configs.base_config import PrintableConfig
from nerfstudio.field_components.encodings import SHEncoding
from nerfstudio.field_components.mlp import MLP
import numpy as np

@dataclass
class OctreeNodeConfig(PrintableConfig):
    """max depth of the octree."""
    max_depth: int = 10
    min_sparse_pt: int = 0 
    """when sparse pt less than this value, stop build tree """
    root_take_bg: bool = True 
    """root use scene contraction to take bg sample outside aabb"""
    #field
    hidden_dim: int = 64
    hidden_dim_color:int = 64

    num_layers_color: int = 3
    """Dimension of hidden layers for color network"""
    num_levels: int = 16 # 14
    """Number of levels of the hashmap for the base mlp."""
    max_res: int = 2048 # 1024
    """Maximum resolution of the hashmap for the base mlp."""
    res_upsample_factor: float = 2
    log2_hashmap_size: int = 19 # 18
    """Size of hash map is 2^log2_hashmap_size"""
    render_tree: bool = True
    device: str = 'cpu'
    implementation: Literal["tcnn", "torch"] = "tcnn"
    geo_feat_dim: int = 15
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 128, "use_linear": False},
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 256, "use_linear": False},
        ]
    )

class OctreeNode(Field):
    def __init__(
        self,
        config: OctreeNodeConfig,
        aabb: Tensor,
        sparse_pt: Tensor, #[N,4] xyzs
        depth: int,
        num_photo: int,
        id: str = None,
    )->None:
        super().__init__()
        self.config=config
        self.register_buffer("aabb",aabb) # [2,3] [min/max,xyz]
        self.register_buffer("depth",torch.Tensor([depth]).to(torch.int32))
        self.register_buffer("gsd",torch.max(self.aabb[1,:]-self.aabb[0,:])/(self.config.max_res*self.config.res_upsample_factor)) # physical size per voxel
        self.register_buffer("RGB_id",torch.rand(3)) # node's color, for tree rendering
        self.id=id 
        if (not self.config.root_take_bg) or self.depth.item()==0:#root take bg, so it need scene contraction
            scene_contraction = SceneContraction(order=float("inf"))
        else:
            scene_contraction = None
            
        # Build the proposal network(s)
        #self.density_fns = []
        num_prop_nets = len(self.config.proposal_net_args_list)
        self.proposal_networks = torch.nn.ModuleList()
        for i in range(num_prop_nets):
            prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
            network = HashMLPDensityField(
                aabb,
                spatial_distortion=scene_contraction,
                **prop_net_args,
                implementation=self.config.implementation,
            )
            self.proposal_networks.append(network)
        #self.density_fns.extend([network.density_fn for network in self.proposal_networks])
            
        self.field=InfNerfField(
            self.aabb,
            hidden_dim=config.hidden_dim,
            num_levels=config.num_levels,
            max_res=config.max_res,#2048
            log2_hashmap_size=config.log2_hashmap_size,
            hidden_dim_color=config.hidden_dim_color,
            spatial_distortion=scene_contraction,
            num_images=num_photo,
        )
        self.samples=None
        self.tree_children=torch.nn.ModuleList()

        self.direction_encoding = SHEncoding(
            levels=4,
            implementation=self.config.implementation,
        )
        
        if self.depth.item()<config.max_depth-1:#recusively build
            mask,children_mask=self.mask_pt_build(sparse_pt[...,:3],sparse_pt[...,-1:]) # 4 dimension: x, y, z, r (r means 由 pixel 放射出的锥体的 r 的大小)

            if(~mask).count_nonzero()>0:
                for i in range(len(children_mask)):
                    # construct sparse octree? if mask all zero means dont need to construct child tree
                    # can change this >= to > or change min_sparse_pt to 1(or other value > 0)
                    if children_mask[i].count_nonzero() > config.min_sparse_pt: # >= should change into >
                        self.tree_children.append(OctreeNode(
                            config=config,
                            aabb=self.idx2childbox(i),
                            sparse_pt=sparse_pt[children_mask[i]],
                            depth=self.depth.item()+1,
                            num_photo=num_photo,
                            id=self.id+str(i),
                        ))
                    else:
                        self.tree_children.append(None)

    def get_self_params(self):
        return list(self.field.parameters())
    def get_tree_params(self):
        ret=self.get_self_params()
        for child in self.tree_children:
            if child is not None:
                ret.extend(child.get_tree_params())
        return ret
    
    # """the splitting point of the children"""
    def cutting_pt(self)->Tensor:
        return torch.mean(self.aabb,0)
    
    # """get the bounding box of the i-th child"""
    def idx2childbox(self, idx: int)->Tensor:
        cutting_pt=self.cutting_pt()
        ret=self.aabb.clone()
        for i in range(3):
            if np.binary_repr(idx,3)[i]=='0':
                ret[1,i]=cutting_pt[i] #max=cut_pt
            else:
                ret[0,i]=cutting_pt[i] #min=cut pt
        return ret
    
    def is_leaf(self)->bool:
        return len(self.tree_children)==0
    
    def print_tree(self):
        print("\t"*self.depth.item()+f"{self.id}: [{self.aabb[0,0]},{self.aabb[1,0]}]*[{self.aabb[0,1]},{self.aabb[1,1]}]*[{self.aabb[0,2]},{self.aabb[1,2]}], gsd:{self.gsd} {self.aabb.device}")
        for child in self.tree_children:
            if child is None:
                print("\t"*(self.depth.item()+1)+"None")
            else:
                child.print_tree()

    #recursively get the depth of the tree            
    def get_tree_depth(self)->int:
        max_depth=self.depth
        for child in self.tree_children:
            if child is not None:
                max_depth=max(child.get_tree_depth(),max_depth)
        return max_depth
    
    def count_tree_node(self)->int:
        num=1
        for child in self.tree_children:
            if child is not None:
                num+=child.count_tree_node()
        return num
    # return all nodes as list 
    def collect_tree_node(self, exclude_leaf=False):
        if exclude_leaf and len(self.tree_children)==0:
            return []
        ret=[self]
        for child in self.tree_children:
            if child is not None:
                ret+=child.collect_tree_node(exclude_leaf)
        return ret
    def save_node_aabb(self):
        np.savetxt("tree_aabb/"+ self.id+".xyz",self.aabb.cpu().numpy())
    def save_subtree_aabb(self, only_leaves=False):
        if not only_leaves or self.is_leaf():
            self.save_node_aabb()
        for child in self.tree_children:
            if child is not None:
                child.save_subtree_aabb(only_leaves)

    #split pt in R^3 into 8 tensor
    def cut_pt(self,pt:Tensor)->List:
        ret=[]
        masks=self.cut(pt)
        for i in range(8):
            ret.append(pt[masks[i]])
        return ret
    
    # create a mask for splitting pt into 8 children along the cutting point
    def cut(self,pt:Tensor)-> List[Tensor]:#[N,3]->[8][N]
        cutting_pt=self.cutting_pt().to(pt.device)
        split=pt<cutting_pt # b 3 in boolean, like [True,True,False]
        ret=[]
        for i in range(8): # 8 children
            mask=torch.tensor(np.array(list(np.binary_repr(i,3)))=='0').to(pt.device) # like 0->[F,F,F] 6->[T,T,F]
            ret.append(torch.all(split==mask,-1)) # b
        return ret
    
    # distribute the sparse points to self or 8 children. only call when build tree
    def mask_pt_build(self,
                pt:Tensor,#[N,3]
                scale:Tensor,#[N,1]
                )-> Tuple[Tensor, List[Tensor]]:#[N],[8][N]
        self_mask=self.self_mask(pt, scale.squeeze(-1)).to(pt.device)
        children_mask=self.cut(pt) # [8, N] type bool, represent whether pt inside current children grid
        for i in range(len(children_mask)):
            children_mask[i]=torch.logical_and(children_mask[i],torch.logical_not(self_mask))
        return self_mask, children_mask
    
    # pick the sample for this node (not for children)
    # all sample bigger then gsd should be rendered by self, otherwise rendered by child
    def self_mask(self,
                  pt: Tensor, #[N,3] 
                  scale:Tensor,#[N]
                  )->Tensor:#[N],bool
        if self.config.root_take_bg and self.depth==0: # root take bg
            bg=torch.logical_or(torch.any(pt>self.aabb[1],-1),torch.any(pt<self.aabb[0],-1))
            return torch.logical_or(scale>self.gsd,bg) 
        return scale>self.gsd
    
    # distribute the sample points to self or 8 children. different from mask_pt_build, only call when rendering
    def mask_pt(self,
                pt:Tensor,#[N,3]
                scale:Tensor,#[N,1]
                idx,
                )-> Tuple[Tensor, List[Tensor]]:#[N],[8][N]
        if len(self.tree_children)==0:#at the leaf, take it all
            return idx, []
        pt = pt.reshape(-1, 3)
        scale = scale.reshape(-1)
        self_mask=self.self_mask(pt[idx],scale[idx])#n_idx->bool
        #if self.depth==2: #debug render by given depth
        #    self_mask=torch.ones_like(self_mask)
        #else:
        #    self_mask=torch.zeros_like(self_mask)
        self_idx = idx[self_mask]

        cutting_pt = self.cutting_pt().to(pt.device)
        split = pt[idx] < cutting_pt #(n_idx,3) bool
        children_idx = []
        for i in range(8):
            mask = torch.tensor(np.array(list(np.binary_repr(i, 3))) == '0').to(pt.device) #like [True,True,False]
            child_idx = idx[torch.logical_and(~self_mask, torch.all(split == mask, -1).flatten())]
            children_idx.append(child_idx)

        return self_idx, children_idx
    
    # get self mask and 8 children mask
    # call only in train and eval, not in construction
    def mask_samples(self, ray_samples, idx) -> Tuple[Tensor, List[Tensor]]:        
        frustums=ray_samples.frustums
        radius = torch.sqrt(frustums.pixel_area)/1.7724
        if False:
            scale=(frustums.starts+frustums.ends)/2*radius#*2**((torch.rand_like(radius))*0.00) #[N,1]
        else:
            scale=(frustums.starts+frustums.ends)/2*radius*2**(torch.rand_like(radius)-0.5) #[N,1]
        self_mask, children_mask = self.mask_pt(frustums.get_positions(), scale, idx)
        # Parent should take care of the None child. 
        # None child means no info, no need to subdivide, doesn't mean physically empty.
        empty_children=[self_mask]
        for i in range(len(children_mask)):
            if self.tree_children[i] is None:
                empty_children.append(children_mask[i])
                children_mask[i]=torch.Tensor([])
        self_mask=torch.cat(empty_children)
        return self_mask, children_mask

    def empty_outputs(self, shape, device) -> Dict[FieldHeadNames, Tensor]:
        outputs ={}
        if self.config.render_tree:
            outputs[FieldHeadNames.SEMANTICS] = self.RGB_id.repeat(*shape,1).to(device)
        outputs[FieldHeadNames.RGB] =torch.zeros(*shape,3, device=device)
        outputs[FieldHeadNames.DENSITY]=torch.zeros(*shape,1,dtype=torch.float32, device=device)
        outputs["level"]=torch.zeros(*shape,1,dtype=torch.int32,device=device)
        
        return outputs

    def get_outputs(
        self, 
        ray_samples: RaySamples,
        embedded_appearance: Tensor,
        idx: Optional[Tensor] = None,
        outputs: Optional[dict] = None,
    ) -> Dict[FieldHeadNames, Tensor]:
        self_idx, children_idx = self.mask_samples(ray_samples, idx)
        if not self_idx.shape[0]==0:
            self_samples=ray_samples[self_idx]
            density, density_embedding=self.field.get_density(self_samples)
            self_output = self.field.get_outputs(self_samples, density_embedding, embedded_appearance[self_idx,:])
            for head in self_output:
                outputs[head][self_idx] = self_output[head]
            outputs[FieldHeadNames.SEMANTICS][self_idx] = 0.8 * self.RGB_id.repeat(*self_samples.shape,1) + 0.2 * self_output[FieldHeadNames.RGB]
            outputs[FieldHeadNames.DENSITY][self_idx] = density
            outputs["level"][self_idx]=self.depth
        for i in range(len(self.tree_children)):
            child_idx = children_idx[i]
            if child_idx.numel()==0 or self.tree_children[i] is None:
                continue
            else:
                self.tree_children[i].get_outputs(ray_samples, embedded_appearance, child_idx, outputs)
        return
    
    def get_density(self, ray_samples:RaySamples, all_density:Optional[Tensor], all_feat:Optional[Tensor], idx:Optional[Tensor], detail_level: Optional[int]):
        self_idx, children_idx = self.mask_samples(ray_samples, idx)
        if detail_level is None or detail_level>=len(self.proposal_networks):
            density_fn=self.field.get_density
        else:
            density_fn=self.proposal_networks[detail_level].get_density
        if not self_idx.shape[0]==0:
            self_sample = ray_samples[self_idx]
            if all_feat is not None:
                all_density[self_idx], all_feat[self_idx] = density_fn(self_sample)
            else:
                all_density[self_idx], _ = density_fn(self_sample)

        for i in range(len(self.tree_children)):#0 or 8
            child_idx = children_idx[i]
            if child_idx.numel()==0 or self.tree_children[i] is None:
                continue
            else:
                self.tree_children[i].get_density(ray_samples, all_density, all_feat, child_idx, detail_level)
        return

    def get_interlevel_loss(self, num_interlvl_sample, device):
        loss=0
        for child in self.tree_children:
            if child is None:
                continue
            delta=child.gsd
            rand_sample=\
                (child.aabb[0,:]+delta)\
                +torch.rand(num_interlvl_sample,3, device = device)*(child.aabb[1,:]-child.aabb[0,:]-2*delta)
            #cut off the limit to avoid overflow
            high_res=torch.stack(
                [   
                    rand_sample,
                    rand_sample+torch.Tensor([-delta,0,0]).to(device),
                    rand_sample+torch.Tensor([delta,0,0]).to(device),
                    rand_sample+torch.Tensor([0,-delta,0]).to(device),
                    rand_sample+torch.Tensor([0,delta,0]).to(device),
                    rand_sample+torch.Tensor([0,0,-delta]).to(device),
                    rand_sample+torch.Tensor([0,0,+delta]).to(device)
                ],1)#[N,7,3]
            den_high,emb_high=child.field.get_density2(high_res)
            trans_high=(-den_high).exp()
            trans_high = trans_high.mean(dim=1) #N,1
            den_low, emb_low=self.field.get_density2(rand_sample) #todo take field out of nerfstudio
            trans_low=(-den_low).exp()
            loss+=mse_loss(trans_high,trans_low) #+self.interlvl_loss(emb_high.mean(dim=1),emb_low) #todo test interlvel loss on emb
        return loss

    def get_transparency_loss(self, num_interlvl_sample, device):
        if self.field.spatial_distortion is None:
            rand_sample=self.aabb[0,:]+torch.rand(num_interlvl_sample,3, device = device)*(self.aabb[1,:]-self.aabb[0,:])
        else:
            rand_sample=self.aabb.mean()+torch.randn(num_interlvl_sample,3, device = device)*(self.aabb[1,:]-self.aabb[0,:])
        density=[]
        field_density,_= self.field.get_density2(rand_sample)
        density.append(field_density)
        for i in range(len(self.proposal_networks)):
            density.append(self.proposal_networks[i].density_fn(rand_sample[:int(num_interlvl_sample/4),:]))
        density=torch.cat(density)
        transparency=(-density).exp()
        return l1_loss(transparency, torch.ones_like(transparency, device=transparency.device))
