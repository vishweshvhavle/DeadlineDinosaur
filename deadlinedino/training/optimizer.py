import torch
import numpy as np

from .. import arguments
from ..utils.wrapper import sparse_adam_update
from .scheduling_utils import Scheduler

class SparseGaussianAdam(torch.optim.Adam):
    def __init__(self, params, lr, eps, bCluster):
        self.bCluster=bCluster
        super().__init__(params=params, lr=lr, eps=eps)
    
    @torch.no_grad()
    def step(self, visible_chunk,primitive_visible):
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]

            assert len(group["params"]) == 1, "more than one tensor in group"
            param = group["params"][0]
            if param.grad is None:
                continue

            # Lazy state initialization
            state = self.state[param]
            if len(state) == 0:
                state['step'] = torch.tensor(0.0, dtype=torch.float32)
                state['exp_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format)
                state['exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)

            if self.bCluster:
                stored_state = self.state.get(param, None)
                exp_avg = stored_state["exp_avg"].view(-1,param.shape[-2],param.shape[-1])
                exp_avg_sq = stored_state["exp_avg_sq"].view(-1,param.shape[-2],param.shape[-1])
                param_view=param.data.view(-1,param.shape[-2],param.shape[-1])
                sparse_adam_update(param_view, param.grad._values().reshape(param_view.shape[0],visible_chunk.shape[0],param_view.shape[-1]), exp_avg, exp_avg_sq, visible_chunk, lr, 0.9, 0.999, eps)
            else:
                stored_state = self.state.get(param, None)
                exp_avg = stored_state["exp_avg"]
                exp_avg_sq = stored_state["exp_avg_sq"]
                N=param.shape[-1]
                sparse_adam_update(param.view(-1,N), param.grad.view(-1,N), exp_avg.view(-1,N), exp_avg_sq.view(-1,N), primitive_visible, lr, 0.9, 0.999, eps)


def get_optimizer(xyz:torch.nn.Parameter,scale:torch.nn.Parameter,rot:torch.nn.Parameter,
                  sh_0:torch.nn.Parameter,sh_rest:torch.nn.Parameter,opacity:torch.nn.Parameter,
                  spatial_lr_scale:float,
                  opt_setting:arguments.OptimizationParams,pipeline_setting:arguments.PipelineParams):
    
    l = [
        {'params': [xyz], 'lr': opt_setting.position_lr_init * spatial_lr_scale, "name": "xyz"},
        {'params': [sh_0], 'lr': opt_setting.feature_lr, "name": "sh_0"},
        {'params': [sh_rest], 'lr': opt_setting.feature_lr / 10.0, "name": "sh_rest"},
        {'params': [opacity], 'lr': opt_setting.opacity_lr, "name": "opacity"},
        {'params': [scale], 'lr': opt_setting.scaling_lr, "name": "scale"},
        {'params': [rot], 'lr': opt_setting.rotation_lr, "name": "rot"}
    ]
    if pipeline_setting.sparse_grad:
        optimizer = SparseGaussianAdam(l, lr=0, eps=1e-15,bCluster=pipeline_setting.cluster_size>0)
    else:
        optimizer = torch.optim.Adam(l, lr=0, eps=1e-15)
    scheduler = Scheduler(optimizer,opt_setting.position_lr_init*spatial_lr_scale,
              opt_setting.position_lr_final*spatial_lr_scale,
              max_epochs=opt_setting.position_lr_max_steps)
    
    return optimizer,scheduler