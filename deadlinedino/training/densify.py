import torch
import math

from typing import Optional

class DensityControllerBase:
    def __init__(self, densify_params, bCluster: bool) -> None:
        self.densify_params = densify_params
        self.bCluster = bCluster
        return
    
    @torch.no_grad()
    def step(self, optimizer: torch.optim.Optimizer, epoch: int, scheduler=None):
        return
    
    @torch.no_grad()
    def _get_params_from_optimizer(self, optimizer: torch.optim.Optimizer):
        param_dict = {}
        for param_group in optimizer.param_groups:
            name = param_group['name']
            tensor = param_group['params'][0]
            param_dict[name] = tensor
        xyz = param_dict["xyz"]
        rot = param_dict["rot"]
        scale = param_dict["scale"]
        sh_0 = param_dict["sh_0"]
        sh_rest = param_dict["sh_rest"]
        opacity = param_dict["opacity"]
        return xyz, scale, rot, sh_0, sh_rest, opacity

    @torch.no_grad()
    def _cat_tensors_to_optimizer(self, tensors_dict: dict, optimizer: torch.optim.Optimizer):
        cat_dim = -1
        if self.bCluster:
            cat_dim = -2
        for group in optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = optimizer.state.get(group['params'][0], None)
            assert stored_state["exp_avg"].shape == stored_state["exp_avg_sq"].shape and \
                   stored_state["exp_avg"].shape == group["params"][0].shape
            if stored_state is not None:
                stored_state["exp_avg"].data = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=cat_dim
                ).contiguous()
                stored_state["exp_avg_sq"].data = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=cat_dim
                ).contiguous()
            new_param = torch.cat((group["params"][0], extension_tensor), dim=cat_dim).contiguous()
            optimizer.state.pop(group['params'][0])
            group["params"][0] = torch.nn.Parameter(new_param)
            optimizer.state[group["params"][0]] = stored_state
            assert stored_state["exp_avg"].shape == stored_state["exp_avg_sq"].shape and \
                   stored_state["exp_avg"].shape == group["params"][0].shape
        return
    
    @torch.no_grad()
    def _replace_tensor_to_optimizer(self, tensor: torch.Tensor, name: str, 
                                      optimizer: torch.optim.Optimizer):
        for group in optimizer.param_groups:
            if group["name"] in ["appearance_embeddings", "appearance_network"]:
                continue
            if group["name"] == name:
                stored_state = optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                del optimizer.state[group['params'][0]]
                group["params"][0] = torch.nn.Parameter(tensor.requires_grad_(True))
                optimizer.state[group['params'][0]] = stored_state
        return
    
    @torch.no_grad()
    def _prune_optimizer(self, valid_mask: torch.Tensor, optimizer: torch.optim.Optimizer):
        # Import cluster module locally to avoid circular import
        try:
            from ..scene import cluster
        except:
            from scene import cluster
            
        for group in optimizer.param_groups:
            stored_state = optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                if self.bCluster:
                    chunk_size = stored_state["exp_avg"].shape[-1]
                    uncluster_avg, uncluster_avg_sq = cluster.uncluster(
                        stored_state["exp_avg"], stored_state["exp_avg_sq"]
                    )
                    uncluster_avg = uncluster_avg[..., valid_mask]
                    uncluster_avg_sq = uncluster_avg_sq[..., valid_mask]
                    new_avg, new_avg_sq = cluster.cluster_points(
                        chunk_size, uncluster_avg, uncluster_avg_sq
                    )
                else:
                    new_avg = stored_state["exp_avg"][..., valid_mask]
                    new_avg_sq = stored_state["exp_avg_sq"][..., valid_mask]
                stored_state["exp_avg"].data = new_avg
                stored_state["exp_avg_sq"].data = new_avg_sq
            
            if self.bCluster:
                chunk_size = group["params"][0].shape[-1]
                uncluster_param, = cluster.uncluster(group["params"][0])
                uncluster_param = uncluster_param[..., valid_mask]
                new_param, = cluster.cluster_points(chunk_size, uncluster_param)
            else:
                new_param = group["params"][0][..., valid_mask]
            optimizer.state.pop(group['params'][0])
            group["params"][0] = torch.nn.Parameter(new_param)
            optimizer.state[group["params"][0]] = stored_state
        return


class DensityControllerOfficial(DensityControllerBase):
    @torch.no_grad()
    def __init__(self, screen_extent: float, densify_params, bCluster: bool, 
                 init_points_num: int) -> None:
        self.grad_threshold = densify_params.densify_grad_threshold
        self.min_opacity = densify_params.opacity_threshold
        self.percent_dense = densify_params.percent_dense
        self.screen_extent = screen_extent
        self.max_screen_size = densify_params.screen_size_threshold
        self.init_points_num = init_points_num
        super(DensityControllerOfficial, self).__init__(densify_params, bCluster)
        return
    
    @torch.no_grad()
    def get_prune_mask(self, actived_opacity: torch.Tensor, actived_scale: torch.Tensor):
        # Import locally
        try:
            from ..utils.statistic_helper import StatisticsHelperInst
        except:
            from utils.statistic_helper import StatisticsHelperInst
            
        transparent = (actived_opacity < self.min_opacity).squeeze()
        invisible = StatisticsHelperInst.get_global_culling()
        prune_mask = transparent
        prune_mask[:invisible.shape[0]] |= invisible
        return prune_mask

    @torch.no_grad()
    def get_clone_mask(self, actived_scale: torch.Tensor):
        try:
            from ..utils.statistic_helper import StatisticsHelperInst
        except:
            from utils.statistic_helper import StatisticsHelperInst
            
        mean2d_grads = StatisticsHelperInst.get_mean('mean2d_grad').squeeze()
        abnormal_mask = mean2d_grads >= self.grad_threshold
        tiny_pts_mask = actived_scale.max(dim=0).values <= self.percent_dense * self.screen_extent
        selected_pts_mask = abnormal_mask & tiny_pts_mask
        return selected_pts_mask
    
    @torch.no_grad()
    def get_split_mask(self, actived_scale: torch.Tensor, N=2):
        try:
            from ..utils.statistic_helper import StatisticsHelperInst
        except:
            from utils.statistic_helper import StatisticsHelperInst
            
        mean2d_grads = StatisticsHelperInst.get_mean('mean2d_grad').squeeze()
        abnormal_mask = mean2d_grads >= self.grad_threshold
        large_pts_mask = actived_scale.max(dim=0).values > self.percent_dense * self.screen_extent
        selected_pts_mask = abnormal_mask & large_pts_mask
        return selected_pts_mask
    
    @torch.no_grad()
    def prune(self, optimizer: torch.optim.Optimizer, epoch: int):
        try:
            from ..scene import cluster
        except:
            from scene import cluster

        xyz, scale, rot, sh_0, sh_rest, opacity = self._get_params_from_optimizer(optimizer)
        if self.bCluster:
            chunk_size = xyz.shape[-1]
            before_count = xyz.shape[-2] * xyz.shape[-1]
            xyz, scale, rot, sh_0, sh_rest, opacity = cluster.uncluster(
                xyz, scale, rot, sh_0, sh_rest, opacity
            )
        else:
            before_count = xyz.shape[-1]

        prune_mask = self.get_prune_mask(opacity.sigmoid(), scale.exp())
        if prune_mask.sum() > 0.8 * opacity.shape[1]:
            assert False, "Pruning too many primitives!"

        if self.bCluster:
            N = prune_mask.sum()
            chunk_num = int(N / chunk_size)
            del_limit = chunk_num * chunk_size
            del_indices = prune_mask.nonzero()[:del_limit, 0]
            prune_mask = torch.zeros_like(prune_mask)
            prune_mask[del_indices] = True

        pruned_count = prune_mask.sum().item()
        self._prune_optimizer(~prune_mask, optimizer)

        # Get count after pruning
        xyz_after, _, _, _, _, _ = self._get_params_from_optimizer(optimizer)
        if self.bCluster:
            after_count = xyz_after.shape[-2] * xyz_after.shape[-1]
        else:
            after_count = xyz_after.shape[-1]

        # Debug output
        print(f"[PRUNE][Epoch {epoch}] before={before_count}, pruned={int(pruned_count)}, after={after_count}")
        return

    @torch.no_grad()
    def split_and_clone(self, optimizer: torch.optim.Optimizer, epoch: int, 
                       densify_rate: Optional[float] = None):
        try:
            from ..scene import cluster
            from ..utils import wrapper
        except:
            from scene import cluster
            from utils import wrapper
            
        xyz, scale, rot, sh_0, sh_rest, opacity = self._get_params_from_optimizer(optimizer)
        if self.bCluster:
            chunk_size = xyz.shape[-1]
            xyz, scale, rot, sh_0, sh_rest, opacity = cluster.uncluster(
                xyz, scale, rot, sh_0, sh_rest, opacity
            )

        clone_mask = self.get_clone_mask(scale.exp())
        split_mask = self.get_split_mask(scale.exp())

        # Split
        stds = scale[..., split_mask].exp()
        means = torch.zeros((3, stds.size(-1)), device="cuda")
        samples = torch.normal(mean=means, std=stds).unsqueeze(0)
        transform_matrix = wrapper.CreateTransformMatrix.call_fused(
            torch.ones_like(scale[..., split_mask].exp()),
            torch.nn.functional.normalize(rot[..., split_mask], dim=0)
        )
        transform_matrix = transform_matrix[:3, :3]
        shift = (samples.permute(2, 0, 1)) @ transform_matrix.permute(2, 0, 1)
        shift = shift.permute(1, 2, 0).squeeze(0)
        
        split_xyz = xyz[..., split_mask] + shift
        clone_xyz = xyz[..., clone_mask]
        append_xyz = torch.cat((split_xyz, clone_xyz), dim=-1)
        
        split_scale = (scale[..., split_mask].exp() / (0.8 * 2)).log()
        clone_scale = scale[..., clone_mask]
        append_scale = torch.cat((split_scale, clone_scale), dim=-1)

        split_rot = rot[..., split_mask]
        clone_rot = rot[..., clone_mask]
        append_rot = torch.cat((split_rot, clone_rot), dim=-1)

        split_sh_0 = sh_0[..., split_mask]
        clone_sh_0 = sh_0[..., clone_mask]
        append_sh_0 = torch.cat((split_sh_0, clone_sh_0), dim=-1)

        split_sh_rest = sh_rest[..., split_mask]
        clone_sh_rest = sh_rest[..., clone_mask]
        append_sh_rest = torch.cat((split_sh_rest, clone_sh_rest), dim=-1)

        split_opacity = opacity[..., split_mask]
        clone_opacity = opacity[..., clone_mask]
        append_opacity = torch.cat((split_opacity, clone_opacity), dim=-1)

        if self.bCluster:
            N = append_xyz.shape[-1]
            chunk_num = int(N / chunk_size)
            append_limit = chunk_num * chunk_size
            append_xyz, append_scale, append_rot, append_sh_0, append_sh_rest, append_opacity = \
                cluster.cluster_points(
                    chunk_size, append_xyz[..., :append_limit], append_scale[..., :append_limit],
                    append_rot[..., :append_limit], append_sh_0[..., :append_limit],
                    append_sh_rest[..., :append_limit], append_opacity[..., :append_limit]
                )

        dict_clone = {
            "xyz": append_xyz,
            "scale": append_scale,
            "rot": append_rot,
            "sh_0": append_sh_0,
            "sh_rest": append_sh_rest,
            "opacity": append_opacity
        }
        
        self._cat_tensors_to_optimizer(dict_clone, optimizer)
        
        # Return number of naturally added primitives for momentum tracking
        return append_xyz.shape[-1] if not self.bCluster else append_xyz.shape[-2]
    
    @torch.no_grad()
    def reset_opacity(self, optimizer: torch.optim.Optimizer, epoch: int):
        xyz, scale, rot, sh_0, sh_rest, opacity = self._get_params_from_optimizer(optimizer)
        
        def inverse_sigmoid(x):
            return torch.log(x / (1 - x))
        
        actived_opacities = opacity.sigmoid()
        if self.densify_params.opacity_reset_mode == 'decay':
            decay_rate = 0.5
            opacity.data = inverse_sigmoid((actived_opacities * decay_rate).clamp_min(1.0 / 128))
            optimizer.state.clear()
        elif self.densify_params.opacity_reset_mode == 'reset':
            opacity.data = inverse_sigmoid(actived_opacities.clamp_max(0.005))
            self._replace_tensor_to_optimizer(opacity, "opacity", optimizer)
        return
    
    @torch.no_grad()
    def is_densify_actived(self, epoch: int):
        return epoch < self.densify_params.densify_until and \
               epoch >= self.densify_params.densify_from and \
               (epoch % self.densify_params.densification_interval == 0)

    @torch.no_grad()
    def step(self, optimizer: torch.optim.Optimizer, epoch: int, scheduler=None,
             current_iteration=None, current_n_primitives=None, current_render_scale=None):
        try:
            from ..utils.statistic_helper import StatisticsHelperInst
        except:
            from utils.statistic_helper import StatisticsHelperInst

        # Store initial stats for debug output
        xyz, scale, rot, sh_0, sh_rest, opacity = self._get_params_from_optimizer(optimizer)
        if self.bCluster:
            before_count = xyz.shape[-2] * xyz.shape[-1]
        else:
            before_count = xyz.shape[-1]
        before_opacity = opacity.sigmoid().mean().item()
        before_scale = scale.exp().mean().item()

        if epoch < self.densify_params.densify_until and epoch >= self.densify_params.densify_from:
            bUpdate = False
            densification_happened = False

            if epoch % self.densify_params.densification_interval == 0:
                # Get densify rate from scheduler if available
                densify_rate = None
                if scheduler is not None and current_iteration is not None and \
                   current_n_primitives is not None and current_render_scale is not None:
                    densify_rate = scheduler.get_densify_rate(
                        current_iteration, current_n_primitives, current_render_scale
                    )

                # Perform densification
                momentum_add = self.split_and_clone(optimizer, epoch, densify_rate)

                # Update momentum (Phase 1 only)
                if scheduler is not None and momentum_add is not None:
                    scheduler.update_momentum(momentum_add)

                self.prune(optimizer, epoch)
                bUpdate = True
                densification_happened = True

            if epoch % self.densify_params.opacity_reset_interval == 0:
                self.reset_opacity(optimizer, epoch)
                bUpdate = True

            if bUpdate:
                xyz, scale, rot, sh_0, sh_rest, opacity = self._get_params_from_optimizer(optimizer)
                StatisticsHelperInst.reset(xyz.shape[-2], xyz.shape[-1], self.is_densify_actived)
                torch.cuda.empty_cache()

                # Get stats after densification
                if self.bCluster:
                    after_count = xyz.shape[-2] * xyz.shape[-1]
                else:
                    after_count = xyz.shape[-1]
                after_opacity = opacity.sigmoid().mean().item()
                after_scale = scale.exp().mean().item()

                if densification_happened:
                    delta = after_count - before_count
                    print(f"[DENSIFY][Epoch {epoch}] Gaussians: {before_count} -> {after_count} "
                          f"(Δ = {delta:+d}), mean_opacity: {before_opacity:.6f} -> {after_opacity:.6f}, "
                          f"mean_scale: {before_scale:.6f} -> {after_scale:.6f}")
            else:
                # Active window but no densification step
                print(f"[DENSIFY][Epoch {epoch}] active window but no densification step. "
                      f"Gaussians: {before_count}, mean_opacity: {before_opacity:.6f}, "
                      f"mean_scale: {before_scale:.6f}")

        return self._get_params_from_optimizer(optimizer)


class DensityControllerTamingGS(DensityControllerOfficial):
    """
    Hybrid densification controller:
    - Phase 1 (scale > 1): Score-based multinomial sampling (TamingGS)
    - Phase 2 (scale = 1): Linear target growth (LiteGS)
    
    Uses scheduler's densify_rate to control budget.
    """
    
    @torch.no_grad()
    def __init__(self, screen_extent: float, densify_params, bCluster: bool, 
                 init_points_num: int) -> None:
        # TamingGS requires target_primitives, but we'll compute it dynamically
        if densify_params.target_primitives == 0:
            # Will be set by scheduler dynamically
            densify_params.target_primitives = init_points_num * 5
        
        self.target_points_num = densify_params.target_primitives
        super(DensityControllerTamingGS, self).__init__(
            screen_extent, densify_params, bCluster, init_points_num
        )
        return
    
    @torch.no_grad()
    def get_prune_mask(self, actived_opacity: torch.Tensor, actived_scale: torch.Tensor):
        try:
            from ..utils.statistic_helper import StatisticsHelperInst
        except:
            from utils.statistic_helper import StatisticsHelperInst
            
        if self.densify_params.prune_mode == 'weight':
            prune_mask = torch.zeros(actived_opacity.shape[1], device=actived_opacity.device).bool()
            frag_weight, frag_count = StatisticsHelperInst.get_mean('fragment_weight')
            weight_sum = (frag_weight * frag_count).nan_to_num(0).squeeze()
            invisible = weight_sum == 0
            prune_mask[:invisible.shape[0]] |= invisible
        elif self.densify_params.prune_mode == 'threshold':
            prune_mask = super(DensityControllerTamingGS, self).get_prune_mask(
                actived_opacity, actived_scale
            )
        return prune_mask
    
    def get_score(self, xyz, scale, rot, sh_0, sh_rest, opacity):
        try:
            from ..utils.statistic_helper import StatisticsHelperInst
        except:
            from utils.statistic_helper import StatisticsHelperInst
            
        var, frag_count = StatisticsHelperInst.get_var('fragment_err')
        score = var * frag_count * (opacity.sigmoid() * opacity.sigmoid())
        score = score.squeeze().nan_to_num(0)
        score.clamp_min_(0)
        return score
    
    @torch.no_grad()
    def split_and_clone(self, optimizer: torch.optim.Optimizer, epoch: int, 
                       densify_rate: Optional[float] = None):
        """
        Hybrid densification with scheduler-controlled budget:
        - Uses densify_rate from scheduler to compute budget
        - Phase 1: Momentum-based adaptive budget
        - Phase 2: Fixed linear target budget
        """
        try:
            from ..scene import cluster
            from ..utils import wrapper
        except:
            from scene import cluster
            from utils import wrapper
            
        xyz, scale, rot, sh_0, sh_rest, opacity = self._get_params_from_optimizer(optimizer)
        if self.bCluster:
            chunk_size = xyz.shape[-1]
            xyz, scale, rot, sh_0, sh_rest, opacity = cluster.uncluster(
                xyz, scale, rot, sh_0, sh_rest, opacity
            )

        before_count = xyz.shape[-1]
        prune_num = self.get_prune_mask(opacity.sigmoid(), scale.exp()).sum().item()

        # === BUDGET CALCULATION ===
        if densify_rate is not None:
            # Use scheduler's rate (hybrid mode)
            target_add = int(densify_rate * self.init_points_num)
            budget = min(max(target_add + int(prune_num), 1), xyz.shape[-1])
        else:
            # Fallback: original TamingGS linear growth
            cur_target_count = (self.target_points_num - self.init_points_num) / \
                              (self.densify_params.densify_until - self.densify_params.densify_from) * \
                              (epoch - self.densify_params.densify_from) + self.init_points_num
            budget = min(max(int(cur_target_count - xyz.shape[-1]), 1) + int(prune_num), xyz.shape[-1])

        # Score-based multinomial sampling (TamingGS)
        score = self.get_score(xyz, scale, rot, sh_0, sh_rest, opacity)
        densify_index = torch.multinomial(score, budget, replacement=False)
        
        # Separate clone/split based on size
        clone_index = densify_index[
            (scale[:, densify_index].exp().max(dim=0).values <= self.percent_dense * self.screen_extent)
        ]
        split_index = densify_index[
            (scale[:, densify_index].exp().max(dim=0).values > self.percent_dense * self.screen_extent)
        ]

        # Split
        stds = scale[..., split_index].exp()
        means = torch.zeros((3, stds.size(-1)), device="cuda")
        samples = torch.normal(mean=means, std=stds).unsqueeze(0)
        transform_matrix = wrapper.CreateTransformMatrix.call_fused(
            torch.ones_like(scale[..., split_index]),
            torch.nn.functional.normalize(rot[..., split_index], dim=0)
        )
        transform_matrix = transform_matrix[:3, :3]
        shift = (samples.permute(2, 0, 1)) @ transform_matrix.permute(2, 0, 1)
        shift = shift.permute(1, 2, 0).squeeze(0)
        
        split_xyz = xyz[..., split_index] + shift
        clone_xyz = xyz[..., clone_index]
        append_xyz = torch.cat((split_xyz, clone_xyz), dim=-1)
        
        split_scale = (scale[..., split_index].exp() / (0.8 * 2)).log()
        clone_scale = scale[..., clone_index]
        append_scale = torch.cat((split_scale, clone_scale), dim=-1)

        split_rot = rot[..., split_index]
        clone_rot = rot[..., clone_index]
        append_rot = torch.cat((split_rot, clone_rot), dim=-1)

        split_sh_0 = sh_0[..., split_index]
        clone_sh_0 = sh_0[..., clone_index]
        append_sh_0 = torch.cat((split_sh_0, clone_sh_0), dim=-1)

        split_sh_rest = sh_rest[..., split_index]
        clone_sh_rest = sh_rest[..., clone_index]
        append_sh_rest = torch.cat((split_sh_rest, clone_sh_rest), dim=-1)

        split_opacity = opacity[..., split_index]
        clone_opacity = opacity[..., clone_index]
        append_opacity = torch.cat((split_opacity, clone_opacity), dim=-1)

        if self.bCluster:
            N = append_xyz.shape[-1]
            chunk_num = int(N / chunk_size)
            append_limit = chunk_num * chunk_size
            append_xyz, append_scale, append_rot, append_sh_0, append_sh_rest, append_opacity = \
                cluster.cluster_points(
                    chunk_size, append_xyz[..., :append_limit], append_scale[..., :append_limit],
                    append_rot[..., :append_limit], append_sh_0[..., :append_limit],
                    append_sh_rest[..., :append_limit], append_opacity[..., :append_limit]
                )

        # Calculate counts for debug output
        added_count = append_xyz.shape[-1] if not self.bCluster else append_xyz.shape[-2]
        after_count = before_count + added_count
        split_count = split_index.shape[0]
        clone_count = clone_index.shape[0]

        # Debug output
        print(f"[TAMING-GS][Epoch {epoch}] target={self.target_points_num}, budget={budget}, "
              f"before={before_count}, added={added_count}, after={after_count}, "
              f"prune_num={int(prune_num)}, split={split_count}, clone={clone_count}")

        dict_clone = {
            "xyz": append_xyz,
            "scale": append_scale,
            "rot": append_rot,
            "sh_0": append_sh_0,
            "sh_rest": append_sh_rest,
            "opacity": append_opacity
        }

        self._cat_tensors_to_optimizer(dict_clone, optimizer)

        # Return naturally added count for momentum tracking
        return added_count