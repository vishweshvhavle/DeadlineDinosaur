import torch
from torch.utils.data import DataLoader
import fused_ssim
from torchmetrics.image import psnr
from tqdm import tqdm
import numpy as np
import math
import os
import time
import torch.cuda.nvtx as nvtx
import matplotlib.pyplot as plt
import json

from .. import arguments
from .. import data
from .. import io_manager
from .. import scene
from . import optimizer
from ..data import CameraFrameDataset
from .. import render
from ..utils.statistic_helper import StatisticsHelperInst
from . import densify
from .. import utils
from . import schedule_utils 

def start(
    lp: arguments.ModelParams,
    op: arguments.OptimizationParams,
    pp: arguments.PipelineParams,
    dp: arguments.DensifyParams,
    test_epochs: list = [],
    save_ply: list = [],
    save_checkpoint: list = [],
    start_checkpoint: str = None
):
    """
    Main training loop for 3D Gaussian Splatting with coarse-to-fine resolution scheduling.
    
    Args:
        lp: Model parameters (source path, resolution, SH degree)
        op: Optimization parameters (iterations, learning rates, regularization)
        pp: Pipeline parameters (device settings, clustering, resolution mode)
        dp: Densification parameters (pruning, splitting intervals)
        test_epochs: List of epochs to run evaluation
        save_ply: List of epochs to save point cloud
        save_checkpoint: List of epochs to save full checkpoint
        start_checkpoint: Path to checkpoint for resuming training
    """
    
    # ========== 1. DATA LOADING ==========
    cameras_info, camera_frames, init_xyz, init_color = _load_scene_data(lp)
    
    # Preload all images to memory
    for frame in camera_frames:
        frame.load_image(lp.resolution)
    
    # ========== 2. TRAIN/TEST SPLIT ==========
    training_frames, test_frames = _create_train_test_split(
        lp.source_path, camera_frames, lp.eval
    )
    
    # Create data loaders
    trainingset = CameraFrameDataset(
        cameras_info, training_frames, lp.resolution, pp.device_preload
    )
    train_loader = DataLoader(
        trainingset, batch_size=1, shuffle=True, 
        pin_memory=not pp.device_preload
    )
    
    test_loader = None
    if lp.eval:
        testset = CameraFrameDataset(
            cameras_info, test_frames, lp.resolution, pp.device_preload
        )
        test_loader = DataLoader(
            testset, batch_size=1, shuffle=False, 
            pin_memory=not pp.device_preload
        )
    
    norm_trans, norm_radius = trainingset.get_norm()
    
    # ========== 3. SCHEDULER INITIALIZATION ==========
    # Prepare images for frequency analysis (if using freq-based scheduling)
    original_images_for_fft = []
    if pp.resolution_mode == "freq":
        for frame in training_frames:
            original_images_for_fft.append(frame.image[lp.resolution])
    
    # Initialize training scheduler for coarse-to-fine resolution scaling
    init_points_num = init_xyz.shape[0]
    scheduler = schedule_utils.TrainingScheduler(
        op, dp, pp, init_points_num, original_images_for_fft
    )
    
    # Free memory after scheduler init
    del original_images_for_fft
    torch.cuda.empty_cache()
    
    # Get dynamic LR decay iteration from scheduler
    decay_from_iter = scheduler.lr_decay_from_iter()
    
    # ========== 4. MODEL & OPTIMIZER INITIALIZATION ==========
    cluster_origin, cluster_extend = None, None
    
    if start_checkpoint is None:
        # Initialize from scratch
        xyz, scale, rot, sh_0, sh_rest, opacity = _initialize_gaussians(
            init_xyz, init_color, lp.sh_degree, pp.cluster_size
        )
        opt, schedular = optimizer.get_optimizer(
            xyz, scale, rot, sh_0, sh_rest, opacity,
            norm_radius, op, pp, decay_from_iter=decay_from_iter
        )
        start_epoch = 0
    else:
        # Resume from checkpoint
        xyz, scale, rot, sh_0, sh_rest, opacity, start_epoch, opt, schedular = \
            io_manager.load_checkpoint(start_checkpoint)
        if pp.cluster_size:
            cluster_origin, cluster_extend = _compute_cluster_aabb(
                xyz, scale, rot, pp.cluster_size
            )
    
    # ========== 5. ADDITIONAL COMPONENTS ==========
    actived_sh_degree = 0
    
    # Optional: Learnable camera parameters
    view_params, camera_focal_params, view_opt, proj_opt = None, None, None, None
    if op.learnable_viewproj:
        view_params, camera_focal_params, view_opt, proj_opt = \
            _initialize_learnable_cameras(trainingset)
    
    # Densification controller
    density_controller = densify.DensityControllerTamingGS(
        norm_radius, dp, pp.cluster_size > 0, init_points_num
    )
    
    # ========== 6. TRAINING SETUP ==========
    total_epoch = int(op.iterations / len(trainingset))
    global_step = start_epoch * len(train_loader)
    res_scale_buffer = []  # Track resolution scaling per step
    
    if dp.densify_until < 0:
        dp.densify_until = int(
            total_epoch * 0.8 / dp.opacity_reset_interval
        ) * dp.opacity_reset_interval + 1
    
    StatisticsHelperInst.reset(
        xyz.shape[-2], xyz.shape[-1], density_controller.is_densify_actived
    )
    
    progress_bar = tqdm(range(start_epoch, total_epoch), desc="Training progress")
    training_start_time = time.time()
    
    # ========== 7. MAIN TRAINING LOOP ==========
    for epoch in range(start_epoch, total_epoch):
        
        # Check timeout (60 second limit with buffer)
        elapsed_time = time.time() - training_start_time
        if _should_timeout(elapsed_time, epoch, start_epoch):
            _save_timeout_checkpoint(
                lp, pp, op, epoch, total_epoch, elapsed_time,
                xyz, scale, rot, sh_0, sh_rest, opacity,
                res_scale_buffer, view_params, camera_focal_params
            )
            break
        
        # Update SH degree progressively and cluster AABB
        with torch.no_grad():
            if pp.cluster_size > 0 and (epoch - 1) % dp.densification_interval == 0:
                scene.spatial_refine(pp.cluster_size > 0, opt, xyz)
                cluster_origin, cluster_extend = _compute_cluster_aabb(
                    xyz, scale, rot, pp.cluster_size
                )
            
            if actived_sh_degree < lp.sh_degree:
                actived_sh_degree = min(int(epoch / 5), lp.sh_degree)
        
        # Training iteration
        with StatisticsHelperInst.try_start(epoch):
            for view_matrix, proj_matrix, frustumplane, gt_image, idx in train_loader:
                
                # Move data to GPU
                view_matrix = view_matrix.cuda()
                proj_matrix = proj_matrix.cuda()
                frustumplane = frustumplane.cuda()
                gt_image = gt_image.cuda() / 255.0
                
                # === COARSE-TO-FINE RESOLUTION SCALING ===
                render_scale = scheduler.get_res_scale(global_step)
                res_scale_buffer.append({
                    "global_step": global_step,
                    "render_scale": float(render_scale)
                })
                
                # Downsample GT image if rendering at lower resolution
                if render_scale > 1:
                    gt_image = torch.nn.functional.interpolate(
                        gt_image,
                        scale_factor=1.0 / render_scale,
                        mode="bilinear",
                        recompute_scale_factor=True,
                        antialias=True
                    )
                
                # Update camera parameters if learnable
                if op.learnable_viewproj:
                    view_matrix, proj_matrix = _update_camera_matrices(
                        view_params, camera_focal_params, idx,
                        view_matrix, proj_matrix, gt_image.shape
                    )
                
                # Render current view
                visible_chunkid, culled_xyz, culled_scale, culled_rot, \
                culled_sh_0, culled_sh_rest, culled_opacity = render.render_preprocess(
                    cluster_origin, cluster_extend, frustumplane,
                    xyz, scale, rot, sh_0, sh_rest, opacity, op, pp
                )
                
                img, transmitance, depth, normal, primitive_visible = render.render(
                    view_matrix, proj_matrix, culled_xyz, culled_scale,
                    culled_rot, culled_sh_0, culled_sh_rest, culled_opacity,
                    actived_sh_degree, gt_image.shape[2:], pp
                )
                
                # Compute loss
                img_b = img.unsqueeze(0)  # Add batch dimension
                l1_loss = __l1_loss(img_b, gt_image)
                ssim_loss = 1 - fused_ssim.fused_ssim(img_b, gt_image)
                loss = (1.0 - op.lambda_dssim) * l1_loss + op.lambda_dssim * ssim_loss
                loss += (culled_scale).square().mean() * op.reg_weight
                
                # Backward pass
                loss.backward()
                if StatisticsHelperInst.bStart:
                    StatisticsHelperInst.backward_callback()
                
                # Optimizer step
                if pp.sparse_grad:
                    opt.step(visible_chunkid, primitive_visible)
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
                
                # Update learnable camera parameters
                if op.learnable_viewproj:
                    view_opt.step()
                    view_opt.zero_grad()
                
                schedular.step()
                global_step += 1
        
        # Evaluation
        if epoch in test_epochs:
            _run_evaluation(
                epoch, train_loader, test_loader, lp, op, pp,
                xyz, scale, rot, sh_0, sh_rest, opacity,
                actived_sh_degree, view_params, camera_focal_params
            )
        
        # Densification step
        xyz, scale, rot, sh_0, sh_rest, opacity = density_controller.step(opt, epoch)
        progress_bar.update()
        
        # Save checkpoints
        if epoch in save_ply or epoch == total_epoch - 1:
            elapsed_time = time.time() - training_start_time
            _save_model_checkpoint(
                lp, pp, op, epoch, total_epoch, elapsed_time,
                xyz, scale, rot, sh_0, sh_rest, opacity,
                res_scale_buffer, view_params, camera_focal_params,
                is_final=(epoch == total_epoch - 1)
            )
        
        if epoch in save_checkpoint:
            io_manager.save_checkpoint(lp.model_path, epoch, opt, schedular)
    
    progress_bar.close()
    return


# ========== HELPER FUNCTIONS ==========

def _load_scene_data(lp: arguments.ModelParams):
    """Load camera info and 3D points from COLMAP or SLAM."""
    if lp.source_type == "colmap":
        return io_manager.load_colmap_result(lp.source_path, lp.images)
    elif lp.source_type == "slam":
        return io_manager.load_slam_result(lp.source_path)
    else:
        raise ValueError(f"Unknown source type: {lp.source_type}")


def _create_train_test_split(source_path: str, camera_frames: list, eval_mode: bool):
    """Create train/test split from JSON or default 8-fold split."""
    split_json_path = os.path.join(source_path, "train_test_split.json")
    
    if os.path.exists(split_json_path):
        with open(split_json_path, 'r') as f:
            split_data = json.load(f)
        train_names = set(split_data.get("train", []))
        test_names = set(split_data.get("test", []))
        
        training_frames = [c for c in camera_frames if c.name in train_names]
        test_frames = [c for c in camera_frames if c.name in test_names] if eval_mode else None
        
        print(f"Loaded split: {len(training_frames)} train, "
              f"{len(test_frames) if test_frames else 0} test images")
    else:
        # Default: every 8th frame for testing
        if eval_mode:
            training_frames = [c for idx, c in enumerate(camera_frames) if idx % 8 != 0]
            test_frames = [c for idx, c in enumerate(camera_frames) if idx % 8 == 0]
        else:
            training_frames = camera_frames
            test_frames = None
    
    return training_frames, test_frames


def _initialize_gaussians(init_xyz, init_color, sh_degree, cluster_size):
    """Initialize 3D Gaussian parameters."""
    init_xyz = torch.tensor(init_xyz, dtype=torch.float32, device='cuda')
    init_color = torch.tensor(init_color, dtype=torch.float32, device='cuda')
    
    xyz, scale, rot, sh_0, sh_rest, opacity = scene.create_gaussians(
        init_xyz, init_color, sh_degree
    )
    
    # Optional clustering for memory efficiency
    if cluster_size:
        xyz, scale, rot, sh_0, sh_rest, opacity = scene.cluster.cluster_points(
            cluster_size, xyz, scale, rot, sh_0, sh_rest, opacity
        )
    
    # Convert to learnable parameters
    return (
        torch.nn.Parameter(xyz),
        torch.nn.Parameter(scale),
        torch.nn.Parameter(rot),
        torch.nn.Parameter(sh_0),
        torch.nn.Parameter(sh_rest),
        torch.nn.Parameter(opacity)
    )


def _compute_cluster_aabb(xyz, scale, rot, cluster_size):
    """Compute cluster axis-aligned bounding boxes."""
    if cluster_size > 0:
        return scene.cluster.get_cluster_AABB(
            xyz, scale.exp(), torch.nn.functional.normalize(rot, dim=0)
        )
    return None, None


def _initialize_learnable_cameras(trainingset):
    """Initialize learnable camera pose and intrinsic parameters."""
    # Camera poses (rotation + translation)
    view_params = [
        np.concatenate([frame.qvec, frame.tvec])[None, :]
        for frame in trainingset.frames
    ]
    view_params = torch.tensor(
        np.concatenate(view_params), dtype=torch.float32, device='cuda'
    )
    view_params = torch.nn.Embedding(
        view_params.shape[0], view_params.shape[1],
        _weight=view_params, sparse=True
    )
    
    # Camera focal length
    camera_focal_params = torch.nn.Parameter(
        torch.tensor(trainingset.cameras[0].focal_x, dtype=torch.float32, device='cuda')
    )
    
    # Optimizers
    view_opt = torch.optim.SparseAdam(view_params.parameters(), lr=1e-4)
    proj_opt = torch.optim.Adam([camera_focal_params], lr=1e-5)
    
    return view_params, camera_focal_params, view_opt, proj_opt


def _update_camera_matrices(view_params, camera_focal_params, idx, 
                            view_matrix, proj_matrix, gt_shape):
    """Update view and projection matrices with learnable parameters."""
    view_param_vec = view_params(idx.cuda())
    qvec = torch.nn.functional.normalize(view_param_vec[:, :4], dim=1)
    tvec = view_param_vec[:, 4:]
    
    rot_matrix = utils.wrapper.CreateTransformMatrix.call_fused(
        torch.ones((3, qvec.shape[0]), device='cuda'),
        qvec.transpose(0, 1).contiguous()
    )
    
    view_matrix[:, :3, :3] = rot_matrix.permute(2, 0, 1)
    view_matrix[:, 3, :3] = tvec
    
    # Update focal length in projection matrix
    focal_x = camera_focal_params
    focal_y = camera_focal_params * gt_shape[3] / gt_shape[2]
    proj_matrix[:, 0, 0] = focal_x
    proj_matrix[:, 1, 1] = focal_y
    
    return view_matrix, proj_matrix


def _should_timeout(elapsed_time: float, current_epoch: int, start_epoch: int) -> bool:
    """Check if training should timeout (60 second limit with buffer)."""
    if current_epoch == start_epoch:
        return False
    avg_epoch_time = elapsed_time / (current_epoch - start_epoch)
    return elapsed_time >= 60 - avg_epoch_time - 0.5


def _save_timeout_checkpoint(lp, pp, op, epoch, total_epoch, elapsed_time,
                             xyz, scale, rot, sh_0, sh_rest, opacity,
                             res_scale_buffer, view_params, camera_focal_params):
    """Save checkpoint when training times out."""
    save_path = os.path.join(lp.model_path, "point_cloud", f"timeout_epoch_{epoch}")
    os.makedirs(save_path, exist_ok=True)
    
    # Save metrics
    metrics = {
        "time": elapsed_time,
        "model_path": lp.model_path,
        "status": "timeout",
        "final_epoch": epoch,
        "total_epochs": total_epoch,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(os.path.join(save_path, "training_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Save resolution schedule
    with open(os.path.join(save_path, "res_scale.json"), 'w') as f:
        json.dump(res_scale_buffer, f, indent=4)
    
    # Save point cloud
    _save_point_cloud(save_path, pp, xyz, scale, rot, sh_0, sh_rest, opacity)
    
    # Save learnable camera parameters
    if op.learnable_viewproj:
        torch.save(
            list(view_params.parameters()) + [camera_focal_params],
            os.path.join(save_path, "viewproj.pth")
        )
    
    print(f"Timeout checkpoint saved to {save_path}")


def _save_model_checkpoint(lp, pp, op, epoch, total_epoch, elapsed_time,
                           xyz, scale, rot, sh_0, sh_rest, opacity,
                           res_scale_buffer, view_params, camera_focal_params,
                           is_final: bool):
    """Save model checkpoint at specified epochs."""
    if is_final:
        save_path = os.path.join(lp.model_path, "point_cloud", "finish")
        status = "completed"
    else:
        save_path = os.path.join(lp.model_path, "point_cloud", f"iteration_{epoch}")
        status = "checkpoint"
    
    os.makedirs(save_path, exist_ok=True)
    
    # Save metrics
    metrics = {
        "time": elapsed_time,
        "model_path": lp.model_path,
        "status": status,
        "epoch": epoch,
        "total_epochs": total_epoch,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(os.path.join(save_path, "training_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Save resolution schedule
    with open(os.path.join(save_path, "res_scale.json"), 'w') as f:
        json.dump(res_scale_buffer, f, indent=4)
    
    # Save point cloud
    _save_point_cloud(save_path, pp, xyz, scale, rot, sh_0, sh_rest, opacity)
    
    # Save learnable camera parameters
    if op.learnable_viewproj:
        torch.save(
            list(view_params.parameters()) + [camera_focal_params],
            os.path.join(save_path, "viewproj.pth")
        )
    
    if is_final:
        print(f"{lp.model_path} training completed in {elapsed_time:.2f}s")


def _save_point_cloud(save_path, pp, xyz, scale, rot, sh_0, sh_rest, opacity):
    """Save 3D Gaussian point cloud to PLY file."""
    if pp.cluster_size:
        tensors = scene.cluster.uncluster(xyz, scale, rot, sh_0, sh_rest, opacity)
    else:
        tensors = (xyz, scale, rot, sh_0, sh_rest, opacity)
    
    param_nyp = [tensor.detach().cpu().numpy() for tensor in tensors]
    io_manager.save_ply(os.path.join(save_path, "point_cloud.ply"), *param_nyp)


def _run_evaluation(epoch, train_loader, test_loader, lp, op, pp,
                   xyz, scale, rot, sh_0, sh_rest, opacity,
                   actived_sh_degree, view_params, camera_focal_params):
    """Run PSNR evaluation on train and test sets."""
    with torch.no_grad():
        _cluster_origin, _cluster_extend = _compute_cluster_aabb(
            xyz, scale, rot, pp.cluster_size
        )
        
        psnr_metrics = psnr.PeakSignalNoiseRatio(data_range=(0.0, 1.0)).cuda()
        
        loaders = {"Trainingset": train_loader}
        if lp.eval:
            loaders["Testset"] = test_loader
        
        for name, loader in loaders.items():
            psnr_list = []
            
            for view_matrix, proj_matrix, frustumplane, gt_image, idx in loader:
                view_matrix = view_matrix.cuda()
                proj_matrix = proj_matrix.cuda()
                frustumplane = frustumplane.cuda()
                gt_image = gt_image.cuda() / 255.0
                
                # Update camera matrices for training set
                if name == "Trainingset" and op.learnable_viewproj:
                    view_matrix, proj_matrix = _update_camera_matrices(
                        view_params, camera_focal_params, idx,
                        view_matrix, proj_matrix, gt_image.shape
                    )
                
                # Render
                _, culled_xyz, culled_scale, culled_rot, culled_sh_0, \
                culled_sh_rest, culled_opacity = render.render_preprocess(
                    _cluster_origin, _cluster_extend, frustumplane,
                    xyz, scale, rot, sh_0, sh_rest, opacity, op, pp
                )
                
                img, _, _, _, _ = render.render(
                    view_matrix, proj_matrix, culled_xyz, culled_scale,
                    culled_rot, culled_sh_0, culled_sh_rest, culled_opacity,
                    actived_sh_degree, gt_image.shape[2:], pp
                )
                
                psnr_list.append(psnr_metrics(img, gt_image).unsqueeze(0))
            
            avg_psnr = torch.concat(psnr_list, dim=0).mean()
            tqdm.write(f"\n[EPOCH {epoch}] {name} PSNR: {avg_psnr:.4f}")


def __l1_loss(network_output: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Compute L1 loss between network output and ground truth."""
    return torch.abs(network_output - gt).mean()