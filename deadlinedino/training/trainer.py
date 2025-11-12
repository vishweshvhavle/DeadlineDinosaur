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
from .schedule_utils import TrainingScheduler, resize_image_with_scale
from .. import debug_utils

def __l1_loss(network_output:torch.Tensor, gt:torch.Tensor)->torch.Tensor:
    return torch.abs((network_output - gt)).mean()

def start(lp:arguments.ModelParams,op:arguments.OptimizationParams,pp:arguments.PipelineParams,dp:arguments.DensifyParams,
          test_epochs=[],save_ply=[],save_checkpoint=[],start_checkpoint:str=None):

    # Initialize debugger (set enabled=True to enable debugging)
    debugger = debug_utils.init_debugger(lp.model_path, enabled=True)

    cameras_info:dict[int,data.CameraInfo]=None
    camera_frames:list[data.ImageFrame]=None
    if lp.source_type=="colmap":
        cameras_info,camera_frames,init_xyz,init_color=io_manager.load_colmap_result(lp.source_path,lp.images)#lp.sh_degree,lp.resolution
    elif lp.source_type=="slam":
        cameras_info,camera_frames,init_xyz,init_color=io_manager.load_slam_result(lp.source_path)#lp.sh_degree,lp.resolution

    # Debug: Log initial point cloud statistics
    debugger.log_pointcloud_init(init_xyz, init_color)

    #preload
    for camera_frame in camera_frames:
        camera_frame.load_image(lp.resolution)

    #Dataset
    if lp.eval:
        training_frames=[c for idx, c in enumerate(camera_frames) if idx % 8 != 0]
        test_frames=[c for idx, c in enumerate(camera_frames) if idx % 8 == 0]
    else:
        training_frames=camera_frames
        test_frames=None
    trainingset=CameraFrameDataset(cameras_info,training_frames,lp.resolution,pp.device_preload)
    train_loader = DataLoader(trainingset, batch_size=1,shuffle=True,pin_memory=not pp.device_preload)
    test_loader=None
    if lp.eval:
        testset=CameraFrameDataset(cameras_info,test_frames,lp.resolution,pp.device_preload)
        test_loader = DataLoader(testset, batch_size=1,shuffle=False,pin_memory=not pp.device_preload)
    norm_trans,norm_radius=trainingset.get_norm()

    #torch parameter
    cluster_origin=None
    cluster_extend=None
    init_points_num=init_xyz.shape[0]
    if start_checkpoint is None:
        init_xyz=torch.tensor(init_xyz,dtype=torch.float32,device='cuda')
        init_color=torch.tensor(init_color,dtype=torch.float32,device='cuda')
        xyz,scale,rot,sh_0,sh_rest,opacity=scene.create_gaussians(init_xyz,init_color,lp.sh_degree)

        # Debug: Log initialized Gaussian parameters
        debugger.log_gaussian_init(xyz, scale, rot, sh_0, sh_rest, opacity)

        if pp.cluster_size:
            xyz,scale,rot,sh_0,sh_rest,opacity=scene.cluster.cluster_points(pp.cluster_size,xyz,scale,rot,sh_0,sh_rest,opacity)
        xyz=torch.nn.Parameter(xyz)
        scale=torch.nn.Parameter(scale)
        rot=torch.nn.Parameter(rot)
        sh_0=torch.nn.Parameter(sh_0)
        sh_rest=torch.nn.Parameter(sh_rest)
        opacity=torch.nn.Parameter(opacity)
        opt,schedular=optimizer.get_optimizer(xyz,scale,rot,sh_0,sh_rest,opacity,norm_radius,op,pp)
        start_epoch=0
    else:
        xyz,scale,rot,sh_0,sh_rest,opacity,start_epoch,opt,schedular=io_manager.load_checkpoint(start_checkpoint)
        if pp.cluster_size:
            cluster_origin,cluster_extend=scene.cluster.get_cluster_AABB(xyz,scale.exp(),torch.nn.functional.normalize(rot,dim=0))
    actived_sh_degree=0

    #learnable view matrix
    if op.learnable_viewproj:
        view_params=[np.concatenate([frame.qvec,frame.tvec])[None,:] for frame in trainingset.frames]
        view_params=torch.tensor(np.concatenate(view_params),dtype=torch.float32,device='cuda')
        view_params=torch.nn.Embedding(view_params.shape[0],view_params.shape[1],_weight=view_params,sparse=True)
        camera_focal_params=torch.nn.Parameter(torch.tensor(trainingset.cameras[0].focal_x,dtype=torch.float32,device='cuda'))#todo fix multi cameras
        view_opt=torch.optim.SparseAdam(view_params.parameters(),lr=1e-4)
        proj_opt=torch.optim.Adam([camera_focal_params,],lr=1e-5)

    #init
    total_epoch=int(op.iterations/len(trainingset))
    if dp.densify_until<0:
        dp.densify_until=int(total_epoch*0.8/dp.opacity_reset_interval)*dp.opacity_reset_interval+1
    density_controller=densify.DensityControllerTamingGS(norm_radius,dp,pp.cluster_size>0,init_points_num)
    StatisticsHelperInst.reset(xyz.shape[-2],xyz.shape[-1],density_controller.is_densify_actived)
    progress_bar = tqdm(range(start_epoch, total_epoch), desc="Training progress")
    progress_bar.update(0)

    # Initialize progressive training scheduler (DashGaussian strategy)
    # Collect original images for significance calculation
    original_images = [frame.load_image(lp.resolution) for frame in trainingset.frames]
    training_scheduler = TrainingScheduler(
        op, dp,
        init_n_gaussian=xyz.shape[-2] if pp.cluster_size == 0 else (xyz.shape[-2] * xyz.shape[-1]),
        original_images=original_images,
        total_iterations=op.iterations
    )
    current_render_scale = training_scheduler.max_reso_scale  # Start with lowest resolution

    # Time-based stopping: track start time for 59.5 second timeout
    training_start_time = time.time()

    for epoch in range(start_epoch,total_epoch):
        # Check if training time has exceeded 59.5 seconds BEFORE starting the epoch
        # This ensures we don't overshoot by running another full epoch
        elapsed_time = time.time() - training_start_time
        if elapsed_time >= 59.5:
            progress_bar.close()
            print(f"Training stopped at {elapsed_time:.2f}s (target: 59.5s) at epoch {epoch}")

            # Save the most recent .ply file
            save_path = os.path.join(lp.model_path, "point_cloud", f"timeout_epoch_{epoch}")

            if pp.cluster_size:
                tensors = scene.cluster.uncluster(xyz, scale, rot, sh_0, sh_rest, opacity)
            else:
                tensors = xyz, scale, rot, sh_0, sh_rest, opacity
            param_nyp = []
            for tensor in tensors:
                param_nyp.append(tensor.detach().cpu().numpy())
            io_manager.save_ply(os.path.join(save_path, "point_cloud.ply"), *param_nyp)
            if op.learnable_viewproj:
                torch.save(list(view_params.parameters())+[camera_focal_params], os.path.join(save_path, "viewproj.pth"))

            print(f"Saved checkpoint to {save_path}")
            break

        with torch.no_grad():
            if pp.cluster_size>0 and (epoch-1)%dp.densification_interval==0:
                scene.spatial_refine(pp.cluster_size>0,opt,xyz)
                cluster_origin,cluster_extend=scene.cluster.get_cluster_AABB(xyz,scale.exp(),torch.nn.functional.normalize(rot,dim=0))
            if actived_sh_degree<lp.sh_degree:
                actived_sh_degree=min(int(epoch/5),lp.sh_degree)

        with StatisticsHelperInst.try_start(epoch):
            for view_matrix,proj_matrix,frustumplane,gt_image,idx in train_loader:
                # Update resolution scale based on current iteration
                current_iteration = epoch * len(train_loader) + idx.item()
                current_render_scale = training_scheduler.get_res_scale(current_iteration)

                # Update optimizer scheduler to delay xyz LR decay until max resolution (DashGaussian)
                if training_scheduler.max_resolution_reached and schedular.decay_from_iter == 0:
                    schedular.decay_from_iter = training_scheduler.max_resolution_iter
                    print(f"\n[DashGaussian] Max resolution reached at iteration {training_scheduler.max_resolution_iter}")
                    print(f"[DashGaussian] XYZ learning rate decay will start from this point")

                view_matrix=view_matrix.cuda()
                proj_matrix=proj_matrix.cuda()
                frustumplane=frustumplane.cuda()
                gt_image=gt_image.cuda()/255.0

                # Apply progressive resolution scaling (DashGaussian approach)
                # Render AND compute loss at downscaled resolution for gradient consistency
                if current_render_scale > 1:
                    gt_image = resize_image_with_scale(gt_image, current_render_scale)
                    # CRITICAL: Create a copy of projection matrix to avoid in-place modification
                    # of the shared tensor from the dataset. Otherwise, the projection matrix
                    # gets divided multiple times as views are reused across epochs.
                    proj_matrix = proj_matrix.clone()
                    # Adjust projection matrix focal lengths for the new resolution
                    # Focal length must scale proportionally with image dimensions
                    proj_matrix[:, 0, 0] = proj_matrix[:, 0, 0] / current_render_scale
                    proj_matrix[:, 1, 1] = proj_matrix[:, 1, 1] / current_render_scale

                if op.learnable_viewproj:
                    # Create copies to avoid modifying shared dataset tensors
                    view_matrix = view_matrix.clone()
                    if current_render_scale == 1:  # Already cloned above if > 1
                        proj_matrix = proj_matrix.clone()

                    #fix view matrix
                    view_param_vec=view_params(idx.cuda())
                    qvec=torch.nn.functional.normalize(view_param_vec[:,:4],dim=1)
                    tvec=view_param_vec[:,4:]
                    rot_matrix=utils.wrapper.CreateTransformMatrix.call_fused(torch.ones((3,qvec.shape[0]),device='cuda'),qvec.transpose(0,1).contiguous())
                    view_matrix[:,:3, :3] = rot_matrix.permute(2,0,1)
                    view_matrix[:,3, :3] = tvec

                    #fix proj matrix
                    focal_x=camera_focal_params
                    focal_y=camera_focal_params*gt_image.shape[3]/gt_image.shape[2]
                    # Scale focal lengths for progressive resolution
                    if current_render_scale > 1:
                        focal_x = focal_x / current_render_scale
                        focal_y = focal_y / current_render_scale
                    proj_matrix[:,0,0]=focal_x
                    proj_matrix[:,1,1]=focal_y

                #cluster culling
                visible_chunkid,culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity=render.render_preprocess(cluster_origin,cluster_extend,frustumplane,
                                                                                                               xyz,scale,rot,sh_0,sh_rest,opacity,op,pp)
                img,transmitance,depth,normal,primitive_visible=render.render(view_matrix,proj_matrix,culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity,
                                                            actived_sh_degree,gt_image.shape[2:],pp)

                # Debug: Check rendered output
                debugger.check_render_output(img, current_iteration)

                # CRITICAL: Compute loss at DOWNSCALED resolution (DashGaussian approach)
                # This ensures gradients match the rendering resolution, avoiding gradient scaling mismatches
                # that cause NaN opacities during resolution transitions.
                # Unlike upscaling approaches, this keeps gradient flow consistent with render resolution.
                l1_loss=__l1_loss(img,gt_image)
                ssim_loss:torch.Tensor=1-fused_ssim.fused_ssim(img,gt_image)
                loss=(1.0-op.lambda_dssim)*l1_loss+op.lambda_dssim*ssim_loss
                loss+=(culled_scale).square().mean()*op.reg_weight

                # Debug: Log training iteration statistics
                debugger.log_training_iteration(current_iteration, xyz, scale, rot, sh_0, sh_rest, opacity,
                                              img, gt_image, loss, l1_loss, ssim_loss)

                # Debug: Save comparison images
                debugger.save_comparison_image(current_iteration, img, gt_image, prefix="train")

                loss.backward()
                if StatisticsHelperInst.bStart:
                    StatisticsHelperInst.backward_callback()
                if pp.sparse_grad:
                    opt.step(visible_chunkid,primitive_visible)
                else:
                    opt.step()
                opt.zero_grad(set_to_none = True)
                if op.learnable_viewproj:
                    view_opt.step()
                    view_opt.zero_grad()
                    # proj_opt.step()
                    # proj_opt.zero_grad()
                schedular.step()

        if epoch in test_epochs:
            with torch.no_grad():
                _cluster_origin=None
                _cluster_extend=None
                if pp.cluster_size:
                    _cluster_origin,_cluster_extend=scene.cluster.get_cluster_AABB(xyz,scale.exp(),torch.nn.functional.normalize(rot,dim=0))
                psnr_metrics=psnr.PeakSignalNoiseRatio(data_range=(0.0,1.0)).cuda()
                loaders={"Trainingset":train_loader}
                if lp.eval:
                    loaders["Testset"]=test_loader
                for name,loader in loaders.items():
                    psnr_list=[]
                    for view_matrix,proj_matrix,frustumplane,gt_image,idx in loader:
                        view_matrix=view_matrix.cuda()
                        proj_matrix=proj_matrix.cuda()
                        frustumplane=frustumplane.cuda()
                        gt_image=gt_image.cuda()/255.0

                        if name=="Trainingset" and op.learnable_viewproj:
                            # Create copies to avoid modifying shared dataset tensors
                            view_matrix = view_matrix.clone()
                            proj_matrix = proj_matrix.clone()

                            view_param_vec=view_params(idx.cuda())
                            qvec=torch.nn.functional.normalize(view_param_vec[:,:4],dim=1)
                            tvec=view_param_vec[:,4:]
                            rot_matrix=utils.wrapper.CreateTransformMatrix.call_fused(torch.ones((3,qvec.shape[0]),device='cuda'),qvec.transpose(0,1).contiguous())
                            view_matrix[:,:3, :3] = rot_matrix.permute(2,0,1)
                            view_matrix[:,3, :3] = tvec

                            focal_x=camera_focal_params
                            focal_y=camera_focal_params*gt_image.shape[3]/gt_image.shape[2]
                            proj_matrix[:,0,0]=focal_x
                            proj_matrix[:,1,1]=focal_y

                        _,culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity=render.render_preprocess(_cluster_origin,_cluster_extend,frustumplane,
                                                                                                                xyz,scale,rot,sh_0,sh_rest,opacity,op,pp)
                        img,transmitance,depth,normal,_=render.render(view_matrix,proj_matrix,culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity,
                                                                    actived_sh_degree,gt_image.shape[2:],pp)
                        psnr_list.append(psnr_metrics(img,gt_image).unsqueeze(0))
                    tqdm.write("\n[EPOCH {}] {} Evaluating: PSNR {}".format(epoch,name,torch.concat(psnr_list,dim=0).mean()))

        # Pass scheduler information to density controller for progressive densification
        current_iter = epoch * len(train_loader)
        current_n_primitives = xyz.shape[-2] if pp.cluster_size == 0 else (xyz.shape[-2] * xyz.shape[-1])
        xyz,scale,rot,sh_0,sh_rest,opacity=density_controller.step(
            opt, epoch,
            scheduler=training_scheduler,
            current_iteration=current_iter,
            current_n_primitives=current_n_primitives,
            current_render_scale=current_render_scale
        )
        progress_bar.update()

        if epoch in save_ply or epoch==total_epoch-1:
            if epoch==total_epoch-1:
                progress_bar.close()
                print("{} takes: {} s".format(lp.model_path,progress_bar.format_dict['elapsed']))
                save_path=os.path.join(lp.model_path,"point_cloud","finish")
            else:
                save_path=os.path.join(lp.model_path,"point_cloud","iteration_{}".format(epoch))    

            if pp.cluster_size:
                tensors=scene.cluster.uncluster(xyz,scale,rot,sh_0,sh_rest,opacity)
            else:
                tensors=xyz,scale,rot,sh_0,sh_rest,opacity
            param_nyp=[]
            for tensor in tensors:
                param_nyp.append(tensor.detach().cpu().numpy())
            io_manager.save_ply(os.path.join(save_path,"point_cloud.ply"),*param_nyp)
            if op.learnable_viewproj:
                torch.save(list(view_params.parameters())+[camera_focal_params],os.path.join(save_path,"viewproj.pth"))
            pass

        if epoch in save_checkpoint:
            io_manager.save_checkpoint(lp.model_path,epoch,opt,schedular)
    
    return