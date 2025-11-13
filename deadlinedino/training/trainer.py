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
from PIL import Image, ImageDraw, ImageFont

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
from .scheduling_utils import ResolutionScheduler

def __l1_loss(network_output:torch.Tensor, gt:torch.Tensor)->torch.Tensor:
    return torch.abs((network_output - gt)).mean()

def __save_debug_view(debug_dir, iteration, render_img, gt_img, view_id, image_resolution,
                     downsampled_render_img=None, downsampled_gt_img=None, resolution_info=None):
    """Save debug visualization with render and GT, plus optional downsampled versions"""
    # Convert tensors to numpy arrays (CHW -> HWC, 0-1 range)
    render_np = render_img.detach().cpu().permute(1, 2, 0).numpy()
    gt_np = gt_img.detach().cpu().permute(1, 2, 0).numpy()

    # Clip and convert to uint8
    render_np = np.clip(render_np * 255, 0, 255).astype(np.uint8)
    gt_np = np.clip(gt_np * 255, 0, 255).astype(np.uint8)

    # Create PIL images
    render_pil = Image.fromarray(render_np)
    gt_pil = Image.fromarray(gt_np)

    # Process downsampled images if provided
    if downsampled_render_img is not None and downsampled_gt_img is not None:
        downsampled_render_np = downsampled_render_img.detach().cpu().permute(1, 2, 0).numpy()
        downsampled_gt_np = downsampled_gt_img.detach().cpu().permute(1, 2, 0).numpy()

        downsampled_render_np = np.clip(downsampled_render_np * 255, 0, 255).astype(np.uint8)
        downsampled_gt_np = np.clip(downsampled_gt_np * 255, 0, 255).astype(np.uint8)

        # Resize downsampled images to match full resolution for comparison
        downsampled_render_pil = Image.fromarray(downsampled_render_np).resize(
            (render_pil.width, render_pil.height), Image.NEAREST
        )
        downsampled_gt_pil = Image.fromarray(downsampled_gt_np).resize(
            (gt_pil.width, gt_pil.height), Image.NEAREST
        )

        # Create 2x2 grid layout
        h, w = render_np.shape[:2]
        label_height = 30

        # Create combined image with labels (2x2 grid)
        combined_width = w * 2
        combined_height = (h + label_height) * 2
        combined = Image.new('RGB', (combined_width, combined_height), color=(255, 255, 255))

        # Paste images in 2x2 grid
        # Top row: Full resolution render and GT
        combined.paste(render_pil, (0, label_height))
        combined.paste(gt_pil, (w, label_height))
        # Bottom row: Downsampled render and GT
        combined.paste(downsampled_render_pil, (0, h + label_height * 2))
        combined.paste(downsampled_gt_pil, (w, h + label_height * 2))

        # Add labels
        draw = ImageDraw.Draw(combined)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()

        # Top row labels
        render_label = f"Render (Iter: {iteration})"
        gt_label = f"GT (View: {view_id}, Res: {image_resolution})"
        draw.text((10, 5), render_label, fill=(0, 0, 0), font=font)
        draw.text((w + 10, 5), gt_label, fill=(0, 0, 0), font=font)

        # Bottom row labels
        if resolution_info:
            stage = resolution_info.get('stage', 1)
            scale = resolution_info.get('scale', 1.0)
            elapsed = resolution_info.get('elapsed_time', 0.0)
            ds_render_res = f"{downsampled_render_np.shape[1]}x{downsampled_render_np.shape[0]}"
            ds_gt_res = f"{downsampled_gt_np.shape[1]}x{downsampled_gt_np.shape[0]}"
            downsampled_render_label = f"Downsampled Render (Stage {stage}, {scale:.2%}, {ds_render_res})"
            downsampled_gt_label = f"Downsampled GT ({ds_gt_res}, Time: {elapsed:.1f}s)"
        else:
            downsampled_render_label = "Downsampled Render"
            downsampled_gt_label = "Downsampled GT"

        draw.text((10, h + label_height + 5), downsampled_render_label, fill=(0, 0, 0), font=font)
        draw.text((w + 10, h + label_height + 5), downsampled_gt_label, fill=(0, 0, 0), font=font)

    else:
        # Original 1x2 layout if no downsampled images
        h, w = render_np.shape[:2]
        label_height = 30

        combined_width = w * 2
        combined_height = h + label_height
        combined = Image.new('RGB', (combined_width, combined_height), color=(255, 255, 255))

        combined.paste(render_pil, (0, label_height))
        combined.paste(gt_pil, (w, label_height))

        draw = ImageDraw.Draw(combined)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()

        render_label = f"Render (Iter: {iteration})"
        gt_label = f"GT (View: {view_id}, Res: {image_resolution})"

        draw.text((10, 5), render_label, fill=(0, 0, 0), font=font)
        draw.text((w + 10, 5), gt_label, fill=(0, 0, 0), font=font)

    # Save image
    save_path = os.path.join(debug_dir, f"debug_iter_{iteration:06d}.png")
    combined.save(save_path)

def start(lp:arguments.ModelParams,op:arguments.OptimizationParams,pp:arguments.PipelineParams,dp:arguments.DensifyParams,
          test_epochs=[],save_ply=[],save_checkpoint=[],start_checkpoint:str=None):
    
    cameras_info:dict[int,data.CameraInfo]=None
    camera_frames:list[data.ImageFrame]=None
    if lp.source_type=="colmap":
        cameras_info,camera_frames,init_xyz,init_color=io_manager.load_colmap_result(lp.source_path,lp.images)#lp.sh_degree,lp.resolution
    elif lp.source_type=="slam":
        cameras_info,camera_frames,init_xyz,init_color=io_manager.load_slam_result(lp.source_path)#lp.sh_degree,lp.resolution

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

    # Debug mode initialization
    debug_dir = None
    debug_view_data = None
    last_debug_save_time = None
    debug_iteration = 0
    resolution_scheduler = None
    if pp.debug:
        debug_dir = os.path.join(lp.model_path, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        print(f"Debug mode enabled. Debug visualizations will be saved to: {debug_dir}")

        # Select a fixed view for debugging (use the first training view)
        debug_view_idx = 0
        debug_frame = training_frames[debug_view_idx]
        debug_camera = cameras_info[debug_frame.camera_id]

        # Precompute debug view data
        debug_dataset = CameraFrameDataset(cameras_info, [debug_frame], lp.resolution, pp.device_preload)
        debug_loader = DataLoader(debug_dataset, batch_size=1, shuffle=False, pin_memory=not pp.device_preload)

        # Get the view data
        for view_matrix, proj_matrix, frustumplane, gt_image, idx in debug_loader:
            debug_view_data = {
                'view_matrix': view_matrix.cuda(),
                'proj_matrix': proj_matrix.cuda(),
                'frustumplane': frustumplane.cuda(),
                'gt_image': gt_image.cuda() / 255.0,
                'view_id': debug_view_idx,
                'resolution': f"{gt_image.shape[3]}x{gt_image.shape[2]}",
                'camera': debug_camera
            }
            break

        # Initialize resolution scheduler
        resolution_scheduler = ResolutionScheduler(num_stages=6, stage_duration=9.0)
        print(f"Resolution scheduler initialized: {resolution_scheduler.num_stages} stages, {resolution_scheduler.stage_duration}s per stage")

        last_debug_save_time = time.time()

    # Time-based stopping: track start time for 59.5 second timeout
    training_start_time = time.time()

    # Start resolution scheduler if in debug mode
    if pp.debug and resolution_scheduler is not None:
        resolution_scheduler.start()

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
                view_matrix=view_matrix.cuda()
                proj_matrix=proj_matrix.cuda()
                frustumplane=frustumplane.cuda()
                gt_image=gt_image.cuda()/255.0
                if op.learnable_viewproj:
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
                    proj_matrix[:,0,0]=focal_x
                    proj_matrix[:,1,1]=focal_y

                #cluster culling
                visible_chunkid,culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity=render.render_preprocess(cluster_origin,cluster_extend,frustumplane,
                                                                                                               xyz,scale,rot,sh_0,sh_rest,opacity,op,pp)
                img,transmitance,depth,normal,primitive_visible=render.render(view_matrix,proj_matrix,culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity,
                                                            actived_sh_degree,gt_image.shape[2:],pp)
                
                l1_loss=__l1_loss(img,gt_image)
                ssim_loss:torch.Tensor=1-fused_ssim.fused_ssim(img,gt_image)
                loss=(1.0-op.lambda_dssim)*l1_loss+op.lambda_dssim*ssim_loss
                loss+=(culled_scale).square().mean()*op.reg_weight
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

                # Debug visualization: save every 5 seconds
                if pp.debug and debug_view_data is not None and resolution_scheduler is not None:
                    current_time = time.time()
                    if current_time - last_debug_save_time >= 5.0:
                        with torch.no_grad():
                            # Get cluster information for rendering
                            _cluster_origin = None
                            _cluster_extend = None
                            if pp.cluster_size:
                                _cluster_origin, _cluster_extend = scene.cluster.get_cluster_AABB(
                                    xyz, scale.exp(), torch.nn.functional.normalize(rot, dim=0)
                                )

                            # Render at full resolution
                            _, culled_xyz, culled_scale, culled_rot, culled_sh_0, culled_sh_rest, culled_opacity = \
                                render.render_preprocess(
                                    _cluster_origin, _cluster_extend, debug_view_data['frustumplane'],
                                    xyz, scale, rot, sh_0, sh_rest, opacity, op, pp
                                )

                            debug_render, _, _, _, _ = render.render(
                                debug_view_data['view_matrix'], debug_view_data['proj_matrix'],
                                culled_xyz, culled_scale, culled_rot, culled_sh_0, culled_sh_rest, culled_opacity,
                                actived_sh_degree, debug_view_data['gt_image'].shape[2:], pp
                            )

                            # Get downsampled resolution from scheduler
                            resolution_info = resolution_scheduler.get_info_dict()
                            full_height = debug_view_data['gt_image'].shape[2]
                            full_width = debug_view_data['gt_image'].shape[3]
                            ds_height, ds_width = resolution_scheduler.get_downsampled_shape(full_height, full_width)

                            # Create downsampled projection matrix
                            proj_matrix_np = debug_view_data['proj_matrix'].cpu().numpy()[0]
                            downsampled_proj_matrix_np = resolution_scheduler.get_downsampled_proj_matrix(
                                proj_matrix_np, full_height, full_width
                            )
                            downsampled_proj_matrix = torch.tensor(downsampled_proj_matrix_np, dtype=torch.float32, device='cuda').unsqueeze(0)

                            # Create frustum plane for downsampled resolution
                            view_matrix_np = debug_view_data['view_matrix'].cpu().numpy()[0]
                            viewproj_matrix = view_matrix_np @ downsampled_proj_matrix_np
                            downsampled_frustumplane = np.zeros((6, 4), dtype=np.float32)
                            # Left plane
                            downsampled_frustumplane[0] = [viewproj_matrix[0, 3] + viewproj_matrix[0, 0],
                                                           viewproj_matrix[1, 3] + viewproj_matrix[1, 0],
                                                           viewproj_matrix[2, 3] + viewproj_matrix[2, 0],
                                                           viewproj_matrix[3, 3] + viewproj_matrix[3, 0]]
                            # Right plane
                            downsampled_frustumplane[1] = [viewproj_matrix[0, 3] - viewproj_matrix[0, 0],
                                                           viewproj_matrix[1, 3] - viewproj_matrix[1, 0],
                                                           viewproj_matrix[2, 3] - viewproj_matrix[2, 0],
                                                           viewproj_matrix[3, 3] - viewproj_matrix[3, 0]]
                            # Bottom plane
                            downsampled_frustumplane[2] = [viewproj_matrix[0, 3] + viewproj_matrix[0, 1],
                                                           viewproj_matrix[1, 3] + viewproj_matrix[1, 1],
                                                           viewproj_matrix[2, 3] + viewproj_matrix[2, 1],
                                                           viewproj_matrix[3, 3] + viewproj_matrix[3, 1]]
                            # Top plane
                            downsampled_frustumplane[3] = [viewproj_matrix[0, 3] - viewproj_matrix[0, 1],
                                                           viewproj_matrix[1, 3] - viewproj_matrix[1, 1],
                                                           viewproj_matrix[2, 3] - viewproj_matrix[2, 1],
                                                           viewproj_matrix[3, 3] - viewproj_matrix[3, 1]]
                            # Near plane
                            downsampled_frustumplane[4] = [viewproj_matrix[0, 2],
                                                           viewproj_matrix[1, 2],
                                                           viewproj_matrix[2, 2],
                                                           viewproj_matrix[3, 2]]
                            # Far plane
                            downsampled_frustumplane[5] = [viewproj_matrix[0, 3] - viewproj_matrix[0, 2],
                                                           viewproj_matrix[1, 3] - viewproj_matrix[1, 2],
                                                           viewproj_matrix[2, 3] - viewproj_matrix[2, 2],
                                                           viewproj_matrix[3, 3] - viewproj_matrix[3, 2]]
                            downsampled_frustumplane = torch.tensor(downsampled_frustumplane, dtype=torch.float32, device='cuda').unsqueeze(0)

                            # Render at downsampled resolution
                            _, ds_culled_xyz, ds_culled_scale, ds_culled_rot, ds_culled_sh_0, ds_culled_sh_rest, ds_culled_opacity = \
                                render.render_preprocess(
                                    _cluster_origin, _cluster_extend, downsampled_frustumplane,
                                    xyz, scale, rot, sh_0, sh_rest, opacity, op, pp
                                )

                            downsampled_render, _, _, _, _ = render.render(
                                debug_view_data['view_matrix'], downsampled_proj_matrix,
                                ds_culled_xyz, ds_culled_scale, ds_culled_rot, ds_culled_sh_0, ds_culled_sh_rest, ds_culled_opacity,
                                actived_sh_degree, (ds_height, ds_width), pp
                            )

                            # Downsample the GT image using high-quality method
                            downsampled_gt = resolution_scheduler.downsample_image_hq(
                                debug_view_data['gt_image'],
                                target_height=ds_height,
                                target_width=ds_width,
                                use_lanczos=True
                            )

                            # Save the debug visualization
                            __save_debug_view(
                                debug_dir, debug_iteration,
                                debug_render[0], debug_view_data['gt_image'][0],
                                debug_view_data['view_id'], debug_view_data['resolution'],
                                downsampled_render[0], downsampled_gt[0],
                                resolution_info
                            )

                            debug_iteration += 1
                            last_debug_save_time = current_time

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

        xyz,scale,rot,sh_0,sh_rest,opacity=density_controller.step(opt,epoch)
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