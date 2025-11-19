from argparse import ArgumentParser, Namespace
import torch
from torch.utils.data import DataLoader
from torchmetrics.image import psnr,ssim,lpip
import sys
import os
import json
import matplotlib.pyplot as plt
import torchvision
from tqdm import tqdm

import deadlinedino
import deadlinedino.config
import deadlinedino.utils
import shutil

OUTPUT_FILE=False

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp_cdo,op_cdo,pp_cdo,dp_cdo=deadlinedino.config.get_default_arg()
    deadlinedino.arguments.ModelParams.add_cmdline_arg(lp_cdo,parser)
    deadlinedino.arguments.OptimizationParams.add_cmdline_arg(op_cdo,parser)
    deadlinedino.arguments.PipelineParams.add_cmdline_arg(pp_cdo,parser)
    deadlinedino.arguments.DensifyParams.add_cmdline_arg(dp_cdo,parser)
    
    parser.add_argument("--test_epochs", nargs="+", type=int, default=[])
    parser.add_argument("--save_epochs", nargs="+", type=int, default=[])
    parser.add_argument("--checkpoint_epochs", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    
    lp=deadlinedino.arguments.ModelParams.extract(args)
    op=deadlinedino.arguments.OptimizationParams.extract(args)
    pp=deadlinedino.arguments.PipelineParams.extract(args)
    dp=deadlinedino.arguments.DensifyParams.extract(args)

    cameras_info:dict[int,deadlinedino.data.CameraInfo]=None
    camera_frames:list[deadlinedino.data.ImageFrame]=None
    if lp.source_type=="colmap":
        cameras_info,camera_frames,init_xyz,init_color=deadlinedino.io_manager.load_colmap_result(lp.source_path,lp.images)#lp.sh_degree,lp.resolution
    elif lp.source_type=="slam":
        cameras_info,camera_frames,init_xyz,init_color=deadlinedino.io_manager.load_slam_result(lp.source_path)#lp.sh_degree,lp.resolution

    if OUTPUT_FILE:
        renders_dir = os.path.join(lp.model_path, "renders")
        try:
            shutil.rmtree(renders_dir)
        except:
            pass
        os.makedirs(renders_dir, exist_ok=True)

    #preload
    for camera_frame in camera_frames:
        camera_frame.load_image(lp.resolution)

    #Dataset
    # Load train/test split from train_test_split.json
    split_json_path = os.path.join(lp.source_path, "train_test_split.json")
    if os.path.exists(split_json_path):
        with open(split_json_path, 'r') as f:
            split_data = json.load(f)
        train_image_names = set(split_data.get("train", []))
        test_image_names = set(split_data.get("test", []))

        # Filter frames based on the split
        if lp.eval:
            training_frames = [c for c in camera_frames if c.name in train_image_names]
            test_frames = [c for c in camera_frames if c.name in test_image_names]
            trainingset=deadlinedino.data.CameraFrameDataset(cameras_info,training_frames,lp.resolution,pp.device_preload)
            train_loader = DataLoader(trainingset, batch_size=1,shuffle=False,pin_memory=not pp.device_preload)
            testset=deadlinedino.data.CameraFrameDataset(cameras_info,test_frames,lp.resolution,pp.device_preload)
            test_loader = DataLoader(testset, batch_size=1,shuffle=False,pin_memory=not pp.device_preload)
            print(f"Loaded train/test split: {len(training_frames)} training images, {len(test_frames)} test images")
        else:
            # For evaluation only, use test images
            test_frames = [c for c in camera_frames if c.name in test_image_names]
            trainingset=deadlinedino.data.CameraFrameDataset(cameras_info,test_frames,lp.resolution,pp.device_preload)
            train_loader = DataLoader(trainingset, batch_size=1,shuffle=False,pin_memory=not pp.device_preload)
            print(f"Loaded test split: {len(test_frames)} test images")
    else:
        # Fallback to original logic if train_test_split.json doesn't exist
        if lp.eval:
            training_frames=[c for idx, c in enumerate(camera_frames) if idx % 8 != 0]
            test_frames=[c for idx, c in enumerate(camera_frames) if idx % 8 == 0]
            trainingset=deadlinedino.data.CameraFrameDataset(cameras_info,training_frames,lp.resolution,pp.device_preload)
            train_loader = DataLoader(trainingset, batch_size=1,shuffle=False,pin_memory=not pp.device_preload)
            testset=deadlinedino.data.CameraFrameDataset(cameras_info,test_frames,lp.resolution,pp.device_preload)
            test_loader = DataLoader(testset, batch_size=1,shuffle=False,pin_memory=not pp.device_preload)
        else:
            trainingset=deadlinedino.data.CameraFrameDataset(cameras_info,camera_frames,lp.resolution,pp.device_preload)
            train_loader = DataLoader(trainingset, batch_size=1,shuffle=False,pin_memory=not pp.device_preload)
    norm_trans,norm_radius=trainingset.get_norm()

    #model
    # Check for timeout checkpoint first, otherwise use finish directory
    point_cloud_dir = os.path.join(lp.model_path, "point_cloud")
    ply_path = None

    # Look for timeout_epoch directories
    if os.path.exists(point_cloud_dir):
        timeout_dirs = [d for d in os.listdir(point_cloud_dir) if d.startswith("timeout_epoch_")]
        if timeout_dirs:
            # Use the timeout checkpoint with the highest epoch number
            timeout_dirs.sort(key=lambda x: int(x.split("_")[-1]), reverse=True)
            ply_path = os.path.join(point_cloud_dir, timeout_dirs[0], "point_cloud.ply")

    # Fall back to finish directory if no timeout checkpoint found
    if ply_path is None or not os.path.exists(ply_path):
        finish_path = os.path.join(point_cloud_dir, "finish", "point_cloud.ply")
        if os.path.exists(finish_path):
            ply_path = finish_path
        else:
            print(f"Error: Could not find point cloud file in {point_cloud_dir}")
            print(f"Checked for timeout_epoch_* directories and finish/point_cloud.ply")
            sys.exit(1)

    xyz,scale,rot,sh_0,sh_rest,opacity=deadlinedino.io_manager.load_ply(ply_path,lp.sh_degree)
    xyz=torch.Tensor(xyz).cuda()
    scale=torch.Tensor(scale).cuda()
    rot=torch.Tensor(rot).cuda()
    sh_0=torch.Tensor(sh_0).cuda()
    sh_rest=torch.Tensor(sh_rest).cuda()
    opacity=torch.Tensor(opacity).cuda()
    cluster_origin=None
    cluster_extend=None
    if pp.cluster_size>0:
        xyz,scale,rot,sh_0,sh_rest,opacity=deadlinedino.scene.point.spatial_refine(False,None,xyz,scale,rot,sh_0,sh_rest,opacity)
        xyz,scale,rot,sh_0,sh_rest,opacity=deadlinedino.scene.cluster.cluster_points(pp.cluster_size,xyz,scale,rot,sh_0,sh_rest,opacity)
        cluster_origin,cluster_extend=deadlinedino.scene.cluster.get_cluster_AABB(xyz,scale.exp(),torch.nn.functional.normalize(rot,dim=0))
    if op.learnable_viewproj:
        # Use the same directory as the ply file for viewproj.pth
        viewproj_path = os.path.join(os.path.dirname(ply_path), "viewproj.pth")
        view_params,proj_parmas=torch.load(viewproj_path)
        qvec=torch.nn.functional.normalize(view_params[:,:4],dim=1)
        rot_matrix=deadlinedino.utils.wrapper.CreateTransformMatrix.call_fused(torch.ones((3,qvec.shape[0]),device='cuda'),qvec.transpose(0,1).contiguous()).permute(2,0,1)
        tvec=view_params[:,4:]

    #metrics
    psnr_metrics=psnr.PeakSignalNoiseRatio(data_range=1.0).cuda()

    #iter
    with torch.no_grad():
        psnr_list=[]

        # Create progress bar
        pbar = tqdm(train_loader, desc=f"Processing images", unit="img")

        for index,(view_matrix,proj_matrix,frustumplane,gt_image,idx) in enumerate(pbar):
            view_matrix=view_matrix.cuda()
            proj_matrix=proj_matrix.cuda()
            frustumplane=frustumplane.cuda()
            gt_image=gt_image.cuda()/255.0
            if op.learnable_viewproj:
                #fix view matrix
                view_matrix[:,:3, :3] = rot_matrix[idx:idx+1]
                view_matrix[:,3, :3] = tvec[idx:idx+1]

                #fix proj matrix
                focal_x=proj_parmas
                focal_y=proj_parmas*gt_image.shape[3]/gt_image.shape[2]
                proj_matrix[:,0,0]=focal_x
                proj_matrix[:,1,1]=focal_y


            _,culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity=deadlinedino.render.render_preprocess(cluster_origin,cluster_extend,frustumplane,
                                                                                                    xyz,scale,rot,sh_0,sh_rest,opacity,op,pp)
            img,transmitance,depth,normal,_=deadlinedino.render.render(view_matrix,proj_matrix,culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity,
                                                        lp.sh_degree,gt_image.shape[2:],pp)
            psnr_value=psnr_metrics(img,gt_image)
            psnr_list.append(psnr_value.unsqueeze(0))

            # Update progress bar with current PSNR
            pbar.set_postfix({
                'PSNR': f'{psnr_value.item():.2f}',
            })

            if OUTPUT_FILE:

                torchvision.utils.save_image(
                    img[0],  # Remove batch dimension
                    os.path.join(lp.model_path, "renders", f"{index}-{float(psnr_value):.2f}-render.png")
                )

                torchvision.utils.save_image(
                    gt_image[0],  # Remove batch dimension
                    os.path.join(lp.model_path, "renders", f"{index}-{float(psnr_value):.2f}-gt.png")
                )

        pbar.close()

        psnr_mean=torch.concat(psnr_list,dim=0).mean()

        print("  Scene:{0}".format(lp.model_path))
        print("  PSNR : {:>12.7f}".format(float(psnr_mean)))
        print("")
