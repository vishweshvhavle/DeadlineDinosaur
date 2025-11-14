from argparse import ArgumentParser, Namespace
import torch
from torch.utils.data import DataLoader
from torchmetrics.image import psnr,ssim,lpip
import sys
import os
import matplotlib.pyplot as plt
import torchvision
from tqdm import tqdm

import deadlinedino
import deadlinedino.config
import deadlinedino.utils
import shutil

OUTPUT_FILE=True

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
        try:
            shutil.rmtree(os.path.join(lp.model_path,"Trainingset"))
            shutil.rmtree(os.path.join(lp.model_path,"Testset"))
        except:
            pass
        os.makedirs(os.path.join(lp.model_path,"Trainingset"),exist_ok=True)
        os.makedirs(os.path.join(lp.model_path,"Testset"),exist_ok=True)

    #preload
    for camera_frame in camera_frames:
        camera_frame.load_image(lp.resolution)

    #Dataset
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
        ply_path = os.path.join(point_cloud_dir, "finish", "point_cloud.ply")

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
    if lp.eval:
        loaders={"Trainingset":train_loader,"Testset":test_loader}
    else:
        loaders={"Trainingset":train_loader}

    with torch.no_grad():
        for loader_name,loader in loaders.items():
            psnr_list=[]

            # Create progress bar for this loader
            pbar = tqdm(loader, desc=f"Processing {loader_name}", unit="img")
            
            for index,(view_matrix,proj_matrix,frustumplane,gt_image,idx) in enumerate(pbar):
                view_matrix=view_matrix.cuda()
                proj_matrix=proj_matrix.cuda()
                frustumplane=frustumplane.cuda()
                gt_image=gt_image.cuda()/255.0
                if loader_name=="Trainingset" and op.learnable_viewproj:
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
                        os.path.join(lp.model_path, loader_name, f"{index}-{float(psnr_value):.2f}-rd.png")
                    )

                    torchvision.utils.save_image(
                        gt_image[0],  # Remove batch dimension
                        os.path.join(lp.model_path, loader_name, f"{index}-{float(psnr_value):.2f}-gt.png")
                    )

            pbar.close()

            psnr_mean=torch.concat(psnr_list,dim=0).mean()

            print("  Scene:{0}".format(lp.model_path+" "+loader_name))
            print("  PSNR : {:>12.7f}".format(float(psnr_mean)))
            print("")
