from argparse import ArgumentParser, Namespace
import torch
import sys

import deadlinedino
import deadlinedino.config
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


    deadlinedino.training.start(lp,op,pp,dp,args.test_epochs,args.save_epochs,args.checkpoint_epochs,args.start_checkpoint)