
"""
select config.yaml  by model
"""

from yacs.config import CfgNode as CN

def config_path(args):
    if args.model == "IMDN":
        path = "/home/ubuntu/syb/bench/experiments/IMDN.yaml"
    elif args.model == "FDAN":
        path = "/home/csjunxu-3090/syb/bench/experiments/fdan.yaml"
    elif args.model == "FDADN":
        path = "/home/songyongbao/syb/decomp_cat/experiments/FDADN.yaml"
    
    else:
        path = None
    
    return path