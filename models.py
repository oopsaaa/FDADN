"""
using args select model 
"""
from model.FDADN import FDADN


def Net(args, cfg=None):
    if args.dataset=="ITM":
        model = FDADN(upscale=1)
    else:
        model = FDADN(upscale=4)
    return model