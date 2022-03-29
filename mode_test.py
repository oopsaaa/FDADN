# coding=gbk
import torch
import numpy as np
from models import *
from datasets import *
import argparse
from torch.utils.data import DataLoader
import pandas as pd
from yacs.config import CfgNode as CN
from configs import *
from initializers import *
from optimizers import *
from schedulers import *
import os




# args
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',type=str,default="SSR",help="select dataset")
parser.add_argument('--model',type=str,default="FDADN",help="select model")
parser.add_argument('--shuffle',type=str,default=True,help=" train shuffle")
parser.add_argument("--server",type=str,default="ubuntu")
args = parser.parse_args()






# cfg
cfg_path = config_path(args=args)
cfg = CN.load_cfg(open(cfg_path))
cfg.freeze()




# dataset

test_set = Datasets(args=args,train=False,cfg=cfg)



# dataloader
test_dataloader = DataLoader(test_set,sampler=None,batch_size=1,shuffle=False,num_workers=2)


# model
path ="save_models/SRITM_FDADN_True.pt"
model = torch.load(path).cuda()

with torch.no_grad():
    metrics = {}
    metrics['psnr'] = []
    metrics['ssim'] = []
    
    for valid_index, dataset_package in enumerate(test_dataloader):
        data = dataset_package[0]
        gt = dataset_package[1]
        data = data.cuda(non_blocking=True)
        gt = gt.cuda(non_blocking=True)
        output = model(data)
        
        # psnr ssim 字典 key为列表
        metrics = test_set.__measure__(output=output, gt=gt,metrics=metrics)
    mean_psnr = sum(metrics['psnr'])/len(metrics['psnr'])
    mean_ssim = sum(metrics['ssim'])/len(metrics['ssim'])
    
    print("psnr:"+str(mean_psnr)+" , ssim:"+str(mean_ssim))
    metrics_csv["psnr"].append(mean_psnr)
    metrics_csv["ssim"].append(mean_ssim)
            



