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
from tensorboardX import SummaryWriter


# args
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',type=str,default="SRITM",help="select dataset")
parser.add_argument('--model',type=str,default="FDADN",help="select model")
parser.add_argument('--shuffle',type=str,default=True,help=" train shuffle")
args = parser.parse_args()


# save model path

save_path = "/home/songyongbao/syb/decomp_cat/save_models/"+args.dataset+"_"+args.model+"_"+str(args.shuffle)+".pt"


# SummaryWriter
log_path = args.dataset+"_"+args.model+"_"+str(args.shuffle)+"_tb_logs/"
if not os.path.exists(log_path):
    os.mkdir(log_path)
writer = SummaryWriter(log_dir=log_path,flush_secs=300)


# cfg
cfg_path = config_path(args=args)
cfg = CN.load_cfg(open(cfg_path))
cfg.freeze()



# random seed
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# dataset
train_set = Datasets(args=args,train=True,cfg=cfg)
valid_set = Datasets(args=args,train=False,cfg=cfg)



# dataloader

train_dataloader = DataLoader(train_set,sampler=None,batch_size=cfg.TRAIN_BATCH_SIZE,shuffle=True,num_workers=2)

valid_dataloader = DataLoader(valid_set,sampler=None,batch_size=1,shuffle=False,num_workers=2)


# 初始化模型
model = Net(args,cfg=cfg).cuda()
choose_initializer(cfg=cfg) 
model.apply(init_weight)


# optimizer
optimizer = Optimizer(model=model,args=args,cfg=cfg)

#scheduler
scheduler = Scheduler(optimizer=optimizer,args=args,cfg=cfg)


mean_psnr_max = 0
mean_ssim_max = 0
metrics_csv = {}
metrics_csv["psnr"] = []
metrics_csv["ssim"] = []

# train valid 
for epoch_index in range(cfg.MAX_EPOCH):
    
    # train
    model.train()
    torch.cuda.empty_cache()
    for batch_index, dataset_pakage in enumerate(train_dataloader):
        data = dataset_pakage[0]
        gt = dataset_pakage[1]
        data = data.cuda(non_blocking=True)
        gt = gt.cuda(non_blocking=True)
        
        output = model(data)
        loss = torch.nn.functional.l1_loss(output, gt)
        optimizer.zero_grad()  # 将所有参数的梯度都置零
        loss.backward()        # 误差反向传播计算参数梯度
        optimizer.step()
        writer.add_scalar("train_loss", loss, (epoch_index*len(train_dataloader)+batch_index+1))
 
    scheduler.step()
        
    # valid
    model.eval()
    
    with torch.no_grad():
        metrics = {}
        metrics['psnr'] = []
        metrics['ssim'] = []
        
        for valid_index, dataset_package in enumerate(valid_dataloader):
            data = dataset_package[0]
            gt = dataset_package[1]
            data = data.cuda(non_blocking=True)
            gt = gt.cuda(non_blocking=True)
            output = model(data)
            
            # psnr ssim 字典 key为列表
            metrics = valid_set.__measure__(output=output, gt=gt,metrics=metrics)
        mean_psnr = sum(metrics['psnr'])/len(metrics['psnr'])
        mean_ssim = sum(metrics['ssim'])/len(metrics['ssim'])
        
        print("psnr:"+str(mean_psnr)+" , ssim:"+str(mean_ssim))
        metrics_csv["psnr"].append(mean_psnr)
        metrics_csv["ssim"].append(mean_ssim)
                
        if (mean_psnr >= mean_psnr_max) & (mean_ssim >= mean_ssim_max):
            mean_psnr_max = mean_psnr
            mean_ssim_max = mean_ssim
            torch.save(model,save_path)
            
            mt_max = pd.DataFrame({"psnr_max":[mean_psnr_max],"ssim_max":[mean_ssim_max]})
            mt_max.to_csv(args.dataset+"_"+args.model+"_"+str(args.shuffle)+'_metrics_max',index=False,sep=",")
        
        writer.add_scalar("psnr", mean_psnr, (epoch_index+1))
        writer.add_scalar("ssim", mean_ssim, (epoch_index+1))
    print("num epoch:"+str(epoch_index))


dataframe = pd.DataFrame(metrics_csv)
dataframe.to_csv(args.dataset+"_"+args.model+"_"+str(args.shuffle)+"_metrics.csv",index=False,sep=",")
        