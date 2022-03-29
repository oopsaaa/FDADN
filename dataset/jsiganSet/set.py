import random
import numpy as np
import cv2
import torch
import torch.utils.data as data
import time
import os
import sys
import lpips
from dataset.HDRTV_set import HDRTV_utils as util


class Dataset(data.Dataset):
    def __init__(self,args=None, dataset_train=None, cfg=None):
        self.args = args
        self.cfg = cfg
        self.dataset_train = dataset_train
        if dataset_train:
            self.GT = np.load(self.cfg.SRITM.TRAIN_DATAROOT_GT)
            self.LQ = np.load(self.cfg.SRITM.TRAIN_DATAROOT_LQ)
                
        else:
            self.GT = np.load(self.cfg.SRITM.VALID_DATAROOT_GT)
            self.LQ = np.load(self.cfg.SRITM.VALID_DATAROOT_LQ)
    def __getitem__(self, index):
        img_GT = self.GT[index]
        img_LQ = self.LQ[index]
        # 归一化[0,1]  RGB->YUV
        
        img_GT = (img_GT/1023).astype(np.float32)
        img_LQ = (img_LQ/255).astype(np.float32)

        
        img_GT = torch.from_numpy(img_GT).float().clamp(min=0, max=1)
        img_LQ = torch.from_numpy(img_LQ).float().clamp(min=0, max=1)
        
        return img_LQ, img_GT

    def __len__(self):
        return len(self.GT)
    
    def __measure__(self, output, gt,metrics):
        # 四维的矩阵，0维表示图片的数量，这里是归一化后的
        outputBatch_yuvNumpy_bhwc = output.detach().cpu().numpy().transpose(0, 2, 3, 1).clip(0, 1)
        gtBatch_yuvNumpy_bhwc = gt.detach().cpu().numpy().transpose(0, 2, 3, 1).clip(0, 1)
        
        start_tme = time.time()

        output_yuvNumpy_hwc = outputBatch_yuvNumpy_bhwc[0,:,:,:]
        gt_yuvNumpy_hwc = gtBatch_yuvNumpy_bhwc[0,:,:,:]
        metrics['psnr'].append(sm.peak_signal_noise_ratio(image_true=output_yuvNumpy_hwc[:,:,0], image_test=gt_yuvNumpy_hwc[:,:,0], data_range=1))
        metrics['ssim'].append(util.calculate_ssim(img=np.expand_dims(output_yuvNumpy_hwc[:, :, 0] * 255, axis=-1), img2=np.expand_dims(gt_yuvNumpy_hwc[:, :, 0] * 255, axis=-1)))
        endup_time = time.time()
        
        print('\rValidation with: ' + str(endup_time - start_tme) + ' (s) per inference ', end="")
        return metrics