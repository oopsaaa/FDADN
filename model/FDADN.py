import torch
import torch.nn as nn
import model.block as B

def make_model(args, parent=False):
    model = FDADN()
    return model


class FDADN(nn.Module):
    def __init__(self, in_nc=3, nf=64, num_modules=4, out_nc=3, upscale=4):
        super(FDADN, self).__init__()

        self.fea_conv = B.conv_layer(in_nc, nf, kernel_size=3)

        self.B1 = B.FDADB(in_channels=nf)
        self.B2 = B.FDADB(in_channels=nf)
        self.B3 = B.FDADB(in_channels=nf)
        self.B4 = B.FDADB(in_channels=nf)
        self.c = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')

        self.LR_conv = B.conv_layer(nf, nf, kernel_size=3)

        upsample_block = B.pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=4)
        self.scale_idx = 0
        
        # self.out_conv3 = nn.Conv2d(nf, 3, 3, 1, padding="same", bias=True, dilation=1,groups=1)

    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)


        out_lr = self.LR_conv(out_B1) + out_fea

        output = self.upsampler(out_lr)

        return output

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
