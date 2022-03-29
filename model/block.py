import torch.nn as nn

import torch
import torch.nn.functional as F



class CCALayer1(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CCALayer1, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )


    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)


def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu'):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias, groups=groups)
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)


def mean_channels(inp):
    assert(inp.dim() == 4)
    spatial_sum = inp.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (inp.size(2) * inp.size(3))

def stdv_channels(inp):
    assert(inp.dim() == 4)
    F_mean = mean_channels(inp)
    F_variance = (inp - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (inp.size(2) * inp.size(3))
    return F_variance.pow(0.5)


class CCALayer(nn.Module):
    def __init__(self, channel):
        super(CCALayer, self).__init__()

        self.conv3 = conv_layer(channel, channel, 3)
        self.act = activation('lrelu', neg_slope=0.05)


    def forward(self, x):
        y = self.act(self.conv3(x))

        return x-y, y



def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer



def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

class ESA(nn.Module):
    """
    attention
    """
    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False) 
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3+cf)
        m = self.sigmoid(c4)
        
        return x * m


class FDADB(nn.Module):
    def __init__(self, in_channels):
        super(FDADB, self).__init__()
        self.rc = self.remaining_channels = in_channels
        self.c1_d = conv_layer(in_channels, in_channels, 1)
        self.c1_r = FDAB(in_channels,de_split_rate=1)
        self.c2_d = conv_layer(self.remaining_channels, in_channels, 1)
        self.c2_r = FDAB(self.remaining_channels,de_split_rate=1)
        self.c3_d = conv_layer(self.remaining_channels, in_channels, 1)
        self.c3_r = FDAB(self.remaining_channels,de_split_rate=1)
        self.c4 = conv_layer(self.remaining_channels, in_channels, 3)
        self.c5 = conv_layer(in_channels, in_channels, 1)
        self.act = activation('lrelu', neg_slope=0.05)
        self.esa = CCALayer1(in_channels)
        self.distil_1 = CCALayer(in_channels)
        self.distil_2 = CCALayer(in_channels)
        self.distil_3 = CCALayer(in_channels)

    def forward(self, input):
        dist1,rem1 = self.distil_1(input)
        distilled_c1 = self.act(self.c1_d(dist1))
        r_c1 = (self.c1_r(rem1))

        dist2,rem2 = self.distil_2(r_c1)
        distilled_c2 = self.act(self.c2_d(dist2))
        r_c2 = (self.c2_r(rem2))
        
        dist3,rem3 = self.distil_3(r_c2)
        distilled_c3 = self.act(self.c3_d(dist3))
        r_c3 = (self.c3_r(rem3))

        r_c4 = self.act(self.c4(r_c3))

        out = distilled_c1 + distilled_c2+ distilled_c3+r_c4
        out_fused = self.esa(self.c5(out))
        
        output = input + out_fused

        return output



def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)



class FDAB(nn.Module):
    """
    A-A模式
    """
    def __init__(self,in_channels,de_split_rate=1):
        super(FDAB, self).__init__()
        self.de_in_channels = self.de_out_channels = int(in_channels * de_split_rate)

        self.conv_3 = conv_layer(self.de_in_channels, self.de_out_channels, 3)
        self.act = activation('lrelu', neg_slope=0.05)

    def forward(self,input):
        """
        input.shape = n * c * h * w
        """
    
        output = input - (self.act(self.conv_3(input)))
    
    
        return output
    
class FDAB_2(nn.Module):
    """
    A-B模式
    """
    def __init__(self,in_channels,de_split_rate=0.5):
        super(FDAB_2, self).__init__()
        self.de_in_channels = self.de_out_channels = int(in_channels * de_split_rate)
        self.conv_3 = conv_layer(self.de_in_channels, self.de_out_channels, 3)
        self.act = activation('lrelu', neg_slope=0.05)
    
    def forward(self,input):
        """
        input.shape = n * c * h * w
        """
        de_input = input[:,:self.de_in_channels,...]
        re_input = input[:,self.de_in_channels:,...]
        
        de_output = self.act(self.conv_3(de_input))
        
        detial_out = re_input - de_output
        
        output = torch.cat([detial_out,de_output], dim=1)
        
        return output