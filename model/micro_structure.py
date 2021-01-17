import torch.nn as nn


# 卷积 + 归一化 + leaky relu
def conv_norm_lrelu(in_dim, out_dim, kernel_size, stride=1, padding=0, norm_layer=nn.BatchNorm2d, bias=False):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=bias),
        norm_layer(out_dim), nn.LeakyReLU(0.2, True))


# 卷积 + 归一化 + 普通relu
def conv_norm_relu(in_dim, out_dim, kernel_size, stride=1, padding=0, norm_layer=nn.BatchNorm2d, bias=False):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=bias),
        norm_layer(out_dim), nn.ReLU(True))


# 去卷积 + 归一化 + relu
def dconv_norm_relu(in_dim, out_dim, kernel_size, stride=1, padding=0, output_padding=0, norm_layer=nn.BatchNorm2d, bias=False):
    return nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride, padding, output_padding, bias=bias),
        norm_layer(out_dim), nn.ReLU(True))
