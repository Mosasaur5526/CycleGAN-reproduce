import functools
from abc import ABC
import torch
from torch import nn
from .micro_structure import conv_norm_relu, dconv_norm_relu


# 每一个这个block代表的是对称的一层欠采样和一层上采样
class UnetSkipConnectionBlock(nn.Module, ABC):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()

        # innermost表示最里面那层，outermost表示最外面那层，submodule是从innermost一层层往外套娃套到了哪一层
        self.outermost = outermost

        # 还是说如果是instance的norm就用bias
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # 这个指的是靠外侧的输入和输出通道数，通常来讲应该是保持一致的
        if input_nc is None:
            input_nc = outer_nc
        # inner_nc指的是欠采样的输出通道数和上采样的输入通道数

        # 欠采样的卷积层
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)

        if outermost:
            # 这里上采样输入通道数乘以2是因为输入还包括skip connection做的concatenation中的原数据带来的通道数
            upconv = nn.ConvTranspose2d(inner_nc*2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [nn.ReLU(True), upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [nn.LeakyReLU(0.2, True), downconv]
            up = [nn.ReLU(True), upconv, norm_layer(outer_nc)]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc*2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [nn.LeakyReLU(0.2, True), downconv, norm_layer(inner_nc)]
            up = [nn.ReLU(True), upconv, norm_layer(outer_nc)]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)  # 如果是最外层，这已经是整个模型了
        else:
            return torch.cat([x, self.model(x)], 1)  # 如果不是最外层，那还需要将输入和输出拼接起来，对应着非最内层上采样的输入尺寸


# U-net，参数比ResNet多但需要的显存更少且速度更快
class UnetGenerator(nn.Module, ABC):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        # 下面的过程就是由内而外地套娃
        unet_block = UnetSkipConnectionBlock(ngf*8, ngf*8, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf*8, ngf*8, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf*4, ngf*8, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf*2, ngf*4, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)
        self.unet_model = unet_block

    def forward(self, x):
        return self.unet_model(x)


# ResNet用的这个单元，按照论文的说法使用了两个3×3的块
class ResidualBlock(nn.Module, ABC):
    def __init__(self, dim, norm_layer, use_dropout, use_bias):
        super(ResidualBlock, self).__init__()
        res_block = [nn.ReflectionPad2d(1),
                     conv_norm_relu(dim, dim, kernel_size=3,
                     norm_layer=norm_layer, bias=use_bias)]
        if use_dropout:
            res_block += [nn.Dropout(0.5)]
        res_block += [nn.ReflectionPad2d(1),
                      nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
                      norm_layer(dim)]

        self.res_block = nn.Sequential(*res_block)

    def forward(self, x):
        return x + self.res_block(x)  # 这个地方体现出来residual


# 论文中选用的网络结构
class ResnetGenerator(nn.Module, ABC):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=True, num_blocks=6):
        super(ResnetGenerator, self).__init__()

        # 这里和D一样，如果是instance的归一化就加bias
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        res_model = [nn.ReflectionPad2d(3),
                     conv_norm_relu(input_nc, ngf * 1, 7, norm_layer=norm_layer, bias=use_bias),  # c7s1-64
                     conv_norm_relu(ngf * 1, ngf * 2, 3, 2, 1, norm_layer=norm_layer, bias=use_bias),  # d128
                     conv_norm_relu(ngf * 2, ngf * 4, 3, 2, 1, norm_layer=norm_layer, bias=use_bias)]  # d256

        # 6个R256
        for i in range(num_blocks):
            res_model += [ResidualBlock(ngf * 4, norm_layer, use_dropout, use_bias)]

        res_model += [dconv_norm_relu(ngf * 4, ngf * 2, 3, 2, 1, 1, norm_layer=norm_layer, bias=use_bias),  # u128
                      dconv_norm_relu(ngf * 2, ngf * 1, 3, 2, 1, 1, norm_layer=norm_layer, bias=use_bias),  # u64
                      nn.ReflectionPad2d(3),  # 长和宽各往两边扩3的padding
                      nn.Conv2d(ngf, output_nc, 7),  # c7s1-3
                      nn.Tanh()]
        self.res_model = nn.Sequential(*res_model)

    def forward(self, x):
        return self.res_model(x)
