from abc import ABC
from torch import nn
from .micro_structure import conv_norm_lrelu


# 应该是按照论文附录进行的网络结构搭建
class NLayerDiscriminator(nn.Module, ABC):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_bias=False):
        super(NLayerDiscriminator, self).__init__()
        # 彩色图像输入就是3层，第一层卷积将其映射到64层
        dis_model = [nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
                     nn.LeakyReLU(0.2, True)]
        nf_mult = 1

        # 第二和第三层卷积分别映射到128、256层
        for n in range(1, n_layers):
            # nf_mult_prev表示前一层输出的层数，而nf_mult表示当前这一层输出的层数
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)  # 这些层数最大就是8 * 64 = 512
            dis_model += [conv_norm_lrelu(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, norm_layer=norm_layer, padding=1, bias=use_bias)]

        # 第四层卷积映射到512层，但注意这时候stride=1
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        dis_model += [conv_norm_lrelu(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, norm_layer=norm_layer, padding=1, bias=use_bias)]

        # 最后这一层卷积把输出大小映射为1
        dis_model += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]

        self.dis_model = nn.Sequential(*dis_model)  # 这里*dis_model表示将这个list中的元素作为参数传入

    def forward(self, x):
        return self.dis_model(x)


# 这个网络结构相比之下更简单一些
class PixelDiscriminator(nn.Module, ABC):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_bias=False):
        super(PixelDiscriminator, self).__init__()
        dis_model = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.dis_model = nn.Sequential(*dis_model)

    def forward(self, x):
        return self.dis_model(x)
