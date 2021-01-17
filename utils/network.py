from model import discriminator as dis, generator as gen
import functools
from torch.nn import init
import torch.nn as nn
import torch


# 获取一个归一化层
def get_norm_layer(norm_type='instance'):
    # functools.partial本质上就是一个wrapper，将后面的affine等参数设定好使得norm_layer变为一个包装好的函数
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


# 初始化网络权重
def init_weights(net, gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            init.normal_(m.weight.data, 0.0, gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


# 初始化网络，包括将数据转移到GPU上以及初始化参数
def init_network(net, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net)
    return net


# 根据参数构造一个G并初始化
def choose_generator(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, gpu_ids=[0]):
    norm_layer = get_norm_layer(norm_type=norm)

    # 如果用instance的归一化才要用bias
    if netG == 'resnet_9blocks':
        gen_net = gen.ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, num_blocks=9)
    elif netG == 'resnet_6blocks':
        gen_net = gen.ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, num_blocks=6)
    elif netG == 'unet_128':
        gen_net = gen.UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        gen_net = gen.UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)

    return init_network(gen_net, gpu_ids)


# 根据参数构造一个D并初始化
def choose_discriminator(input_nc, ndf, netD, n_layers_D=3, norm='batch', gpu_ids=[0]):
    norm_layer = get_norm_layer(norm_type=norm)

    # 如果用instance的归一化才要用bias
    if type(norm_layer) == functools.partial:
        use_bias = norm_layer.func == nn.InstanceNorm2d
    else:
        use_bias = norm_layer == nn.InstanceNorm2d

    if netD == 'n_layers':
        dis_net = dis.NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_bias=use_bias)
    elif netD == 'pixel':
        dis_net = dis.PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_bias=use_bias)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)

    return init_network(dis_net, gpu_ids)


# 转换为CUDA向量
def cuda(xs):
    if torch.cuda.is_available():
        if not isinstance(xs, (list, tuple)):
            return xs.cuda()
        else:
            return [x.cuda() for x in xs]
