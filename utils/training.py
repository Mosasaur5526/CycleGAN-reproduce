import torch
import numpy as np
import copy


# 保存断点，savepath是存储路径
def save_checkpoint(state, save_path):
    torch.save(state, save_path)


# 从断点处加载续训
def load_checkpoint(ckpt_path, map_location=None):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    print('Loading checkpoint from %s succeed!' % ckpt_path)
    return ckpt


# 屏蔽梯度，在训练时被调用
def set_grad(nets, requires_grad=False):
    for net in nets:
        for param in net.parameters():
            param.requires_grad = requires_grad


# 在论文第5页下方提出了这个buffer，目的是将G新生成的一些图片放入buffer中缓冲一下再交给D去训练
class ImageBuffer(object):
    def __init__(self, max_elements=50):
        self.max_elements = max_elements
        self.cur_elements = 0
        self.items = []

    def __call__(self, in_items):
        return_items = []
        for in_item in in_items:
            # 当buffer还没满的时候直接将新加进来的图片加到buffer并且加入到返回队列中
            if self.cur_elements < self.max_elements:
                self.items.append(in_item)
                self.cur_elements = self.cur_elements + 1
                return_items.append(in_item)
            # 如果buffer已经满了
            else:
                # 有1/2的概率将新来的图片加入到buffer中随机替换出去一张图，并且将那张图加入到返回队列中
                if np.random.ranf() > 0.5:
                    idx = np.random.randint(0, self.max_elements)
                    tmp = copy.copy(self.items[idx])
                    self.items[idx] = in_item
                    return_items.append(tmp)
                # 另外1/2的概率直接将新来的图片加入返回队列
                else:
                    return_items.append(in_item)
        return return_items


# 控制学习率指数衰减
class LambdaLR:
    def __init__(self, epochs, offset, decay_epoch):
        self.epochs = epochs
        self.offset = offset
        self.decay_epoch = decay_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_epoch)/(self.epochs - self.decay_epoch)
