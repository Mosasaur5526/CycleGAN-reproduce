from data import sum2win
from utils import training, network
from utils.network import choose_generator
import torch
import torchvision
import os
from torch.autograd import Variable


class Tester:
    def __init__(self, args):
        self.args = args

        self.a_loader = None
        self.b_loader = None
        self.__load_the_data__()

        self.Gab = None
        self.Gba = None
        self.__build_the_networks__()

    # 加载好训练数据
    def __load_the_data__(self):
        domains = sum2win.DomainLoader(self.args)
        self.a_loader = domains.get_test_domain_a()
        self.b_loader = domains.get_test_domain_b()

    # 根据args选取网络结构，因为是测试集所以只需要搞出G就行了
    def __build_the_networks__(self):
        self.Gab = choose_generator(input_nc=3, output_nc=3, ngf=self.args.ngf, netG=self.args.gen_net,
                                    norm=self.args.norm, use_dropout=not self.args.no_dropout,
                                    gpu_ids=self.args.gpu_ids)
        self.Gba = choose_generator(input_nc=3, output_nc=3, ngf=self.args.ngf, netG=self.args.gen_net,
                                    norm=self.args.norm, use_dropout=not self.args.no_dropout,
                                    gpu_ids=self.args.gpu_ids)

    # 试图从断点处加载模型，如果没有说明还未训练
    def __checkpoints__(self):
        try:
            ckpt = training.load_checkpoint('%s/latest.ckpt' % self.args.checkpoint_dir)
            self.start_epoch = ckpt['epoch']
            self.Gab.load_state_dict(ckpt['Gab'])
            self.Gba.load_state_dict(ckpt['Gba'])
        except:
            print('Model is still untrained!')

    def test(self):
        self.__checkpoints__()

        # 调整网络到评估模式，这样做是为了处理训练时使用的dropout以及batch normalisation
        self.Gab.eval()
        self.Gba.eval()

        # 准备好数据送到显卡上
        a_real_test = Variable(iter(self.a_loader).next()[0], requires_grad=True)
        b_real_test = Variable(iter(self.b_loader).next()[0], requires_grad=True)
        a_real_test, b_real_test = network.cuda([a_real_test, b_real_test])

        # 生成对应的测试结果
        with torch.no_grad():
            a_fake_test = self.Gab(b_real_test)
            b_fake_test = self.Gba(a_real_test)
            a_recon_test = self.Gab(b_fake_test)
            b_recon_test = self.Gba(a_fake_test)

        # 最后的排版每行都是 原图 | 转换后的图 | 重构的图
        pic = (torch.cat([a_real_test, b_fake_test, a_recon_test, b_real_test, a_fake_test, b_recon_test],
                         dim=0).data + 1) / 2.0

        # 保存到对应路径之下
        if not os.path.isdir(self.args.results_dir):
            os.makedirs(self.args.results_dir)

        torchvision.utils.save_image(pic, self.args.results_dir + '/sample.jpg', nrow=3)
