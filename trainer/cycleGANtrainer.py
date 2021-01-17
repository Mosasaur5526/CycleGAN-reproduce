from data import sum2win
from utils.network import choose_generator, choose_discriminator
from utils import training, network
import torch.nn as nn
import torch.optim
from torch.autograd import Variable
import itertools
import os


class CycleGANTrainer:
    def __init__(self, args):
        self.args = args

        # 加载好训练数据
        self.a_loader = None
        self.b_loader = None
        self.__load_the_data__()

        # 根据args选取网络结构并进行初始化
        self.Gab = None
        self.Gba = None
        self.Da = None
        self.Db = None
        self.__build_the_networks__()

        # 定义好两种loss
        self.MSE = nn.MSELoss()
        self.L1 = nn.L1Loss()

        # 设置好更新网络参数的优化器
        self.g_optimizer = None
        self.d_optimizer = None
        self.__get_optimiser__()

        # 设置好更新学习率的优化器
        self.g_lr_scheduler = None
        self.d_lr_scheduler = None
        self.__get_lr_scheduler__()

        # 设定好两个生成图片buffer
        self.a_fake_sample = training.ImageBuffer(max_elements=50)
        self.b_fake_sample = training.ImageBuffer(max_elements=50)

        # 试图加载断点
        self.start_epoch = 0
        self.__checkpoints__()

    # 真正外部可调用的训练函数
    def train(self):
        # 训练的每个epoch，start_epoch是根据断点来的，在没有断点的时候就是0
        for epoch in range(self.start_epoch, self.args.epochs):

            # optimizer.param_groups[0]：长度为6的字典，包括[‘amsgrad’, ‘params’, ‘lr’, ‘betas’, ‘weight_decay’, ‘eps’]这6个参数
            # 这里不过是获取一下学习率打印出来罢了
            lr = self.g_optimizer.param_groups[0]['lr']
            print('learning rate = %.7f' % lr)

            for i, (a_real, b_real) in enumerate(zip(self.a_loader, self.b_loader)):
                # Generator Computations
                training.set_grad([self.Da, self.Db], False)  # 固定D训练G，猜测D的loss和G无关所以不需要锁定G
                self.g_optimizer.zero_grad()

                a_real = Variable(a_real[0])
                b_real = Variable(b_real[0])
                a_real, b_real = network.cuda([a_real, b_real])

                # Forward pass through generators
                a_fake = self.Gab(b_real)  # 从b的真图片生成假的a
                b_fake = self.Gba(a_real)  # 从a的真图片生成假的b

                a_recon = self.Gab(b_fake)  # 从b的假图片重构a
                b_recon = self.Gba(a_fake)  # 从a的假图片重构b

                a_idt = self.Gab(a_real)  # 注意a_idt是a_real喂进去得到的，本来Gab应该是从b转换到a
                b_idt = self.Gba(b_real)

                # Identity losses
                # 这个loss似乎在论文中没有提及，不过也是常用的，意义在于希望转换后基本上保证画面内要素不变
                a_idt_loss = self.L1(a_idt, a_real) * self.args.lamda * self.args.idt_coef
                b_idt_loss = self.L1(b_idt, b_real) * self.args.lamda * self.args.idt_coef

                # Adversarial losses
                # 对抗loss，从这里可以推断出D输出的是图片为真的概率（注意是G的视角）
                a_fake_dis = self.Da(a_fake)
                b_fake_dis = self.Db(b_fake)

                # 这些假图片判为真的概率应为1
                real_label = network.cuda(Variable(torch.ones(a_fake_dis.size())))

                # 按照论文的说法计算MSE作为G的对抗loss
                a_gen_loss = self.MSE(a_fake_dis, real_label)
                b_gen_loss = self.MSE(b_fake_dis, real_label)

                # 循环不变loss
                a_cycle_loss = self.L1(a_recon, a_real) * self.args.lamda
                b_cycle_loss = self.L1(b_recon, b_real) * self.args.lamda

                # 总的生成loss
                gen_loss = a_gen_loss + b_gen_loss + a_cycle_loss + b_cycle_loss + a_idt_loss + b_idt_loss

                # loss进行反向传播，优化器更新G的参数
                gen_loss.backward()
                self.g_optimizer.step()

                # 开始训练D
                training.set_grad([self.Da, self.Db], True)  # 解锁D训练D
                self.d_optimizer.zero_grad()

                # buffer对象是可以call的，就是将新生成的一波假图到buffer中涮一下拿出一些给D更新
                a_fake = Variable(torch.Tensor(self.a_fake_sample([a_fake.cpu().data.numpy()])[0]))
                b_fake = Variable(torch.Tensor(self.b_fake_sample([b_fake.cpu().data.numpy()])[0]))
                a_fake, b_fake = network.cuda([a_fake, b_fake])

                # 让两个D分别对真实样本和生成样本进行判断
                a_real_dis = self.Da(a_real)
                a_fake_dis = self.Da(a_fake)
                b_real_dis = self.Db(b_real)
                b_fake_dis = self.Db(b_fake)
                real_label = network.cuda(Variable(torch.ones(a_real_dis.size())))
                fake_label = network.cuda(Variable(torch.zeros(a_fake_dis.size())))

                # 这些loss都和标签都用MSE
                a_dis_real_loss = self.MSE(a_real_dis, real_label)
                a_dis_fake_loss = self.MSE(a_fake_dis, fake_label)
                b_dis_real_loss = self.MSE(b_real_dis, real_label)
                b_dis_fake_loss = self.MSE(b_fake_dis, fake_label)

                # Total discriminators losses
                a_dis_loss = (a_dis_real_loss + a_dis_fake_loss) * 0.5
                b_dis_loss = (b_dis_real_loss + b_dis_fake_loss) * 0.5

                # loss进行反向传播，优化器更新D的参数
                a_dis_loss.backward()
                b_dis_loss.backward()
                self.d_optimizer.step()

                print("Epoch: (%3d) (%5d/%5d) | Gen Loss:%.2e | Dis Loss:%.2e" % (
                    epoch, i + 1, min(len(self.a_loader), len(self.b_loader)), gen_loss, a_dis_loss + b_dis_loss))

            # 保存最新的断点
            training.save_checkpoint({'epoch': epoch + 1,
                                      'Da': self.Da.state_dict(),
                                      'Db': self.Db.state_dict(),
                                      'Gab': self.Gab.state_dict(),
                                      'Gba': self.Gba.state_dict(),
                                      'd_optimizer': self.d_optimizer.state_dict(),
                                      'g_optimizer': self.g_optimizer.state_dict()},
                                     '%s/latest.ckpt' % self.args.checkpoint_dir)

            # 更新学习率
            self.g_lr_scheduler.step()
            self.d_lr_scheduler.step()

    # 加载好训练数据
    def __load_the_data__(self):
        domains = sum2win.DomainLoader(self.args)
        self.a_loader = domains.get_train_domain_a()
        self.b_loader = domains.get_train_domain_b()

    # 根据args选取网络结构并进行初始化
    def __build_the_networks__(self):
        self.Gab = choose_generator(input_nc=3, output_nc=3, ngf=self.args.ngf, netG=self.args.gen_net,
                                    norm=self.args.norm, use_dropout=not self.args.no_dropout,
                                    gpu_ids=self.args.gpu_ids)
        self.Gba = choose_generator(input_nc=3, output_nc=3, ngf=self.args.ngf, netG=self.args.gen_net,
                                    norm=self.args.norm, use_dropout=not self.args.no_dropout,
                                    gpu_ids=self.args.gpu_ids)
        self.Da = choose_discriminator(input_nc=3, ndf=self.args.ndf, netD=self.args.dis_net, n_layers_D=3,
                                       norm=self.args.norm, gpu_ids=self.args.gpu_ids)
        self.Db = choose_discriminator(input_nc=3, ndf=self.args.ndf, netD=self.args.dis_net, n_layers_D=3,
                                       norm=self.args.norm, gpu_ids=self.args.gpu_ids)

    # 更新网络参数的优化器
    def __get_optimiser__(self):
        self.g_optimizer = torch.optim.Adam(itertools.chain(self.Gab.parameters(), self.Gba.parameters()),
                                            lr=self.args.lr,
                                            betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(itertools.chain(self.Da.parameters(), self.Db.parameters()),
                                            lr=self.args.lr,
                                            betas=(0.5, 0.999))

    # 更新学习率的优化器
    def __get_lr_scheduler__(self):
        self.g_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.g_optimizer,
                                                                lr_lambda=training.LambdaLR(self.args.epochs, 0,
                                                                                            self.args.decay_epoch).step)
        self.d_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.d_optimizer,
                                                                lr_lambda=training.LambdaLR(self.args.epochs, 0,
                                                                                            self.args.decay_epoch).step)

    def __checkpoints__(self):
        # 如果没有checkpoint就整一个出来
        if not os.path.isdir(self.args.checkpoint_dir):
            os.makedirs(self.args.checkpoint_dir)

        # 试图进行断点恢复，当然如果没有断点就算了
        try:
            ckpt = training.load_checkpoint('%s/latest.ckpt' % self.args.checkpoint_dir)
            self.start_epoch = ckpt['epoch']
            self.Da.load_state_dict(ckpt['Da'])
            self.Db.load_state_dict(ckpt['Db'])
            self.Gab.load_state_dict(ckpt['Gab'])
            self.Gba.load_state_dict(ckpt['Gba'])
            self.d_optimizer.load_state_dict(ckpt['d_optimizer'])
            self.g_optimizer.load_state_dict(ckpt['g_optimizer'])
        except:
            print(' [*] No checkpoint!')
