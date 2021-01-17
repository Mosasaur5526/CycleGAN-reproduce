from argparse import ArgumentParser
from trainer import cycleGANtrainer as GAN_trainer
from trainer import tester


# To get arguments from commandline
def get_args():
    parser = ArgumentParser(description='cycleGAN PyTorch')
    parser.add_argument('--epochs', type=int, default=50)  # 训练轮数
    parser.add_argument('--decay_epoch', type=int, default=25)  # 训练多少轮后学习率开始指数衰减
    parser.add_argument('--batch_size', type=int, default=1)  # batch_size
    parser.add_argument('--lr', type=float, default=.0002)  # 初始学习率
    parser.add_argument('--load_height', type=int, default=256)  # 加载图片的高度
    parser.add_argument('--load_width', type=int, default=256)  # 加载图片的宽度
    parser.add_argument('--gpu_ids', type=str, default='0')  # 选择GPU
    parser.add_argument('--crop_height', type=int, default=256)  # 剪切后图片的高度
    parser.add_argument('--crop_width', type=int, default=256)  # 剪切后图片的宽度
    parser.add_argument('--lamda', type=int, default=10)  # 循环不变性loss的比例系数，按照论文中的说法通常取10
    parser.add_argument('--idt_coef', type=float, default=0.5)  #
    parser.add_argument('--training', type=bool, default=False)  # 是否训练
    parser.add_argument('--testing', type=bool, default=True)  # 是否测试
    parser.add_argument('--results_dir', type=str, default='./results')  # 结果保存路径
    parser.add_argument('--dataset_dir', type=str, default='./summer2winter_yosemite')  # 数据集所在路径
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/summer2winter_yosemite')  # 断点保存路径
    parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')  # 按照每个样本做正归一化或者做batch归一化
    parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')  # 用dropout
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
    parser.add_argument('--gen_net', type=str, default='resnet_9blocks')
    parser.add_argument('--dis_net', type=str, default='n_layers')
    args = parser.parse_args()
    return args


def main():
    # 获取参数列表
    args = get_args()

    # 捣腾GPU
    str_ids = str(args.gpu_ids).split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        iden = int(str_id)
        if iden >= 0:
            args.gpu_ids.append(iden)

    # 进行训练
    if args.training:
        print("Training")
        trainer = GAN_trainer.CycleGANTrainer(args)
        trainer.train()

    # 进行测试
    if args.testing:
        print("Testing")
        test = tester.Tester(args)
        test.test()


if __name__ == '__main__':
    main()
