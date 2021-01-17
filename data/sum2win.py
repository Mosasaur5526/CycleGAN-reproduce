import torchvision.datasets as sets
import torchvision.transforms as transforms
import torch
import torch.utils.data
import os


class DomainLoader:
    def __init__(self, args):
        self.args = args
        self.path_a = None
        self.path_b = None
        self.path_a2 = None
        self.path_b2 = None
        self.transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),  # 随机翻转图像
             transforms.Resize((args.load_height, args.load_width)),  # 读入图像
             transforms.RandomCrop((args.crop_height, args.crop_width)),  # 将图像按照设定的尺寸进行剪裁
             transforms.ToTensor(),  # 转换为张量
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])  # 归一化
        self.__set_path__(args.dataset_dir)  # 设置好训练集的路径

    def __set_path__(self, path):
        self.path_a = os.path.join(path, "trainA")
        self.path_b = os.path.join(path, "trainB")
        self.path_a2 = os.path.join(path, "testA")
        self.path_b2 = os.path.join(path, "testB")

    def get_train_domain_a(self):
        # 加载数据集A
        a_loader = None
        if self.path_a is not None:
            a_loader = torch.utils.data.DataLoader(sets.ImageFolder(self.path_a, transform=self.transform),
                                                   batch_size=self.args.batch_size, shuffle=True, num_workers=0)
        else:
            print("Path of trainA unset!")
        return a_loader

    def get_train_domain_b(self):
        # 加载数据集B
        b_loader = None
        if self.path_b is not None:
            b_loader = torch.utils.data.DataLoader(sets.ImageFolder(self.path_b, transform=self.transform),
                                                   batch_size=self.args.batch_size, shuffle=True, num_workers=0)
        else:
            print("Path of trainB unset!")
        return b_loader

    def get_test_domain_a(self):
        # 加载测试集A
        a_loader = None
        if self.path_a2 is not None:
            a_loader = torch.utils.data.DataLoader(sets.ImageFolder(self.path_a2, transform=self.transform),
                                                   batch_size=self.args.batch_size, shuffle=True, num_workers=0)
        else:
            print("Path of testA unset!")
        return a_loader

    def get_test_domain_b(self):
        # 加载测试集B
        b_loader = None
        if self.path_b2 is not None:
            b_loader = torch.utils.data.DataLoader(sets.ImageFolder(self.path_b2, transform=self.transform),
                                                   batch_size=self.args.batch_size, shuffle=True, num_workers=0)
        else:
            print("Path of testA unset!")
        return b_loader
