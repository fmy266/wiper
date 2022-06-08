import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.transforms.functional as visionfunc
import os
import PIL

class DataManger:

    dataset_dict = {
                "cifar10":datasets.CIFAR10,
                "SVHN":datasets.SVHN
                }

    @staticmethod
    def get_normalize_params(dataset_name):
        if dataset_name == "cifar10":
            return ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        elif dataset_name == "SVHN":
            return ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        else:
            raise ValueError("The dataset not exists.")

    @staticmethod
    def get_dataloader(dataset_name, root, ratio = None, sampler_num = None, indices = None, batch_size = 128):

        size = 32
        normalize_params = DataManger.get_normalize_params(dataset_name)

        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(normalize_params[0], normalize_params[1]),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((size,size)),
            transforms.Normalize(normalize_params[0], normalize_params[1]),
        ])


        train_dataset = DataManger.dataset_dict[dataset_name](
            root=root, split = "train", download=False, transform=train_transform)
        test_dataset = DataManger.dataset_dict[dataset_name](
            root=root, split = "test", download=False, transform=test_transform)


        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers = 4, pin_memory = True, drop_last= True)

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True,
            num_workers = 4, pin_memory = True, drop_last= True)

        return train_dataloader, test_dataloader


# model code refer to https://github.com/rwightman/pytorch-image-models
class ModelManger:
    @staticmethod
    def get_model(dataset_name, model_name, device):
        import models.resnet as resnet
        nets = {"resnet18":resnet.ResNet18}
        net = nets[model_name]().to(device)
        return net
