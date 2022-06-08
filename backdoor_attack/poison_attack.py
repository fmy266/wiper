import torch
import torchvision
import torchvision.transforms as transforms
import PIL
import numpy as np
import os
from . import generative_model

base_params = {"size":[3,3,3], "poison_label":0, "random_pos":False, "dataset_name":"cifar10",
                "normalize_param" : None, "trigger" : None, "random_init" : None, "trigger_path" : None}

class PoisonTrainer:
    name = "base_attack"
    def __init__(self, model, dataloader, loss_func, optim, trigger, poison_prob, device,
                      trigger_params, scheduler = None, val_loader = None, test_loader = None):
        self.device = device
        self.model = model.to(self.device)
        self.dataloader = dataloader
        self.loss_func = loss_func
        self.trigger = trigger
        self.optim = optim
        self.scheduler = scheduler
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.poison_prob = poison_prob
        self.trigger_params = trigger_params


    def train(self, epoch, valid_func):
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        for epoch_ in range(1, epoch + 1):
            for data, label in self.dataloader:
                self.optim.zero_grad()
                data, label = data.to(self.device), label.to(self.device)
                poison_data, poison_label = self.trigger.paste(data, label, self.poison_prob)

                with torch.cuda.amp.autocast(enabled = True):
                    predict = self.model(data)
                    loss = self.loss_func(predict, label)
                    if poison_label != None:
                        loss = loss + self.poison_prob * self.loss_func(self.model(poison_data), poison_label)
                scaler.scale(loss).backward()
                scaler.step(self.optim)
                scaler.update()
            if self.scheduler:
                self.scheduler.step()
            clean_acc, poison_acc = self.validate()
            print("clean_acc = {:.2f}, poison_acc = {:.2f}\n".format(clean_acc, poison_acc))
        valid_func(self.model, self.trigger.trigger, clean_acc, poison_acc, self.device)


    @torch.no_grad()
    def validate(self):
        num, num2, poison_num, clean_num = 0, 0, 0, 0
        # accuracy and attack success rate
        for data, label in self.val_loader:
            data, label = data.to(self.device), label.to(self.device)
            clean_predict = self.model(data)
            clean_num += (clean_predict.max(dim = 1)[1] == label).sum().item()
            mask = (label != self.trigger.poison_label)
            poison_data, poison_label = self.trigger.paste(data[mask], label[mask], 1.)
            poison_predict = self.model(poison_data)
            poison_num += (poison_predict.max(dim = 1)[1] == self.trigger.poison_label).sum().item()
            num += label.size()[0]
            num2 += mask.sum().item()
        return clean_num / num * 100., poison_num / num2 * 100.

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model"])
        self.trigger.trigger = checkpoint["trigger"]
        if self.name == "TrojanNNAttack":
            self.trigger.trigger = checkpoint["trigger"].trigger
        self.trigger.trigger = self.trigger.trigger.to(self.device)


class BadNet(PoisonTrainer):
    name = "BadNet"
    def __init__(self, model, dataloader, loss_func, optim, poison_prob, device,
                      trigger_params, scheduler = None, val_loader = None, test_loader = None):
        super(BadNet, self).__init__(model, dataloader, loss_func, optim, None, poison_prob, device,
                      trigger_params, scheduler, val_loader, test_loader)
        base_params.update(trigger_params)
        self.trigger = Trigger(size = base_params["size"], poison_label = base_params["poison_label"],
                                random_pos = False, device = device,
                                normalize_param = base_params["normalize_param"], trigger = base_params["trigger"],
                                random_init = base_params["random_init"], trigger_path = base_params["trigger_path"])


class BlendAttack(PoisonTrainer):
    name = "BlendAttack"
    def __init__(self, model, dataloader, loss_func, optim, poison_prob, device,
                      trigger_params, scheduler = None, val_loader = None, test_loader = None):
        super(BlendAttack, self).__init__(model, dataloader, loss_func, optim, None, poison_prob, device,
                      trigger_params, scheduler, val_loader, test_loader)
        base_params.update(trigger_params)
        self.trigger = BlendTrigger(size = base_params["size"], poison_label = base_params["poison_label"],
                                random_pos = base_params["random_pos"], device = device,
                                normalize_param = base_params["normalize_param"], trigger = base_params["trigger"],
                                random_init = base_params["random_init"], trigger_path = base_params["trigger_path"])


class EnhancedTriggerAttack(PoisonTrainer):
    name = "EnhancedTriggerAttack"
    def __init__(self, model, dataloader, loss_func, optim, poison_prob, device,
                      trigger_params, scheduler = None, val_loader = None, test_loader = None):
        super(EnhancedTriggerAttack, self).__init__(model, dataloader, loss_func, optim, None, poison_prob, device,
                      trigger_params, scheduler, val_loader, test_loader)
        base_params.update(trigger_params)
        self.trigger = EnhancedTrigger(size = base_params["size"], poison_label = base_params["poison_label"],
                                random_pos = base_params["random_pos"], device = device,
                                normalize_param = base_params["normalize_param"], trigger = base_params["trigger"],
                                random_init = base_params["random_init"], trigger_path = base_params["trigger_path"])
        self.trigger.data_shape = {"mnist":[1,28,28], "fashionmnist":[1,28,28], "SVHN":[3,32,32], "cifar10":[3,32,32], "cifar100":[3,32,32]}[base_params["dataset_name"]]