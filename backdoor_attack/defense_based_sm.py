#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# author：fmy
import torch
import torch.nn.functional as F
import torch.optim as optim


class PurifyingBackdoor:

    def __init__(self, model, train_loader, test_loader, purifying_layers, alpha, beta, eta, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.purifying_layers = purifying_layers
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.sm_weights = {}
        self.sm_bias = {}
        self.device = device


    @torch.no_grad()
    def init_(self):
        for name, module in self.model.named_modules():
            if name in self.purifying_layers:
                self.sm_weights[name] = torch.zeros_like(module.weight.data)
                self.sm_bias[name] = torch.zeros_like(module.bias.data)


    def defense(self, epochs, regular_way, purifying_strategy):

        self.init_()
        optim = torch.optim.SGD(self.model.parameters(), lr = 0.01, momentum = 0.9, weight_decay = 1e-4)
        loss_func = torch.nn.CrossEntropyLoss()

        # warm-start
        optim.zero_grad()
        for data, label in self.train_loader:
            data, label = data.to(self.device), label.to(self.device)
            loss = loss_func(self.model(data), label)
            loss.backward()
            self.salience_update()
            optim.zero_grad()

        for epoch in range(1, 1 + epochs):
            self.model.train()

            for data, label in self.train_loader:
                optim.zero_grad()
                data, label = data.to(self.device), label.to(self.device)
                predict = self.model(data)
                loss = loss_func(predict, label)

                for name, module in self.model.named_modules():
                    if name in self.purifying_layers:

                        if purifying_strategy == "dynamic":
                            ratio = self.beta + (1-self.beta) * epoch / epochs

                            weight_num = int(ratio * module.weight.numel())
                            bias_num = int(ratio * module.bias.numel())

                            weight_topk = self.sm_weights[name].view(-1).topk(weight_num)[0][-1].item()
                            bias_topk = self.sm_bias[name].view(-1).topk(bias_num)[0][-1].item()

                            weight_mask = self.sm_weights[name] >= weight_topk
                            bias_mask = self.sm_bias[name] >= bias_topk

                        weight_right_mask = (module.weight * weight_mask) > 1.
                        weight_mid_mask = ((module.weight * weight_mask) <= 1) & ((module.weight * weight_mask) >= -1)
                        weight_left_mask = (module.weight * weight_mask) < -1.

                        bias_right_mask = (module.bias * bias_mask) > 1.
                        bias_mid_mask = ((module.bias * bias_mask) <= 1) & ((module.bias * bias_mask) >= -1)
                        bias_left_mask = (module.bias * bias_mask) < -1.

                        loss = loss + (self.alpha * (module.weight-1).exp() * weight_right_mask).sum() +\
                                    (self.alpha * module.weight.abs() * weight_mid_mask).sum() +\
                                    (self.alpha * (-1) * (-module.weight-1).exp() * weight_left_mask).sum()

                        loss = loss + (self.alpha * (module.bias-1).exp() * bias_right_mask).sum() +\
                                    (self.alpha * module.bias.abs() * bias_mid_mask).sum() +\
                                    (self.alpha * (-1) * (-module.bias-1).exp() * bias_left_mask).sum()


                loss.backward()
                self.salience_update()
                optim.step()

    @torch.no_grad()
    def salience_update(self):
        for name, module in self.model.named_modules():
            if name in self.purifying_layers:
                self.sm_weights[name] = self.eta * self.sm_weights[name] - \
                                               (1-self.eta) * (module.weight.grad * module.weight.data)
                self.sm_bias[name] = self.eta * self.sm_bias[name] - \
                                               (1-self.eta) * (module.bias.grad * module.bias.data)

