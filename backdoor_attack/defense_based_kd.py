#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import copy
import torch
import torch.nn.functional as F
import torch.optim as optim


class KD:
    name: str = 'KD'

    def __init__(self, teacher_model, student_model, train_loader, test_loader, device):
        super(KD, self).__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device


    def defense(self, epoch):
        T = 2.
        alpha = 0.5
        optim = torch.optim.SGD(self.student_model.parameters(), lr = 0.01)
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        loss_func = torch.nn.CrossEntropyLoss()
        kn_loss_func = torch.nn.KLDivLoss()

        for epoch in range(1, epoch + 1):
            self.student_model.train()
            for data, label in self.train_loader:
                optim.zero_grad()
                data, label = data.to(self.device), label.to(self.device)
                output_s = self.student_model(data)
                output_t = self.teacher_model(data).detach()
                soft_loss = kn_loss_func(F.log_softmax(output_s / T, dim=1), F.softmax(output_t / T, dim=1))
                # soft_loss = torch.nn.functional.softmax(output_t, dim = 1) * torch.nn.functional.softmax(output_s, dim = 1).log()
                hard_loss = loss_func(output_s, label)
                loss = alpha * soft_loss + (1-alpha) * hard_loss
                loss.backward()
                optim.step()
