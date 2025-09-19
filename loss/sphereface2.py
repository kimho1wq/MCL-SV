#! /usr/bin/python
# -*- encoding: utf-8 -*-
# Adapted from https://github.com/wujiyang/Face_Pytorch (Apache License)

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy, math
from utils import accuracy

class LossFunction(nn.Module):
    def __init__(self, embed_dim, n_classes, scale=32.0, margin=0.2, lanbuda=0.7, t=3, margin_type='C', **kwargs):
        super(LossFunction, self).__init__()

        self.test_normalize = True

        self.in_feats = embed_dim
        self.m = margin
        self.s = scale
        self.weight = torch.nn.Parameter(torch.FloatTensor(n_classes, embed_dim), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)

        self.bias = nn.Parameter(torch.zeros(1, 1))
        self.t = t
        self.lanbuda = lanbuda
        self.margin_type = margin_type

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)

        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - margin)
        self.mmm = 1.0 + math.cos(math.pi - margin)

        print('Initialised SphereFace2 type: %s, m: %.3f, s: %.3f, t: %.3f, l: %.3f'%(
            margin_type, self.m, self.s, t, lanbuda))

    def fun_g(self, z, t: int):
        gz = 2 * torch.pow((z + 1) / 2, t) - 1
        return gz

    def forward(self, x, label):
        
        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats

        # compute similarity
        cos = F.linear(F.normalize(x), F.normalize(self.weight))

        if self.margin_type == 'A':  # arcface type (AAM)
            sin = torch.sqrt(1.0 - torch.pow(cos, 2))
            cos_m_theta_p = self.s * self.fun_g(
                torch.where(cos > self.th, cos * self.cos_m - sin * self.sin_m,
                            cos - self.mmm), self.t) + self.bias[0][0]
            cos_m_theta_n = self.s * self.fun_g(
                cos * self.cos_m + sin * self.sin_m, self.t) + self.bias[0][0]
            cos_p_theta = self.lanbuda * torch.log(
                1 + torch.exp(-1.0 * cos_m_theta_p))
            cos_n_theta = (
                1 - self.lanbuda) * torch.log(1 + torch.exp(cos_m_theta_n))
        elif self.margin_type == 'C':  # cosface type (AM)
            cos_m_theta_p = self.s * (self.fun_g(cos, self.t) -
                                          self.m) + self.bias[0][0]
            cos_m_theta_n = self.s * (self.fun_g(cos, self.t) +
                                          self.m) + self.bias[0][0]
            cos_p_theta = self.lanbuda * torch.log(
                1 + torch.exp(-1.0 * cos_m_theta_p))
            cos_n_theta = (
                1 - self.lanbuda) * torch.log(1 + torch.exp(cos_m_theta_n))

        target_mask = x.new_zeros(cos.size())
        target_mask.scatter_(1, label.view(-1, 1).long(), 1.0)
        nontarget_mask = 1 - target_mask
        cos1 = (cos - self.m) * target_mask + cos * nontarget_mask
        output = self.s * cos1  # for computing the accuracy
        loss = (target_mask * cos_p_theta +
                nontarget_mask * cos_n_theta).sum(1).mean()
        prec1   = accuracy(output.detach(), label.detach(), topk=(1,))[0]

        return loss, prec1
    
