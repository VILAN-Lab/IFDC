import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class Ban(nn.Module):
    def __init__(self):
        super(Ban, self).__init__()
        self.mid_dim = 1024
        self.v_dim = 2048
        self.s_dim = 300
        self.glimpses = 2

        self.biattention = weight_norm(BiAttention(v_dim=self.v_dim,
                                                   s_dim=self.s_dim,
                                                   mid_dim=self.mid_dim,
                                                   glimpses=self.glimpses,
                                                   drop=0.0), name="h_weight", dim=None)

        self.applyattention = ApplyAttention(v_dim=self.v_dim,
                                             s_dim=self.s_dim,
                                             mid_dim=self.mid_dim,
                                             glimpses=self.glimpses,
                                             drop=0.0)

    def forward(self, v, s):

        num_obj = v.size(1)
        att, logit = self.biattention(v, s)
        new_v = self.applyattention(v, s, att, logit)

        return new_v.view(-1, num_obj, self.v_dim)

class BiAttention(nn.Module):
    def __init__(self, v_dim, s_dim, mid_dim, glimpses, drop=0.0):
        super(BiAttention, self).__init__()

        self.hidden_aug = 2
        self.glimpses = glimpses
        self.lin_v = FCNet(in_size=v_dim,
                           out_size=int(mid_dim * self.hidden_aug),
                           activate='relu',
                           drop=drop/2.5)
        self.lin_S = FCNet(s_dim, int(mid_dim * self.hidden_aug), activate='relu', drop=drop/2.5)

        self.h_weight = nn.Parameter(torch.Tensor(1, glimpses, 1, int(mid_dim * self.hidden_aug)).normal_())
        self.h_bias = nn.Parameter(torch.Tensor(1, glimpses, 1, 1).normal_())

        self.drop = nn.Dropout(drop)

    def forward(self, v, s):

        v = v.unsqueeze(2).view(-1, 1, 2048)
        s = s.unsqueeze(2).view(-1, 1, 300)

        v_num = v.size(1)
        s_num = s.size(1)

        v_ = self.lin_v(v).unsqueeze(1)
        s_ = self.lin_S(s).unsqueeze(1)
        s_ = self.drop(s_)

        h_ = s_ * self.h_weight
        logit = torch.matmul(h_, v_.transpose(2, 3))
        logit = logit + self.h_bias

        atten = F.softmax(logit.view(-1, self.glimpses, v_num * s_num), 2)

        return atten.view(-1, self.glimpses, v_num, s_num), logit

class ApplyAttention(nn.Module):
    def __init__(self, v_dim, s_dim, mid_dim, glimpses, drop=0.0):
        super(ApplyAttention, self).__init__()
        self.glimpses = glimpses
        layers = []
        for g in range(self.glimpses):
            layers.append(ApplySingleAttention(v_dim, s_dim, mid_dim, drop))
        self.glimpses_layers = nn.ModuleList(layers)

    def forward(self, v, s, atten, logit):

        v = v.unsqueeze(2).view(-1, 1, 2048)
        s = s.unsqueeze(2).view(-1, 1, 300)

        for g in range(self.glimpses):
            atten_h = self.glimpses_layers[g](v, s, atten[:, g, :, :], logit[:, g, :, :])
            v = v + atten_h

        return v.sum(1)

class ApplySingleAttention(nn.Module):
    def __init__(self, v_dim, s_dim, mid_dim, drop=0.0):
        super(ApplySingleAttention, self).__init__()
        self.lin_v = FCNet(v_dim, mid_dim, activate="relu", drop=drop)
        self.lin_s = FCNet(s_dim, mid_dim, activate="relu", drop=drop)
        self.lin_atten = FCNet(mid_dim, v_dim, drop=drop)

    def forward(self, v, s, atten, logit):

        v_ = self.lin_v(v).transpose(1, 2).unsqueeze(3)
        s_ = self.lin_s(s).transpose(1, 2).unsqueeze(2)

        s_ = torch.matmul(s_, atten.unsqueeze(1))

        h_ = torch.matmul(s_, v_)

        h_ = h_.squeeze(3).squeeze(2)

        atten_h = self.lin_atten(h_.unsqueeze(1))

        return atten_h

class FCNet(nn.Module):
    def __init__(self, in_size, out_size, activate=None, drop=0.0):
        super(FCNet, self).__init__()
        self.lin = weight_norm(nn.Linear(in_size, out_size), dim=None)
        self.drop_value = drop
        self.drop = nn.Dropout(drop)

        self.activate = activate.lower() if (activate is not None) else None
        if activate == "relu":
            self.ac_fn = nn.ReLU()
        elif activate == "sigmoid":
            self.ac_fn = nn.Sigmoid()
        elif activate == "tanh":
            self.ac_fn = nn.Tanh()

    def forward(self, x):
        if self.drop_value > 0:
            x = self.drop(x)

        x = self.lin(x)

        if self.activate is not None:
            x = self.ac_fn(x)

        return x

