#!/usr/bin/env python
# encoding: utf-8
import torch
import torch.nn as nn
from torch.nn import Module, Sequential, ModuleList, ModuleDict, Linear, GELU, Tanh, BatchNorm1d
import random
import math
import numpy as np

from .gcn_layers import GraphConv, GraphConvBlock, ResGCB

class DiverseSampling(Module):
    def __init__(self, node_n=48, hidden_dim=256, base_dim = 96, z_dim=64, dct_n=30, base_num_p1=30, dropout_rate=0):
        super(DiverseSampling, self).__init__()
        self.z_dim = z_dim
        self.base_dim = base_dim
        self.base_num_p1 = base_num_p1

        self.condition_enc = Sequential(
            GraphConvBlock(in_len=dct_n, out_len=hidden_dim, in_node_n=node_n, out_node_n=node_n,
                           dropout_rate=dropout_rate, bias=True, residual=False),
            ResGCB(in_len=hidden_dim, out_len=hidden_dim, in_node_n=node_n, out_node_n=node_n,
                   dropout_rate=dropout_rate, bias=True, residual=True),
            ResGCB(in_len=hidden_dim, out_len=hidden_dim, in_node_n=node_n, out_node_n=node_n,
                   dropout_rate=dropout_rate, bias=True, residual=True)
        )
        
        self.bases_p1 = Sequential(
            Linear(node_n * hidden_dim, self.base_num_p1 * self.base_dim),
            BatchNorm1d(self.base_num_p1 * self.base_dim),
            Tanh()
        )

        self.mean_p1 = Sequential(
            Linear(self.base_dim, 64),
            BatchNorm1d(64),
            Tanh(),
            Linear(64, self.z_dim)
        )
        self.logvar_p1 = Sequential(
            Linear(self.base_dim, 64),
            BatchNorm1d(64),
            Tanh(),
            Linear(64, self.z_dim)
        )

    def forward(self, condition, repeated_eps=None, many_weights=None, multi_modal_head=10):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        b, v, ct = condition.shape
        n1 = self.base_num_p1
        n2 = 120
        _x = condition

        condition_enced = self.condition_enc(condition)
        bases1 = self.bases_p1(condition_enced.view(b, -1)).view(b, self.base_num_p1, self.base_dim)

        mean_condition = torch.mean(_x, dim=(0, 2), keepdim=True)
        std_condition = torch.std(_x, dim=(0, 2), keepdim=True)
        _x = (_x - mean_condition)/std_condition

        U, S, V = torch.svd(_x, some=True)
        Chat = U[:, :, 0:n1]
        W1 = torch.diag_embed(S)
        V = V[:, :, 0:n1]
        V1 = V.permute(0, 2, 1)
        Xhat = torch.matmul(W1[:, 0:n1, 0:n1], V1)
        t1 = Xhat[:, :, 1:ct]
        t2 = torch.linalg.pinv(Xhat[:, :, 0:ct - 1])
        Ahat = torch.matmul(t1, t2)

        O = torch.zeros(b, n1, 2 * v)
        tt = Chat
        for i in range(2):
            t = tt.permute(0, 2, 1)
            O[:, :, i * v:(i + 1) * v] = t
            tt = torch.matmul(tt, Ahat)

        O = O.permute(0, 2, 1)
        P, R, Q = torch.svd(O, some=True)
        S = P[:, :, :n1].to(device)
        bases2 = S.permute(0, 2, 1)
        bases1 = bases1.to(device)
        bases = bases2 + bases1
        Q1, R1 = torch.linalg.qr(bases)

        Q1 = torch.repeat_interleave(Q1, repeats=multi_modal_head, dim=0).to(device)
        R1 = torch.repeat_interleave(R1, repeats=multi_modal_head, dim=0).to(device)
        many_weights = many_weights.permute(0, 2, 1).to(device)
        z, _, _ = torch.svd(many_weights, some=True)
        z = z.permute(0, 2, 1)
        samples = torch.matmul(z, Q1)
        many_bases_blending = torch.matmul(samples, R1).squeeze(dim=1).view(-1, self.base_dim)

        all_mean = self.mean_p1(many_bases_blending)
        all_logvar = self.logvar_p1(many_bases_blending)

        all_z = torch.exp(0.5 * all_logvar) * repeated_eps + all_mean

        return all_z, all_mean, all_logvar


if __name__ == '__main__':
    m = DiverseSampling(node_n=16, hidden_dim=256, base_dim=128, z_dim=128, dct_n=10, base_num_p1=10, dropout_rate=0).cuda()
    print(f"{sum(p.numel() for p in m.parameters()) / 1e6}")

    # >>> many bases
    logtics = torch.ones((4*100, 1, 10), device="cuda:0") / 10  # b*h, 1, 10
    many_weights = m._sample_weight_gumbel_softmax(logtics, temperature=1)  # b*h, 1, 10

    pass



