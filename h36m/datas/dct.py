#!/usr/bin/env python
# encoding: utf-8
import torch
import numpy as np


def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m

def dct_transform_numpy(data, dct_m, dct_n):
    '''
    B, 60, 35
    '''
    batch_size, features, seq_len = data.shape
    data = data.reshape(-1, seq_len)  # [180077*60， 35]
    data = data.transpose(1, 0)  # [35, b*60]

    out_data = np.matmul(dct_m[:dct_n, :], data)  # [dct_n, 180077*60]
    out_data = out_data.transpose().reshape((-1, features, dct_n))  # [b, 60, dct_n]
    return out_data

def reverse_dct_numpy(dct_data, idct_m, seq_len):
    '''
    B, 60, 35
    '''
    batch_size, features, dct_n = dct_data.shape

    dct_data = dct_data.transpose(2, 0, 1).reshape((dct_n, -1))  # dct_n, B*60
    out_data = np.matmul(idct_m[:, :dct_n], dct_data).reshape((seq_len, batch_size, -1)).transpose(1, 2, 0)
    return out_data

def dct_transform_torch(data, dct_m, dct_n):
    '''
    B, 60, 35
    '''
    batch_size, features, seq_len = data.shape

    data = data.contiguous().view(-1, seq_len)  # [180077*60， 35]
    data = data.permute(1, 0)  # [35, b*60]

    out_data = torch.matmul(dct_m[:dct_n, :], data)  # [dct_n, 180077*60]
    out_data = out_data.permute(1, 0).contiguous().view(-1, features, dct_n)  # [b, 60, dct_n]
    return out_data

def reverse_dct_torch(dct_data, idct_m, seq_len):
    '''
    B, 60, 35
    '''
    batch_size, features, dct_n = dct_data.shape

    dct_data = dct_data.permute(2, 0, 1).contiguous().view(dct_n, -1)  # dct_n, B*60
    out_data = torch.matmul(idct_m[:, :dct_n], dct_data).contiguous().view(seq_len, batch_size, -1).permute(1, 2, 0)
    return out_data