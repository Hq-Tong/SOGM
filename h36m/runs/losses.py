#!/usr/bin/env python
# encoding: utf-8
import math
import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform
from ..datas import h36m_valid_angle_check_torch

def loss_kl_normal(mu, logvar):

    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.shape[0]
    return kl

def loss_recons(gt, pred):

    MSE = (gt - pred).pow(2).sum() / pred.shape[0]
    return MSE

def loss_recover_history_t1(pred, gt):

    loss = torch.mean((pred - gt).pow(2).sum(dim=1))
    return loss

def loss_recover_history_t2(pred, gt):

    loss = torch.mean((pred - gt.unsqueeze(dim=1)).pow(2).sum(dim=2))
    return loss


def loss_diversity_hinge(pred, minthreshold):
    b, h, vc, t = pred.shape
    triu_indices = np.triu_indices(h, k=1)
    idx = np.arange(h * h).reshape(h, h)[triu_indices]

    pred = pred.reshape(b, h, -1)
    dist = torch.norm(pred[:, :, None, :] - pred[:, None, :, :], p=2, dim=-1)

    dist = torch.relu(minthreshold - dist)
    loss = dist.view(b, -1)[:, idx]
    return loss


def loss_diversity_hinge_divide(outputs, seperate_head, minthreshold):
    b, h, vc, t = outputs.shape
    all_hinges = []
    for oidx in range(h // seperate_head):
        all_hinges.append(loss_diversity_hinge(outputs[:, oidx * seperate_head:(oidx + 1) * seperate_head, :, :], minthreshold=minthreshold))  # [10, 48, 100], [1, 48, 100]
        for ojdx in range(oidx + 1, h // seperate_head):
            all_hinges.append(loss_diversity_hinge_between_two_part(outputs[:, oidx * seperate_head:(oidx + 1) * seperate_head, :, :], outputs[:, ojdx * seperate_head:(ojdx + 1) * seperate_head, :, :],
                                                                    minthreshold=minthreshold))
    all_hinges = torch.cat(all_hinges, dim=-1).mean(dim=-1).mean()
    return all_hinges


def loss_diversity_hinge_between_two_part(part_a, part_b, minthreshold=25):
    b, h, vc, t = part_a.shape
    part_a = part_a.reshape(b, h, -1)
    part_b = part_b.reshape(b, h, -1)

    dist = torch.norm(part_a[:, :, None, :] - part_b[:, None, :, :], p=2, dim=-1)
    dist = torch.relu(minthreshold - dist)
    loss = dist.view(b, -1)
    return loss

def loss_limb_length_t1(pred, gt, parent_17):

    b, vc, t = pred.shape

    gt = gt[:, :, 0].reshape([b, -1, 3])
    gt = torch.cat((torch.zeros_like(gt[:, 0:1, :]), gt), dim=1)
    limb_gt = torch.norm(gt[:, 1:, :] - gt[:, parent_17[1:], :], dim=2)[:, :, None]

    pred = pred.reshape([b, -1, 3, t])
    pred = torch.cat((torch.zeros_like(pred[:, 0:1, :, :]), pred), dim=1)
    limb_pred = torch.norm(pred[:, 1:, :, :] - pred[:, parent_17[1:], :, :], dim=2)

    loss_limb = torch.mean((limb_gt - limb_pred).pow(2).sum(dim=1))
    return loss_limb

def loss_limb_length_t2(pred, gt, parent_17):

    b, h, vc, t = pred.shape

    gt = gt[:, :, 0].reshape([b, -1, 3])
    gt = torch.cat((torch.zeros_like(gt[:, 0:1, :]), gt), dim=1)
    limb_gt = torch.norm(gt[:, 1:, :] - gt[:, parent_17[1:], :], dim=2)[:, None, :, None]

    pred = pred.reshape([b, h, -1, 3, t])
    pred = torch.cat((torch.zeros_like(pred[:, :, 0:1, :, :]), pred), dim=2)
    limb_pred = torch.norm(pred[:, :, 1:, :, :] - pred[:, :, parent_17[1:], :, :], dim=3)

    loss_limb = torch.mean((limb_gt - limb_pred).pow(2).sum(dim=2))
    return loss_limb


def loss_recons_adelike(pred, gt):

    b, h, vc, t = pred.shape
    diff = pred - gt.unsqueeze(1)
    dist = diff.pow(2).sum(dim=-2).sum(dim=-1)
    loss_recon_ade = dist.min(dim=1)[0].mean()

    return loss_recon_ade


def loss_recons_mmadelike(pred, similars):

    b, h, vc, t = pred.shape
    diff = pred[:, :, None, :, :] - similars[:, None, :, :, :]
    mask = similars.abs().sum(3).sum(-1) > 1e-6
    dist = diff.pow(2).sum(dim=-2).sum(dim=-1)
    loss_recon_multi = dist.min(dim=1)[0][mask].mean()
    if torch.isnan(loss_recon_multi):
        loss_recon_multi = torch.zeros(1, device=pred.device)
    return loss_recon_multi

def loss_valid_angle_t1(pred, valid_ang, data="h36m"):

    b, vc, t = pred.shape

    ang_names = list(valid_ang.keys())

    pred = pred.permute(0, 2, 1).contiguous().view(-1, vc)
    if data == "h36m":
        ang_cos = h36m_valid_angle_check_torch(pred)

    loss = torch.tensor(0, dtype=pred.dtype, device=pred.device)

    for an in ang_names:
        lower_bound = valid_ang[an][0]
        if lower_bound >= -0.98:
            if torch.any(ang_cos[an] < lower_bound):
                loss += (ang_cos[an][ang_cos[an] < lower_bound] - lower_bound).pow(2).mean()
        upper_bound = valid_ang[an][1]
        if upper_bound <= 0.98:
            if torch.any(ang_cos[an] > upper_bound):
                loss += (ang_cos[an][ang_cos[an] > upper_bound] - upper_bound).pow(2).mean()
    return loss

def loss_valid_angle_t2(pred, valid_ang, data="h36m"):

    b, h, vc, t = pred.shape

    ang_names = list(valid_ang.keys())

    pred = pred.permute(0, 1, 3, 2).contiguous().view(-1, vc)
    if data == "h36m":
        ang_cos = h36m_valid_angle_check_torch(pred)

    loss = torch.tensor(0, dtype=pred.dtype, device=pred.device)

    for an in ang_names:
        lower_bound = valid_ang[an][0]
        if lower_bound >= -0.98:
            if torch.any(ang_cos[an] < lower_bound):
                loss += (ang_cos[an][ang_cos[an] < lower_bound] - lower_bound).pow(2).mean()
        upper_bound = valid_ang[an][1]
        if upper_bound <= 0.98:
            if torch.any(ang_cos[an] > upper_bound):
                loss += (ang_cos[an][ang_cos[an] > upper_bound] - upper_bound).pow(2).mean()
    return loss


def loss_diversity_hinge_v2_likedlow(pred, minthreshold=100, scale=50):
    b, h, vc, t = pred.shape
    triu_indices = np.triu_indices(h, k=1)
    idx = np.arange(h * h).reshape(h, h)[triu_indices]

    pred = pred.reshape(b, h, -1)
    dist = torch.norm(pred[:, :, None, :] - pred[:, None, :, :], p=2, dim=-1).square()
    dist = torch.relu(minthreshold - dist)
    dist = (dist / scale).exp()
    loss = dist.view(b, -1)[:, idx]
    return loss


def loss_diversity_hinge_between_two_part_v2_likedlow(part_a, part_b, minthreshold=100, scale=50):
    b, h, vc, t = part_a.shape
    part_a = part_a.reshape(b, h, -1)
    part_b = part_b.reshape(b, h, -1)

    dist = torch.norm(part_a[:, :, None, :] - part_b[:, None, :, :], p=2, dim=-1).square()
    dist = torch.relu(minthreshold - dist)
    dist = (dist / scale).exp()
    loss = dist.view(b, -1)
    return loss

def loss_diversity_v2_likedlow(pred, scale=50):
    b, h, vc, t = pred.shape
    triu_indices = np.triu_indices(h, k=1)
    idx = np.arange(h * h).reshape(h, h)[triu_indices]

    pred = pred.reshape(b, h, -1)
    dist = torch.norm(pred[:, :, None, :] - pred[:, None, :, :], p=2, dim=-1).square()
    dist = (-dist / scale).exp()
    loss = dist.view(b, -1)[:, idx]
    return loss


def loss_diversity_two_part_v2_likedlow(part_a, part_b, scale=50):
    b, h, vc, t = part_a.shape
    part_a = part_a.reshape(b, h, -1)
    part_b = part_b.reshape(b, h, -1)

    dist = torch.norm(part_a[:, :, None, :] - part_b[:, None, :, :], p=2, dim=-1).square()
    dist = (-dist / scale).exp()
    loss = dist.view(b, -1)
    return loss

# --------------------------------------------

"""metrics"""

def compute_diversity(pred):
    h, vc, t = pred.shape
    triu_indices = np.triu_indices(h, k=1)

    pred = pred.reshape(h, -1)
    div = torch.norm(pred[:, None, :] - pred[None, :, :], p=2, dim=2) #
    div = div[triu_indices]
    # div = div.mean()
    return div


def compute_diversity_between_twopart(part_a, part_b):
    assert part_a.shape == part_b.shape
    h, vc, t = part_a.shape

    part_a = part_a.reshape(h, -1)
    part_b = part_b.reshape(h, -1)
    div = torch.norm(part_a[:, None, :] - part_b[None, :, :], p=2, dim=2).view(-1)
    # div = div.mean()
    return div


def compute_ade(pred, gt):

    diff = pred - gt
    dist = torch.norm(diff, dim=1).mean(dim=-1)
    return dist.min()


def compute_fde(pred, gt):

    h, vc, t = pred.shape

    diff = pred - gt
    dist = torch.norm(diff, dim=1)[:, -1]
    return dist.min()


def compute_mmade(pred, gt, gt_multi):
    h, vc, t = pred.shape
    dist = compute_ade(pred, gt)
    if not gt_multi.shape[0] == 0:
        all_results = torch.zeros(1+gt_multi.shape[0], device=pred.device)
        all_results[0] = dist
        next_idx = 1
        for gt_multi_i in gt_multi:
            dist = compute_ade(pred, gt_multi_i[None, ...])
            all_results[next_idx] = dist
            next_idx += 1

        mmade = all_results.mean()
    else:
        mmade = dist
    return mmade


def compute_mmfde(pred, gt, gt_multi):
    h, vc, t = pred.shape

    dist = compute_fde(pred, gt)
    if not gt_multi.shape[0] == 0:
        all_results = torch.zeros(1+gt_multi.shape[0], device=pred.device)
        all_results[0] = dist
        next_idx = 1
        for gt_multi_i in gt_multi:
            dist = compute_fde(pred, gt_multi_i[None, ...])
            all_results[next_idx] = dist
            next_idx += 1

        mmfde = all_results.mean()
    else:
        mmfde = dist
    return mmfde

def compute_bone_percent_error(gt_pose, motions, parent_17):

    n, v, c, t = motions.shape
    gt_pose = torch.cat((torch.zeros_like(gt_pose[:, 0:1, :]), gt_pose), dim=1)
    gt_bone = torch.norm(gt_pose[:, 1:, :] - gt_pose[:, parent_17[1:], :], p=2, dim=2)[..., None]

    motions = torch.cat((torch.zeros_like(motions[:, 0:1, :, :]), motions), dim=1)
    motions_bone = torch.norm(motions[:, 1:, :, :] - motions[:, parent_17[1:], :, :], p=2, dim=2)

    diff = torch.abs(motions_bone - gt_bone) / gt_bone
    avg_bone_error = torch.mean(diff)
    min_bone_error = torch.min(diff.mean(dim=[-1, -2]), dim=0)[0]
    max_bone_error = torch.max(diff.mean(dim=[-1, -2]), dim=0)[0]
    return avg_bone_error, min_bone_error, max_bone_error

def compute_angle_error(pred, valid_ang, data="h36m"):

    h, vc, t = pred.shape

    ang_names = list(valid_ang.keys())

    pred = pred.permute(0, 2, 1).contiguous().view(-1, vc)  # n*100, 48
    if data == "h36m":
        ang_cos = h36m_valid_angle_check_torch(pred)

    loss = torch.tensor(0, dtype=pred.dtype, device=pred.device)
    for an in ang_names:
        lower_bound = valid_ang[an][0]
        if lower_bound >= -0.98:
            if torch.any(ang_cos[an] < lower_bound):
                loss += (ang_cos[an][ang_cos[an] < lower_bound] - lower_bound).pow(2).mean()
        upper_bound = valid_ang[an][1]
        if upper_bound <= 0.98:
            if torch.any(ang_cos[an] > upper_bound):
                loss += (ang_cos[an][ang_cos[an] > upper_bound] - upper_bound).pow(2).mean()
    return loss
