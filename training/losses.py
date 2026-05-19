"""
losses.py — 频响函数 (FRF) 专用损失函数。

使用 Huber Loss (SmoothL1) 替代 MSE:
    - 误差小时为平方误差 (收敛平滑)
    - 误差大时退化为线性误差 (防梯度爆炸)
    特别适合 FRF 这种动态范围极大的数据。

同时支持共振峰自适应加权。
"""

import torch
import torch.nn.functional as F


def _huber(error, beta=1.0):
    """Huber 损失: |e|<=beta 用平方, |e|>beta 用线性"""
    abs_e = torch.abs(error)
    return torch.where(abs_e <= beta,
                       0.5 * error ** 2,
                       beta * (abs_e - 0.5 * beta))


def weighted_huber_loss(prediction, target, alpha=5.0, beta=1.0):
    """
    共振峰自适应加权 Huber Loss。

    weight = 1.0 + alpha * |target|  — 共振峰权重大
    内部用 Huber: 小误差平方收敛, 大误差线性防爆炸
    """
    weight = 1.0 + alpha * torch.abs(target)
    huber = _huber(prediction - target, beta)
    return torch.mean(weight * huber)


def complex_frf_loss(prediction, target, amp_weight=2.0, alpha=0.0, beta=1.0, eps=1e-8):
    """
    复数 FRF 损失 —— Huber 版。

    out_dim=2: loss = huber(Re) + huber(Im) + amp_weight * huber(|pred|, |target|)
    """
    pred_re, pred_im = prediction[..., 0], prediction[..., 1]
    target_re, target_im = target[..., 0], target[..., 1]

    pred_amp = torch.sqrt(pred_re ** 2 + pred_im ** 2 + eps)
    target_amp = torch.sqrt(target_re ** 2 + target_im ** 2 + eps)

    if alpha > 0:
        weight = 1.0 + alpha * target_amp
        loss_re = torch.mean(weight * _huber(pred_re - target_re, beta))
        loss_im = torch.mean(weight * _huber(pred_im - target_im, beta))
        loss_amp = torch.mean(weight * _huber(pred_amp - target_amp, beta))
    else:
        loss_re = F.smooth_l1_loss(pred_re, target_re, beta=beta)
        loss_im = F.smooth_l1_loss(pred_im, target_im, beta=beta)
        loss_amp = F.smooth_l1_loss(pred_amp, target_amp, beta=beta)

    return loss_re + loss_im + amp_weight * loss_amp


def frf_loss(prediction, target, out_dim=1, resonance_weight=5.0, amp_weight=2.0, beta=1.0):
    """
    FRF 统一损失函数 (Huber 版)。

    参数:
        out_dim:          1=振幅, 2=复数
        resonance_weight: 共振峰加权系数 (0=等权)
        amp_weight:       振幅约束权重 (仅 out_dim=2)
        beta:             Huber 转折点 (默认 1.0)
    """
    if out_dim == 2:
        return complex_frf_loss(prediction, target,
                               amp_weight=amp_weight, alpha=resonance_weight, beta=beta)
    else:
        if resonance_weight > 0:
            return weighted_huber_loss(prediction, target, alpha=resonance_weight, beta=beta)
        else:
            return F.smooth_l1_loss(prediction, target, beta=beta)
