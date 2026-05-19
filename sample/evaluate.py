"""
evaluate.py — 训练后完整评估 + 可视化。

步骤:
    1. 加载最佳 checkpoint
    2. 对所有测试样本预测 → 逆变换为物理 FRF
    3. 画图验证 (多点 FRF 曲线 + 空间分布)
    4. 保存逐点 FRF 到文件

用法:
    F:\pytorch_cuda12\python.exe geometric_frf/sample/evaluate.py
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import torch, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from geometric_frf.models import build_geometric_model, GeometryData
from geometric_frf.data.dataset import GeometricHDF5Dataset

# ============ 配置 ============
CONFIG = {
    'freq_sample': False, 'freq_limit': 100, 'n_freqs': 100,
    'data_path_train': ['train.h5'], 'data_path_val': ['val.h5'],
    'data_paths_test': ['test.h5'],
}
MODEL_CFG = {
    'encoder_type': 'gnn',
    'encoder_kwargs': {'in_channels':11, 'hidden_dim':256, 'out_dim':256,
                       'coord_dim':3, 'n_layers':4, 'conv_type':'sage',
                       'use_global_pool':True, 'global_pool':'mean'},
    'decoder_kwargs': {'in_dim':256, 'n_freqs':100, 'hidden_dim':256,
                       'n_layers':4, 'chunk_size':256, 'out_dim':1,
                       'freq_encoding':'sin'},
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_dir = os.path.join(os.path.dirname(__file__), "data")
out_dir  = os.path.join(os.path.dirname(__file__), "output")
ckpt_path = os.path.join(out_dir, "checkpoint_best")

def main():
    print("=" * 60)
    print("模型评估 + 可视化")
    print("=" * 60)

    # ---- 1. 加载数据 ----
    testset = GeometricHDF5Dataset(
        ['test.h5'], CONFIG, data_dir=data_dir, normalization=True, test=True)
    # 原始数据 (用于频率轴)
    testset_raw = GeometricHDF5Dataset(
        ['test.h5'], CONFIG, data_dir=data_dir, normalization=False, test=True)
    print(f"测试集: {len(testset)} 样本")

    # ---- 2. 加载模型 ----
    model = build_geometric_model(
        MODEL_CFG['encoder_type'], MODEL_CFG['encoder_kwargs'],
        MODEL_CFG['decoder_kwargs']).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Checkpoint: epoch={ckpt['epoch']}, loss={ckpt['loss']:.6f}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params:,}")

    # ---- 3. 逐样本预测 ----
    all_preds, all_targets = [], []
    for idx in range(len(testset)):
        s_norm = testset[idx]
        s_raw  = testset_raw[idx]

        gd = GeometryData(
            points=s_norm['geometry'].points.unsqueeze(0),
            point_features=s_norm['geometry'].point_features.unsqueeze(0) \
                if s_norm['geometry'].point_features is not None else None,
        ).to(device)

        freq_norm = s_norm['frequencies'].unsqueeze(0).to(device)

        with torch.no_grad():
            pred_asinh = model(gd, freq_norm).squeeze(0).cpu()

        # 逆变换 → 物理 FRF
        pred_phys = testset.undo_normalize(pred_asinh)  # torch.sinh()
        targ_phys = testset.undo_normalize(s_norm['point_frf'])
        all_preds.append(pred_phys)
        all_targets.append(targ_phys)

    all_preds   = torch.stack(all_preds)   # (50, 240, 100)
    all_targets = torch.stack(all_targets) # (50, 240, 100)
    points_3d   = testset_raw.loaded['points'][:len(testset)]  # (50, 240, 3)
    freq_raw    = testset_raw.loaded['frequencies'][0]         # (100,) Hz
    print(f"预测完成: {all_preds.shape}")

    # ---- 4. 整体误差 ----
    mse = torch.nn.functional.mse_loss(all_preds, all_targets).item()
    l1  = torch.nn.functional.l1_loss(all_preds, all_targets).item()
    print(f"\n整体误差:  MSE={mse:.6f},  L1={l1:.6f}")

    # ---- 5. 可视化: 第0个样本, 选代表性点画 FRF ----
    sample_idx = 0
    pts = points_3d[sample_idx]  # (240, 3)
    target = all_targets[sample_idx]  # (240, 100)
    pred   = all_preds[sample_idx]    # (240, 100)

    # 沿梁长选 5 个点
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 图1: 多点 FRF
    ax = axes[0, 0]
    x_positions = [0, 0.125, 0.25, 0.375, 0.5]
    colors = plt.cm.viridis(np.linspace(0, 1, 5))
    for x_targ, c in zip(x_positions, colors):
        idx = torch.argmin((pts[:,0]-x_targ).abs()).item()
        ax.semilogx(freq_raw, target[idx], color=c, alpha=0.4, linewidth=1)
        ax.semilogx(freq_raw, pred[idx], color=c, linewidth=1.5, linestyle='--',
                   label=f'x={pts[idx,0]:.3f}')
    ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('FRF Amplitude')
    ax.set_title('FRF at 5 Points Along Beam (solid=target, dashed=pred)')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # 图2: 散点图 (预测 vs 目标)
    ax = axes[0, 1]
    ax.scatter(target.flatten()[:5000], pred.flatten()[:5000], s=1, alpha=0.3)
    ax.plot([target.min(), target.max()], [target.min(), target.max()], 'r--', linewidth=1)
    ax.set_xlabel('Target FRF'); ax.set_ylabel('Predicted FRF')
    ax.set_title(f'Prediction vs Target (MSE={mse:.6f})')
    ax.grid(True, alpha=0.3)

    # 图3: 空间分布 (共振频率处)
    ax = axes[1, 0]
    # 找全局幅值最大的频率
    peak_f = torch.argmax(target.abs().max(dim=0)[0]).item()
    resp_targ = target[:, peak_f].numpy()
    resp_pred = pred[:, peak_f].numpy()
    ax.plot(pts[:, 0], resp_targ, 'b-', alpha=0.5, linewidth=1, label='Target')
    ax.plot(pts[:, 0], resp_pred, 'r--', linewidth=1.5, label='Predicted')
    ax.set_xlabel('X coordinate (m)'); ax.set_ylabel('FRF')
    ax.set_title(f'Spatial Pattern @ {freq_raw[peak_f]:.1f} Hz')
    ax.legend(); ax.grid(True, alpha=0.3)

    # 图4: 误差分布直方图
    ax = axes[1, 1]
    errors = (pred - target).flatten().numpy()
    ax.hist(errors, bins=100, alpha=0.7, density=True)
    ax.axvline(x=0, color='r', linestyle='--')
    ax.set_xlabel('Prediction Error'); ax.set_ylabel('Density')
    ax.set_title(f'Error Distribution (std={errors.std():.4f})')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "evaluation.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图表保存: {fig_path}")

    # ---- 6. 保存逐点 FRF ----
    npz_path = os.path.join(out_dir, "final_results.npz")
    np.savez(npz_path,
             points=points_3d.numpy(),           # (50, 240, 3)
             frequencies=freq_raw.numpy(),       # (100,) Hz
             predicted_frf=all_preds.numpy(),    # (50, 240, 100)
             target_frf=all_targets.numpy(),     # (50, 240, 100)
    )
    print(f"数据保存: {npz_path}")
    print(f"\n使用方式:")
    print(f"  data = np.load('{npz_path}')")
    print(f"  data['points'][0, i]     # 第0个样本第i点的 (x,y,z) 坐标")
    print(f"  data['predicted_frf'][0, i]  # 该点的预测FRF曲线")
    print(f"  data['frequencies']      # 频率轴 (Hz)")
    print(f"\n完成!")

if __name__ == '__main__':
    main()
