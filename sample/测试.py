"""
查看生成的 FRF 数据真实值长什么样。
用法: F:\pytorch_cuda12\python.exe geometric_frf/sample/测试.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import torch, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from geometric_frf.data.dataset import GeometricHDF5Dataset

data_dir = os.path.join(os.path.dirname(__file__), "data")
out_dir  = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(out_dir, exist_ok=True)

# 加载原始数据 (不归一化)
CONFIG = {'freq_sample': False, 'freq_limit': 100, 'n_freqs': 100}
ds = GeometricHDF5Dataset(['train.h5'], CONFIG, data_dir=data_dir, normalization=False, test=True)
sample = ds[0]

points_3d = sample['geometry'].points       # (240, 3)
freq_raw  = sample['frequencies']           # (100,) Hz
frf_raw   = sample['point_frf']             # (240, 100) 真实物理 FRF

print(f'节点数: {points_3d.shape[0]}, 频率数: {len(freq_raw)}')
print(f'频率范围: [{freq_raw[0]:.1f}, {freq_raw[-1]:.1f}] Hz')
print(f'FRF 范围: [{frf_raw.min():.6f}, {frf_raw.max():.6f}]')
print(f'FRF 均值: {frf_raw.mean():.6f}, 标准差: {frf_raw.std():.6f}')

# 沿梁长选 5 个位置画 FRF
fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)

x_positions = [0, 0.125, 0.25, 0.375, 0.5]
labels = [u'固定端 (x=0)', 'x=0.125', u'中点 (x=0.25)', 'x=0.375', u'自由端 (x=0.5)']

for ax, x_targ, label in zip(axes, x_positions, labels):
    idx = torch.argmin((points_3d[:,0] - x_targ).abs()).item()
    x, y, z = points_3d[idx].tolist()
    f_data = frf_raw[idx].numpy()

    ax.semilogx(freq_raw, f_data, 'b-', linewidth=1.5)
    ax.set_ylabel('FRF Amplitude')
    # 在标题显示幅值范围
    is_fixed = (x_targ == 0)
    note = ' (必为零+噪声)' if is_fixed else ''
    ax.set_title(f'{label}  x={x:.3f}, y={y:.3f}, z={z:.3f}  [range:{f_data.min():.2e}, {f_data.max():.2e}]{note}')
    ax.grid(True, alpha=0.3)

    # 固定端: 不用标注共振峰 (都是噪声)
    if not is_fixed:
        for i in range(1, len(f_data)-1):
            if abs(f_data[i]) > abs(f_data[i-1]) and abs(f_data[i]) > abs(f_data[i+1]) and abs(f_data[i]) > abs(f_data).std() * 3:
                ax.axvline(x=freq_raw[i], color='red', linestyle=':', alpha=0.4)

    # 固定端: 画零线参考
    if is_fixed:
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

axes[-1].set_xlabel('Frequency (Hz)')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'true_frf.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'图表保存: {out_dir}/true_frf.png')

# 空间分布热力图 (某一共振频率)
peak_global = torch.argmax(frf_raw.abs().max(dim=0)[0]).item()
fig2, ax2 = plt.subplots(figsize=(10, 6))
sc = ax2.scatter(points_3d[:,0], points_3d[:,1], c=frf_raw[:, peak_global].numpy(),
                 cmap='RdBu_r', s=30)
ax2.set_xlabel('X (m)'); ax2.set_ylabel('Y (m)')
ax2.set_title(f'Spatial Pattern @ {freq_raw[peak_global]:.1f} Hz')
plt.colorbar(sc, ax=ax2)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'spatial_pattern.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'图表保存: {out_dir}/spatial_pattern.png')
