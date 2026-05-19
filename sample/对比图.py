"""
快速对比图：指定样本、坐标，画预测 vs 真实 FRF。
第1张为选中点，剩余5张为附近随机点。

修改配置后运行:
    F:\pytorch_cuda12\python.exe geometric_frf/sample/对比图.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

# ============ 修改这里 ============
SAMPLE_IDX  = 0
POINT_X     = 0.25
POINT_Y     = 0.02
POINT_Z     = 0.01
NEAR_RANGE  = 0.02    # 附近点的随机偏移范围 (米)
FREQ_MIN    = 1
FREQ_MAX    = 500
# ================================

data = np.load(os.path.join(os.path.dirname(__file__), 'output', 'final_results.npz'))
freq_axis = data['frequencies']
points    = data['points'][SAMPLE_IDX]
true_frf  = data['target_frf'][SAMPLE_IDX]
pred_frf  = data['predicted_frf'][SAMPLE_IDX]

print(f'X range: [{points[:,0].min():.3f}, {points[:,0].max():.3f}]')
print(f'Y range: [{points[:,1].min():.4f}, {points[:,1].max():.4f}]')
print(f'Z range: [{points[:,2].min():.4f}, {points[:,2].max():.4f}]')
print(f'Frequency: [{freq_axis[0]:.1f}, {freq_axis[-1]:.1f}] Hz')
print()

freq_mask = (freq_axis >= FREQ_MIN) & (freq_axis <= FREQ_MAX)
f = freq_axis[freq_mask]

# 第1个: 用户选中点
idx_main = np.argmin((points[:,0]-POINT_X)**2 + (points[:,1]-POINT_Y)**2 + (points[:,2]-POINT_Z)**2)

# 第2-6个: 附近随机点
rng = np.random.RandomState(42)
near_indices = []
x0, y0, z0 = points[idx_main]
for _ in range(5):
    tx = x0 + rng.uniform(-NEAR_RANGE, NEAR_RANGE)
    ty = y0 + rng.uniform(-NEAR_RANGE, NEAR_RANGE)
    tz = z0 + rng.uniform(-NEAR_RANGE, NEAR_RANGE)
    ni = np.argmin((points[:,0]-tx)**2 + (points[:,1]-ty)**2 + (points[:,2]-tz)**2)
    near_indices.append(ni)

all_indices = [idx_main] + near_indices

fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=True)
for ax, i, tag in zip(axes.flat, all_indices,
                       ['SELECTED'] + [f'Near #{k+1}' for k in range(5)]):
    x, y, z = points[i]
    ax.semilogx(f, true_frf[i][freq_mask], 'b-', linewidth=1.5, alpha=0.7, label='Target')
    ax.semilogx(f, pred_frf[i][freq_mask], 'r--', linewidth=1.5, label='Predicted')
    ax.set_title(f'{tag}  (x={x:.4f}, y={y:.4f}, z={z:.4f})')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_ylabel('FRF')

axes[-1,0].set_xlabel('Frequency (Hz)'); axes[-1,1].set_xlabel('Frequency (Hz)')
plt.tight_layout()

out = os.path.join(os.path.dirname(__file__), 'output', '对比图.png')
plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
print(f'Done: {out}')
print(f'Main  point: ({points[idx_main,0]:.4f}, {points[idx_main,1]:.4f}, {points[idx_main,2]:.4f})')
for k, ni in enumerate(near_indices):
    print(f'Near #{k+1}: ({points[ni,0]:.4f}, {points[ni,1]:.4f}, {points[ni,2]:.4f})')
