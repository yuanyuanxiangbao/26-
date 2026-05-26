"""
问题2.1 可视化辅助函数
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sklearn import manifold

plt.rcParams.update({
    'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Heiti TC'],
    'axes.unicode_minus': False,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'font.size': 10,
})


def _setup_chinese_font():
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
    except Exception:
        pass


def plot_connectivity_graph(communities, dist_values, elderly_pop, filepath):
    n = len(communities)
    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(n):
        for j in range(i + 1, n):
            if dist_values[i][j] <= 1000:
                ax.plot([i, j], [1, 0], 'gray', lw=0.5, alpha=0.4)
    max_pop = max(elderly_pop) if max(elderly_pop) > 0 else 1
    sizes = [150 * (p / max_pop) + 30 for p in elderly_pop]
    for i, c in enumerate(communities):
        ax.scatter(i, 0.5, s=sizes[i], c='steelblue', alpha=0.7, edgecolors='k', linewidths=0.5, zorder=5)
        ax.text(i, 0.5, c, ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.1, 1.1)
    ax.set_title('小区可达关系图（距离≤1000m）')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)


def plot_dendrogram(communities, dist_values, filepath):
    try:
        import scipy.cluster.hierarchy as sch
        import scipy.spatial.distance as ssd
    except ImportError:
        return
    n = len(communities)
    dist_array = ssd.squareform(np.array(dist_values, dtype=float))
    Z = sch.linkage(dist_array, method='average')
    fig, ax = plt.subplots(figsize=(10, 5))
    sch.dendrogram(Z, labels=communities, ax=ax, color_threshold=700)
    ax.set_title('层次聚类树状图（平均距离法）')
    ax.set_xlabel('小区')
    ax.set_ylabel('距离')
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)


def plot_mds_coverage(communities, dist_values, elderly_pop, filepath):
    n = len(communities)
    mds = manifold.MDS(n_components=2, dissimilarity='precomputed', random_state=42, normalized_stress=False)
    pos = mds.fit_transform(np.array(dist_values, dtype=float))
    fig, ax = plt.subplots(figsize=(8, 7))
    max_pop = max(elderly_pop) if max(elderly_pop) > 0 else 1
    sizes = [200 * (p / max_pop) + 50 for p in elderly_pop]
    for i in range(n):
        circle = Circle(pos[i], radius=0.4, fill=False, linestyle='--', linewidth=0.5, alpha=0.3, color='gray')
        ax.add_patch(circle)
    ax.scatter(pos[:, 0], pos[:, 1], s=sizes, c='tomato', alpha=0.7, edgecolors='k', linewidths=0.5, zorder=5)
    for i, c in enumerate(communities):
        ax.text(pos[i, 0], pos[i, 1], c, ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    ax.set_title('MDS 小区空间分布与覆盖范围')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)


def plot_pareto_scatter(pareto, alphas, communities, filepath):
    fig, ax = plt.subplots(figsize=(8, 5))
    xs, ys, labels = [], [], []
    for a in alphas:
        r = pareto.get(a)
        if r is None or r['plan'] is None:
            continue
        xs.append(r['cov_ratio'] * 100)
        ys.append(r['avg_s'])
        labels.append(str(round(a, 2)))
    ax.scatter(xs, ys, c='steelblue', alpha=0.7, s=40)
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(lab, (x, y), textcoords='offset points', xytext=(5, 5), fontsize=7)
    ax.set_xlabel('服务覆盖率 (%)')
    ax.set_ylabel('平均满意度')
    ax.set_title('帕累托前沿：α权重下的覆盖率 vs 满意度')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)
