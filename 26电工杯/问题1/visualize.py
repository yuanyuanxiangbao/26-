"""
问题1可视化：贴合题目要求的图表集
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.append(os.path.join(os.path.dirname(__file__), '1.1', 'code'))
from problem1_1 import read_data, predict_population

BASE_DIR = os.path.join(os.path.dirname(__file__), '..', '题目')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.sans-serif': ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC'],
    'axes.unicode_minus': False,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COMMUNITIES = list('ABCDEFGHIJ')
SERVICES = ['助餐', '日间照料', '上门护理', '康复理疗', '助浴', '紧急救助']
TYPES = ['自理', '半失能', '失能']
TYPE_COLORS = {'自理': '#4C72B0', '半失能': '#DD8452', '失能': '#C44E52'}
SVC_COLORS = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD']


def load_pop_data():
    df_pop, p1, p2 = read_data()
    pop_results = predict_population(df_pop, p1, p2, years=5)
    return pop_results


def load_theoretical_demand():
    path = os.path.join(os.path.dirname(__file__), '1.2', 'data', 'theoretical_demand.xlsx')
    df_summary = pd.read_excel(path, sheet_name='汇总表')
    df_detail = pd.read_excel(path, sheet_name='明细表')
    return df_summary, df_detail


def load_actual_demand():
    path = os.path.join(os.path.dirname(__file__), '1.3', 'data', 'actual_demand.xlsx')
    df_detail = pd.read_excel(path, sheet_name='明细表')
    df_summary = pd.read_excel(path, sheet_name='汇总表')
    return df_detail, df_summary


# 图1: 全区域各类型老人数量逐年变化（堆叠面积图）
def plot_region_pop_structure(pop_results):
    years = list(range(6))
    totals = {t: [] for t in TYPES}
    for yr in range(6):
        for t in TYPES:
            idx = TYPES.index(t) + 1
            s = sum(pop_results[name][yr][idx] for name in COMMUNITIES)
            totals[t].append(s)

    fig, ax = plt.subplots(figsize=(9, 5))
    vals = np.array([totals[t] for t in TYPES])
    ax.stackplot(years, vals, labels=TYPES, colors=[TYPE_COLORS[t] for t in TYPES], alpha=0.85)
    ax.set_xlabel('年份')
    ax.set_ylabel('老人数量（人）')
    ax.set_title('全区域各类型老人数量变化趋势')
    ax.set_xticks(years)
    ax.set_xticklabels([f'第{y}年' for y in years])
    ax.legend(loc='upper left', frameon=False)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig1_region_pop_structure.png'))
    plt.close(fig)
    print('图1已保存: 全区域各类型老人数量变化')


# 图2: 第5年各小区老人结构（堆叠柱状图）
def plot_community_structure(pop_results):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(COMMUNITIES))
    bar_w = 0.6
    bottom = np.zeros(len(COMMUNITIES))
    for t in TYPES:
        idx = TYPES.index(t) + 1
        vals = [pop_results[name][5][idx] for name in COMMUNITIES]
        ax.bar(x, vals, bar_w, bottom=bottom, label=t, color=TYPE_COLORS[t], alpha=0.85)
        bottom += vals
    ax.set_xlabel('小区')
    ax.set_ylabel('老人数量（人）')
    ax.set_title('第5年末各小区老人结构')
    ax.set_xticks(x)
    ax.set_xticklabels(COMMUNITIES)
    ax.legend(loc='upper right', frameon=False)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig2_community_structure.png'))
    plt.close(fig)
    print('图2已保存: 第5年各小区老人结构')


# 图3: 第5年全区域理论需求按老人类型构成（堆叠柱状图×服务）
def plot_demand_by_type(theo_detail):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(SERVICES))
    bar_w = 0.6
    bottom = np.zeros(len(SERVICES))
    for t in TYPES:
        vals = []
        for svc in SERVICES:
            col = f'{svc}_{t}'
            vals.append(theo_detail[col].sum())
        ax.bar(x, vals, bar_w, bottom=bottom, label=t, color=TYPE_COLORS[t], alpha=0.85)
        bottom += vals
    ax.set_xlabel('服务项目')
    ax.set_ylabel('月需求次数（次/月）')
    ax.set_title('第5年末全区域理论月需求构成（按老人类型）')
    ax.set_xticks(x)
    ax.set_xticklabels(SERVICES)
    ax.legend(loc='upper right', frameon=False)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig3_demand_by_type.png'))
    plt.close(fig)
    print('图3已保存: 理论需求按老人类型构成')


# 图4: 各小区理论需求按老人类型分面（分组柱状图）
def plot_demand_by_community(theo_detail):
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    axes = axes.flatten()
    for ci, name in enumerate(COMMUNITIES):
        ax = axes[ci]
        sub = theo_detail[theo_detail['小区'] == name]
        x = np.arange(len(SERVICES))
        bar_w = 0.25
        for ti, t in enumerate(TYPES):
            vals = [sub[f'{svc}_{t}'].values[0] for svc in SERVICES]
            ax.bar(x + ti * bar_w, vals, bar_w, label=t, color=TYPE_COLORS[t], alpha=0.85)
        ax.set_title(f'小区{name}')
        ax.set_xticks(x + bar_w)
        ax.set_xticklabels(SERVICES, rotation=30, ha='right', fontsize=7)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
        ax.grid(axis='y', alpha=0.3)
        if ci == 0:
            ax.legend(loc='upper left', frameon=False, fontsize=7)
    fig.suptitle('第5年末各小区理论月需求（按老人类型）', fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig4_demand_by_community.png'))
    plt.close(fig)
    print('图4已保存: 各小区理论需求分面')


# 图5: 消费约束缩减系数热力图（小区×类型）
def plot_reduction_heatmap(pop_results, actual_detail):
    fig, ax = plt.subplots(figsize=(8, 6))
    mat = np.zeros((len(COMMUNITIES), len(TYPES)))
    for ci, name in enumerate(COMMUNITIES):
        for ti, t in enumerate(TYPES):
            theo_sub = load_theoretical_demand()[1]
            theo_val = theo_sub[theo_sub['小区'] == name][[f'{svc}_{t}' for svc in SERVICES]].sum().sum()
            act_sub = actual_detail[(actual_detail['小区'] == name) & (actual_detail['类型'] == t)]
            act_val = act_sub[SERVICES].sum().sum() if len(act_sub) > 0 else 0
            mat[ci, ti] = act_val / theo_val if theo_val > 0 else 1.0

    im = ax.imshow(mat, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(TYPES)))
    ax.set_xticklabels(TYPES)
    ax.set_yticks(range(len(COMMUNITIES)))
    ax.set_yticklabels(COMMUNITIES)
    ax.set_title('消费约束下需求缩减系数（小区×老人类型）')
    for ci in range(len(COMMUNITIES)):
        for ti in range(len(TYPES)):
            ax.text(ti, ci, f'{mat[ci, ti]:.2f}', ha='center', va='center', fontsize=9)
    plt.colorbar(im, ax=ax, label='缩减系数')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig5_reduction_heatmap.png'))
    plt.close(fig)
    print('图5已保存: 缩减系数热力图')


# 图6: 理论vs实际按老人类型对比（分组柱状图）
def plot_theory_vs_actual_by_type(theo_detail, actual_detail):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ti, t in enumerate(TYPES):
        ax = axes[ti]
        theo_vals = []
        act_vals = []
        for name in COMMUNITIES:
            theo_val = theo_detail[theo_detail['小区'] == name][[f'{svc}_{t}' for svc in SERVICES]].sum().sum()
            act_sub = actual_detail[(actual_detail['小区'] == name) & (actual_detail['类型'] == t)]
            act_val = act_sub[SERVICES].sum().sum() if len(act_sub) > 0 else 0
            theo_vals.append(theo_val)
            act_vals.append(act_val)
        x = np.arange(len(COMMUNITIES))
        bar_w = 0.35
        ax.bar(x - bar_w / 2, theo_vals, bar_w, label='理论需求', color='#4C72B0', alpha=0.85)
        ax.bar(x + bar_w / 2, act_vals, bar_w, label='实际（约束后）', color='#DD8452', alpha=0.85)
        ax.set_title(f'{t}老人')
        ax.set_xlabel('小区')
        ax.set_ylabel('月需求次数（次/月）')
        ax.set_xticks(x)
        ax.set_xticklabels(COMMUNITIES)
        ax.legend(frameon=False, fontsize=8)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle('理论需求 vs 消费约束后实际需求（按老人类型）', fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig6_theory_vs_actual_by_type.png'))
    plt.close(fig)
    print('图6已保存: 理论vs实际按老人类型对比')


if __name__ == '__main__':
    print('加载数据...')
    pop_results = load_pop_data()
    theo_summary, theo_detail = load_theoretical_demand()
    actual_detail, actual_summary = load_actual_demand()

    plot_region_pop_structure(pop_results)
    plot_community_structure(pop_results)
    plot_demand_by_type(theo_detail)
    plot_demand_by_community(theo_detail)
    plot_reduction_heatmap(pop_results, actual_detail)
    plot_theory_vs_actual_by_type(theo_detail, actual_detail)
    print('\n全部图表生成完毕')
