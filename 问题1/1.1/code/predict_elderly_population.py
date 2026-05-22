# 路径配置
BASE_DIR = "问题1/1.1"
DATA_DIR = f"{BASE_DIR}/data"
RESULTS_DIR = f"{BASE_DIR}/results"

import pandas as pd
import numpy as np
import os

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# 读取初始人口数据
pop_df = pd.read_excel('题目/附件1：小区基础数据.xlsx',
                       sheet_name='人口与老人结构', skiprows=1)
pop_df.columns = ['小区', '总人口', '老人总数', '自理', '半失能', '失能', '人均月收入']
pop_df = pop_df.dropna(subset=['小区'])

# 读取转移概率
trans_df = pd.read_excel('题目/附件1：小区基础数据.xlsx',
                         sheet_name='转移概率', skiprows=1)
trans_df.columns = ['转移类型', '概率']
trans_df = trans_df.dropna(subset=['转移类型'])

p_s2m = trans_df[trans_df['转移类型'] == '自理 → 半失能']['概率'].values[0]
p_m2d = trans_df[trans_df['转移类型'] == '半失能 → 失能']['概率'].values[0]

d = 0.05
b = 0.07
n_years = 5

print(f'参数: 死亡率={d}, 新增率={b}, 自理→半失能={p_s2m}, 半失能→失能={p_m2d}')

def round_half_up(x):
    return int(np.floor(x + 0.5))

communities = pop_df['小区'].tolist()
n = len(communities)

# 初始化：第0年末数据
x1 = pop_df['自理'].values.astype(float)   # 自理
x2 = pop_df['半失能'].values.astype(float) # 半失能
x3 = pop_df['失能'].values.astype(float)   # 失能

# 存储所有年份结果
records = []

# 记录第0年
for i in range(n):
    s, m, d = round_half_up(x1[i]), round_half_up(x2[i]), round_half_up(x3[i])
    records.append({'小区': communities[i], '年份': 0,
                    '自理': s, '半失能': m, '失能': d, '合计': s + m + d})

# 递推公式（先死后增）：
# x1_new = 0.905 * x1 + 0.07 * N
# x2_new = 0.045 * x1 + 0.85 * x2
# x3_new = 0.1 * x2 + 0.95 * x3
# 其中 0.905=1-d-p_s2m, 0.85=1-d-p_m2d, 0.95=1-d

for year in range(1, n_years + 1):
    N = x1 + x2 + x3

    x1_next = 0.905 * x1 + b * N
    x2_next = p_s2m * x1 + 0.85 * x2
    x3_next = p_m2d * x2 + 0.95 * x3

    x1, x2, x3 = x1_next, x2_next, x3_next

    for i in range(n):
        s = round_half_up(x1[i])
        m = round_half_up(x2[i])
        d = round_half_up(x3[i])
        records.append({'小区': communities[i], '年份': year,
                        '自理': s, '半失能': m, '失能': d, '合计': s + m + d})

summary = pd.DataFrame(records)
summary.to_csv(f'{DATA_DIR}/prediction_5years.csv', index=False, encoding='utf-8-sig')

for year in range(n_years + 1):
    df = summary[summary['年份'] == year].copy()
    df.to_csv(f'{DATA_DIR}/year_{year}.csv', index=False, encoding='utf-8-sig')

print(f'\n第5年末各小区老人数量:')
year5 = summary[summary['年份'] == 5]
for _, row in year5.iterrows():
    print(f"小区{row['小区']}: 自理={row['自理']}, 半失能={row['半失能']}, 失能={row['失能']}, 合计={row['合计']}")

total_0 = summary[summary['年份'] == 0]['合计'].sum()
total_5 = summary[summary['年份'] == 5]['合计'].sum()
print(f'\n总老人数: 初始={total_0}, 第5年={total_5}, 增长{(total_5/total_0-1)*100:.1f}%')