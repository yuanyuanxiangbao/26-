# 路径配置
BASE_DIR = "问题1/1.1"
DATA_DIR = f"{BASE_DIR}/data"
RESULTS_DIR = f"{BASE_DIR}/results"

import pandas as pd
import numpy as np
import os

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# 读取数据
pop_df = pd.read_excel('题目/附件1：小区基础数据.xlsx', sheet_name='人口与老人结构', skiprows=1)
pop_df.columns = ['小区', '总人口', '老人总数', '自理', '半失能', '失能', '人均月收入']
pop_df = pop_df.dropna(subset=['小区'])

trans_df = pd.read_excel('题目/附件1：小区基础数据.xlsx', sheet_name='转移概率', skiprows=1)
trans_df.columns = ['转移类型', '概率']
trans_df = trans_df.dropna(subset=['转移类型'])

p_s2m = trans_df[trans_df['转移类型'] == '自理 → 半失能']['概率'].values[0]
p_m2d = trans_df[trans_df['转移类型'] == '半失能 → 失能']['概率'].values[0]

death_rate = 0.05
birth_rate = 0.07
n_years = 5

print(f'参数: 死亡率={death_rate}, 新增率={birth_rate}, 自理→半失能={p_s2m}, 半失能→失能={p_m2d}')

# 初始化各小区老人数量
n_communities = len(pop_df)
communities = pop_df['小区'].tolist()

S0 = pop_df['自理'].values.astype(float)
M0 = pop_df['半失能'].values.astype(float)
D0 = pop_df['失能'].values.astype(float)
T0 = S0 + M0 + D0

# 记录每年结果
results = {f'year_{i}': [] for i in range(n_years + 1)}
results['year_0'] = [{'小区': communities[i], '自理': S0[i], '半失能': M0[i], '失能': D0[i], '合计': T0[i]} for i in range(n_communities)]

S, M, D = S0.copy(), M0.copy(), D0.copy()

for year in range(1, n_years + 1):
    T_prev = S + M + D
    new_elderly = birth_rate * T_prev
    
    S_survive = S * (1 - death_rate)
    M_survive = M * (1 - death_rate)
    D_survive = D * (1 - death_rate)
    
    s_to_m = S_survive * p_s2m
    m_to_d = M_survive * p_m2d
    
    S_next = (S_survive - s_to_m) + new_elderly
    M_next = M_survive - m_to_d + s_to_m
    D_next = D_survive + m_to_d
    
    S, M, D = S_next, M_next, D_next
    
    for i in range(n_communities):
        results[f'year_{year}'].append({
            '小区': communities[i],
            '自理': round(S[i]),
            '半失能': round(M[i]),
            '失能': round(D[i]),
            '合计': round(S[i] + M[i] + D[i])
        })

# 保存每年结果
for year in range(n_years + 1):
    df = pd.DataFrame(results[f'year_{year}'])
    df.to_csv(f'{DATA_DIR}/year_{year}.csv', index=False, encoding='utf-8-sig')

# 汇总所有年份
all_data = []
for year in range(n_years + 1):
    df = pd.DataFrame(results[f'year_{year}'])
    df['年份'] = year
    all_data.append(df)

summary = pd.concat(all_data, ignore_index=True)
summary = summary[['年份', '小区', '自理', '半失能', '失能', '合计']]
summary.to_csv(f'{DATA_DIR}/prediction_5years.csv', index=False, encoding='utf-8-sig')

print(f'\n预测完成，第5年末各小区老人数量:')
year5 = pd.DataFrame(results['year_5'])
for _, row in year5.iterrows():
    print(f"小区{row['小区']}: 自理={row['自理']}, 半失能={row['半失能']}, 失能={row['失能']}, 合计={row['合计']}")

total_year5 = year5['合计'].sum()
total_year0 = sum(d['合计'] for d in results['year_0'])
print(f'\n总老人数: 初始={total_year0}, 第5年={total_year5}, 增长{(total_year5/total_year0-1)*100:.1f}%')
