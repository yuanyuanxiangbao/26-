# 路径配置
BASE_DIR = "问题1/1.2"
DATA_DIR = f"{BASE_DIR}/data"
RESULTS_DIR = f"{BASE_DIR}/results"

import pandas as pd
import numpy as np
import os

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# 读取1.1第5年末预测结果
year5 = pd.read_csv('问题1/1.1/data/year_5.csv')

# 读取附件2：每位老人月均服务需求次数
demand_raw = pd.read_excel('题目/附件2：服务需求数据.xlsx',
                           sheet_name='每位老人月均服务需求次数', skiprows=1)
demand_raw.columns = ['服务项目', '自理', '半失能', '失能']
demand_raw = demand_raw.dropna(subset=['服务项目'])

services = demand_raw['服务项目'].tolist()
demand_rate = demand_raw[['自理', '半失能', '失能']].values.astype(float)

def round_half_up(x):
    return int(np.floor(x + 0.5))

print('每位老人月均服务需求次数（次/月）:')
for svc, rates in zip(services, demand_rate):
    print(f'  {svc}: 自理={rates[0]}, 半失能={rates[1]}, 失能={rates[2]}')

# 计算每个小区各项服务的理论月需求次数
records = []
for _, row in year5.iterrows():
    comm = row['小区']
    pop = np.array([row['自理'], row['半失能'], row['失能']])
    for svc, rates in zip(services, demand_rate):
        demand = np.array([round_half_up(x) for x in pop * rates])
        total = int(demand.sum())
        records.append({
            '小区': comm,
            '服务项目': svc,
            '自理需求': demand[0],
            '半失能需求': demand[1],
            '失能需求': demand[2],
            '合计': total
        })

result = pd.DataFrame(records)
result.to_csv(f'{DATA_DIR}/theoretical_demand.csv', index=False, encoding='utf-8-sig')

# 输出结果
print(f'\n第5年末各小区各项服务理论月需求次数:')
for comm in year5['小区']:
    sub = result[result['小区'] == comm]
    print(f'\n小区{comm}:')
    for _, r in sub.iterrows():
        print(f'  {r["服务项目"]}: 自理={r["自理需求"]}, 半失能={r["半失能需求"]}, '
              f'失能={r["失能需求"]}, 合计={r["合计"]}')

# 汇总统计
print(f'\n全区各项服务月需求汇总:')
summary = result.groupby('服务项目')[['自理需求', '半失能需求', '失能需求', '合计']].sum()
summary = summary.loc[services]  # 按原始顺序排列
for svc in services:
    row = summary.loc[svc]
    print(f'  {svc}: 自理={row["自理需求"]}, 半失能={row["半失能需求"]}, '
          f'失能={row["失能需求"]}, 合计={row["合计"]}')

# 生成宽表（每个小区一行，每个服务×类型一列）
pivot_data = []
for comm in year5['小区']:
    sub = result[result['小区'] == comm]
    rec = {'小区': comm}
    for _, r in sub.iterrows():
        rec[f'{r["服务项目"]}_自理'] = r['自理需求']
        rec[f'{r["服务项目"]}_半失能'] = r['半失能需求']
        rec[f'{r["服务项目"]}_失能'] = r['失能需求']
        rec[f'{r["服务项目"]}_合计'] = r['合计']
    pivot_data.append(rec)

pivot_df = pd.DataFrame(pivot_data)
pivot_df.to_csv(f'{DATA_DIR}/demand_by_community.csv', index=False, encoding='utf-8-sig')
