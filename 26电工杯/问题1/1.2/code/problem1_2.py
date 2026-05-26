"""
问题1.2：第5年末各小区理论月需求次数（分自理/半失能/失能）
计算：第5年末人数 × 月均服务需求次数
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../1.1/code'))

import pandas as pd
from problem1_1 import read_data, predict_population

BASE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', '题目')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def read_demand_rates():
    path = os.path.join(BASE_DIR, '附件2：服务需求数据.xlsx')
    df = pd.read_excel(path, sheet_name=0, header=1)
    df.columns = ['服务项目', '自理', '半失能', '失能']
    return df

def compute_theoretical_demand(pop_results, demand_df, year=5):
    services = demand_df['服务项目'].tolist()
    results_detail = []
    results_summary = []

    for name, records in pop_results.items():
        _, c, s, d, _ = records[year]

        detail_row = {'小区': name}
        summary_row = {'小区': name}
        summary_total = 0

        for _, row in demand_df.iterrows():
            svc = row['服务项目']
            r_c = round(float(c) * float(row['自理']))
            r_s = round(float(s) * float(row['半失能']))
            r_d = round(float(d) * float(row['失能']))
            total = r_c + r_s + r_d

            detail_row[f'{svc}_自理'] = r_c
            detail_row[f'{svc}_半失能'] = r_s
            detail_row[f'{svc}_失能'] = r_d
            summary_row[svc] = total
            summary_total += total

        summary_row['总需求'] = summary_total
        results_detail.append(detail_row)
        results_summary.append(summary_row)

    return results_detail, results_summary

def print_results(detail, summary):
    svc_names = ['助餐', '日间照料', '上门护理', '康复理疗', '助浴', '紧急救助']

    print('格式A：明细表（每个小区 × 3类老人 × 6项服务）')
    header = f'{"小区":>4} | {"类型":>6} |'
    for s in svc_names:
        header += f' {s:>10} |'
    print(header)
    print('-' * len(header))
    for row in detail:
        name = row['小区']
        for typ, label in [('自理', '自理'), ('半失能', '半失能'), ('失能', '失能')]:
            line = f'{name:>4} | {label:>6} |'
            for s in svc_names:
                val = row[f'{s}_{typ}']
                line += f' {val:>10} |'
            print(line)

    print()
    print('格式B：汇总表（每个小区6项服务总需求）')
    header2 = f'{"小区":>4} |'
    for s in svc_names:
        header2 += f' {s:>10} |'
    header2 += f' {"总需求":>10} |'
    print(header2)
    print('-' * len(header2))
    for row in summary:
        line = f'{row["小区"]:>4} |'
        for s in svc_names:
            line += f' {row[s]:>10} |'
        line += f' {row["总需求"]:>10} |'
        print(line)

def verify(detail, summary, pop_results):
    a_detail = [r for r in detail if r['小区'] == 'A'][0]
    _, c5, s5, d5, _ = pop_results['A'][5]

    assert a_detail['助餐_自理'] == round(float(c5) * 14)
    assert a_detail['助餐_半失能'] == round(float(s5) * 20)
    assert a_detail['助餐_失能'] == round(float(d5) * 22)

    a_sum = [r for r in summary if r['小区'] == 'A'][0]
    expected = a_detail['助餐_自理'] + a_detail['助餐_半失能'] + a_detail['助餐_失能']
    assert a_sum['助餐'] == expected

    print('校验通过：小区A助餐明细与汇总一致')

def save_results(detail, summary):
    os.makedirs(DATA_DIR, exist_ok=True)

    df_detail = pd.DataFrame(detail)
    df_summary = pd.DataFrame(summary)

    with pd.ExcelWriter(os.path.join(DATA_DIR, 'theoretical_demand.xlsx'), engine='openpyxl') as writer:
        df_detail.to_excel(writer, sheet_name='明细表', index=False)
        df_summary.to_excel(writer, sheet_name='汇总表', index=False)
    print(f'结果已保存到: {DATA_DIR}')

if __name__ == '__main__':
    df_pop, p1, p2 = read_data()
    demand_df = read_demand_rates()

    pop_results = predict_population(df_pop, p1, p2, years=5)
    detail, summary = compute_theoretical_demand(pop_results, demand_df, year=5)

    print_results(detail, summary)
    print()
    verify(detail, summary, pop_results)
    save_results(detail, summary)
