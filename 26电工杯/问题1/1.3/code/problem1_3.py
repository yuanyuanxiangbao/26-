"""
问题1.3：消费约束下实际月需求（等比例缩减）
对每小区×每类老人：
  ① 人均理论月消费 = Σ(需求次数_s × 营收价格_s)
  ② 若超出上限 → 等比例缩减
  ③ round() 取整至整数
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../1.1/code'))

import pandas as pd
import re
from problem1_1 import read_data, predict_population

BASE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', '题目')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

SERVICES = ['助餐', '日间照料', '上门护理', '康复理疗', '助浴', '紧急救助']
TYPES = ['自理', '半失能', '失能']

def read_demand_rates():
    path = os.path.join(BASE_DIR, '附件2：服务需求数据.xlsx')
    df = pd.read_excel(path, sheet_name=0, header=1)
    df.columns = ['服务项目', '自理', '半失能', '失能']
    return df

def read_revenue():
    path = os.path.join(BASE_DIR, '附件2：服务需求数据.xlsx')
    df = pd.read_excel(path, sheet_name=1, header=1)
    df.columns = ['服务项目', '营收', '直接支出']
    df['营收'] = df['营收'].apply(lambda x: float(str(x).split('（')[0]) if '（' in str(x) else float(x))
    return df

def read_consumption_caps():
    path = os.path.join(BASE_DIR, '附件2：服务需求数据.xlsx')
    df = pd.read_excel(path, sheet_name=2, header=None)
    rows = df.values
    caps = {}
    for i in range(len(rows)):
        val = str(rows[i][0]).strip() if pd.notna(rows[i][0]) else ''
        cap_str = str(rows[i][1]) if pd.notna(rows[i][1]) else ''
        match = re.search(r'(\d+\.?\d*)%', cap_str)
        if not match:
            continue
        pct = float(match.group(1)) / 100.0
        if '自理' in val:
            caps['自理'] = pct
        elif '半失能' in val:
            caps['半失能'] = pct
        elif '失能' in val:
            caps['失能'] = pct
    return caps

def compute_actual_demand(pop_results, demand_df, revenue_df, pop_df, caps, year=5):
    rev_dict = dict(zip(revenue_df['服务项目'], revenue_df['营收']))
    income_dict = dict(zip(pop_df['小区'], pop_df['人均月收入']))

    demand_rate = {}
    for _, row in demand_df.iterrows():
        svc = row['服务项目']
        demand_rate[svc] = {
            '自理': float(row['自理']),
            '半失能': float(row['半失能']),
            '失能': float(row['失能']),
        }

    detail = []
    summary = {name: {svc: 0 for svc in SERVICES} for name in pop_df['小区']}
    region_summary = {svc: 0 for svc in SERVICES}

    for name, records in pop_results.items():
        _, c, s, d, _ = records[year]
        pop = {'自理': float(c), '半失能': float(s), '失能': float(d)}
        income = float(income_dict[name])

        for typ in TYPES:
            cap = income * caps[typ]
            per_capita_consumption = sum(demand_rate[svc][typ] * rev_dict[svc] for svc in SERVICES)

            if per_capita_consumption <= cap:
                factor = 1.0
            else:
                factor = cap / per_capita_consumption

            row = {'小区': name, '类型': typ}
            for svc in SERVICES:
                theoretical = pop[typ] * demand_rate[svc][typ]
                actual = round(theoretical * factor)
                row[svc] = actual
                summary[name][svc] += actual
                region_summary[svc] += actual

            detail.append(row)

    return detail, summary, region_summary

def print_results(detail, summary, region_summary):
    print('格式A：明细表（每小区 × 3类老人 × 6项服务，取整后）')
    header = f'{"小区":>4} | {"类型":>6} |'
    for s in SERVICES:
        header += f' {s:>10} |'
    print(header)
    print('-' * len(header))
    for row in detail:
        line = f'{row["小区"]:>4} | {row["类型"]:>6} |'
        for s in SERVICES:
            line += f' {row[s]:>10} |'
        print(line)

    print()
    print('格式B：汇总表（每个小区6项服务总需求）')
    header2 = f'{"小区":>4} |'
    for s in SERVICES:
        header2 += f' {s:>10} |'
    header2 += f' {"总需求":>10} |'
    print(header2)
    print('-' * len(header2))
    for name in summary:
        line = f'{name:>4} |'
        total = 0
        for s in SERVICES:
            val = summary[name][s]
            line += f' {val:>10} |'
            total += val
        line += f' {total:>10} |'
        print(line)

    print()
    print('全区域总需求')
    for s in SERVICES:
        print(f'  {s}: {region_summary[s]}')

def verify(detail, summary, pop_results, revenue_df, demand_df, pop_df, caps, year=5):
    rev_dict = dict(zip(revenue_df['服务项目'], revenue_df['营收']))
    income_dict = dict(zip(pop_df['小区'], pop_df['人均月收入']))

    _, c5, s5, d5, _ = pop_results['A'][5]

    cap_a_self = float(income_dict['A']) * caps['自理']
    assert cap_a_self == 680.0, f'A自理上限应为680，实为{cap_a_self}'

    detail_a_self = [r for r in detail if r['小区'] == 'A' and r['类型'] == '自理'][0]
    assert detail_a_self['助餐'] == round(float(c5) * 14), f'A自理助餐应为{round(float(c5)*14)}，实为{detail_a_self["助餐"]}'

    cap_a_dis = float(income_dict['A']) * caps['失能']
    factor_a_dis = cap_a_dis / (22*10 + 18*20 + 12*30 + 6*28 + 4*25 + 3*0)

    detail_a_dis = [r for r in detail if r['小区'] == 'A' and r['类型'] == '失能'][0]
    expected_meal_a_dis = round(float(d5) * 22 * factor_a_dis)
    assert detail_a_dis['助餐'] == expected_meal_a_dis, f'A失能助餐应为{expected_meal_a_dis}，实为{detail_a_dis["助餐"]}'

    print(f'校验通过：小区A')
    print(f'  自理助餐: {float(c5)} x 14 = {detail_a_self["助餐"]}（不缩减）')
    print(f'  失能缩减系数: {factor_a_dis:.6f}')
    print(f'  失能助餐: {float(d5)} x 22 x {factor_a_dis:.6f} = {detail_a_dis["助餐"]}')

    for row in detail:
        name = row['小区']
        typ = row['类型']
        if typ != '半失能':
            continue
        income = float(income_dict[name])
        cap_val = income * caps['半失能']
        per_cap = 20*10 + 14*20 + 6*30 + 4*28 + 2*25 + 1*0
        if per_cap <= cap_val:
            _, c, s, d, _ = pop_results[name][year]
            pop_val = {'自理': float(c), '半失能': float(s), '失能': float(d)}
            expected_meal = round(pop_val['半失能'] * 20)
            assert row['助餐'] == expected_meal, f'{name}半失能不应缩减，助餐应为{expected_meal}，实为{row["助餐"]}'
        else:
            factor = cap_val / per_cap
            _, c, s, d, _ = pop_results[name][year]
            pop_val = {'自理': float(c), '半失能': float(s), '失能': float(d)}
            expected_meal = round(pop_val['半失能'] * 20 * factor)
            assert row['助餐'] == expected_meal or abs(row['助餐'] - expected_meal) <= 1, \
                f'{name}半失能应缩减，助餐应为{expected_meal}，实为{row["助餐"]}'

    print('  所有小区半失能缩减校验通过')

def save_results(detail, summary, region_summary):
    os.makedirs(DATA_DIR, exist_ok=True)

    df_detail = pd.DataFrame(detail)
    df_summary_list = []
    for name, svcs in summary.items():
        row = {'小区': name}
        total = 0
        for s in SERVICES:
            row[s] = svcs[s]
            total += svcs[s]
        row['总需求'] = total
        df_summary_list.append(row)
    df_summary = pd.DataFrame(df_summary_list)

    region_row = {'小区': '全区域'}
    for s in SERVICES:
        region_row[s] = region_summary[s]
    df_region = pd.DataFrame([region_row])

    with pd.ExcelWriter(os.path.join(DATA_DIR, 'actual_demand.xlsx'), engine='openpyxl') as writer:
        df_detail.to_excel(writer, sheet_name='明细表', index=False)
        df_summary.to_excel(writer, sheet_name='汇总表', index=False)
        df_region.to_excel(writer, sheet_name='全区域', index=False)
    print(f'结果已保存到: {DATA_DIR}')

if __name__ == '__main__':
    df_pop, p1, p2 = read_data()
    demand_df = read_demand_rates()
    revenue_df = read_revenue()
    caps = read_consumption_caps()

    pop_results = predict_population(df_pop, p1, p2, years=5)
    detail, summary, region_summary = compute_actual_demand(pop_results, demand_df, revenue_df, df_pop, caps, year=5)

    print_results(detail, summary, region_summary)
    verify(detail, summary, pop_results, revenue_df, demand_df, df_pop, caps, year=5)
    save_results(detail, summary, region_summary)
