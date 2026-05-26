"""
问题3: 服务定价与政府补贴优化 & 可及性影响分析

基于问题2最优方案(膝点法选α, 从帕累托前沿Excel读取)，允许每站自主定价。
目标: 最大化老人满意度
约束: 政府补贴2元/人次(每日上限), 利润率≤8%

3.1 定价优化:
  输出: 最优定价表, 每站利润&利润率, 每小区满意度(含S3)
3.2 可及性分析:
  对比原价(营收)vs最优价, 按类型(自理/半失能/失能)分析
  经济可及性: 消费约束因子变化, 月支出变化
  补贴受益分布: 各类型有效服务人次 & 补贴额
  地理/信息可及性: 定性讨论（动态生成）
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


import importlib.util
import pandas as pd
import numpy as np
from itertools import product

from problem1_1 import read_data, predict_population
from problem1_3 import read_demand_rates, read_revenue, read_consumption_caps, SERVICES, compute_actual_demand_v2

# Load problem2.1 (dot in filename requires importlib)
_p3_dir = os.path.dirname(os.path.abspath(__file__))
_p2_spec = importlib.util.spec_from_file_location(
    "problem2_1", os.path.join(_p3_dir, 'problem2.1.py')
)
problem2_1 = importlib.util.module_from_spec(_p2_spec)
sys.modules['problem2_1'] = problem2_1
_p2_spec.loader.exec_module(problem2_1)

read_station_data = problem2_1.read_station_data
read_distance = problem2_1.read_distance
build_cover_matrix = problem2_1.build_cover_matrix
get_S1 = problem2_1.get_S1
get_S2 = problem2_1.get_S2
simulate_elderly_choice = problem2_1.simulate_elderly_choice
DAYS_MONTH = problem2_1.DAYS_MONTH
get_station_op_cost = problem2_1.get_station_op_cost

BUILD_COST = [0, 18, 32, 45]
CAPACITY = [0, 1000, 2000, 3000]
S2_DAMP = 0.5

NON_EMERGENCY = ['助餐', '日间照料', '上门护理', '康复理疗', '助浴']
DAILY_SUBSIDY_CAP = {1: 1000, 2: 1800, 3: 2600}

# 价格搜索: 包含低于基准价(补贴可弥补亏损)和S3阈值边界精细化
PRICE_MULTIPLIERS = [0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.30, 1.50]

MULT_LABELS = ['平价', '微溢价', '中溢价', '高溢价']
SIZE_LABEL = ['', '小', '中', '大']

def read_optimal_solution(excel_path, alpha):
    """从问题2帕累托前沿Excel读取α最优方案的站点配置"""
    orig = alpha
    available_sheets = [0.00, 0.50, 1.00]
    try:
        df = pd.read_excel(excel_path, sheet_name=f'α={alpha:.2f}', header=None)
    except ValueError:
        alpha = min(available_sheets, key=lambda x: abs(x - alpha))
        print(f'  [警告] α={orig:.2f} 无独立Sheet，使用 α={alpha:.2f} 的配置')
        df = pd.read_excel(excel_path, sheet_name=f'α={alpha:.2f}', header=None)
    stations = []
    found_header = False
    for i, row in df.iterrows():
        if pd.isna(row.iloc[0]):
            found_header = False
            continue
        label = str(row.iloc[0]).strip()
        if label == '服务站' or '站名' in label:
            found_header = True
            continue
        if found_header and pd.notna(row.iloc[4]):
            loc = str(row.iloc[0]).strip()
            size_str = str(row.iloc[1]).strip()
            comms_str = str(row.iloc[5]).strip()
            size = {'小': 1, '中': 2, '大': 3}.get(size_str, 2)
            comms = [c.strip() for c in comms_str.split(',') if c.strip()]
            stations.append({'loc': loc, 'size': size, 'comms': comms})
    return stations


def select_knee_from_excel(excel_path):
    """从Pareto Overview读覆盖率/满意度，膝点法选择最优α"""
    df = pd.read_excel(excel_path, sheet_name='Pareto Overview', header=None)
    front = []
    for i in range(1, len(df)):
        if pd.isna(df.iloc[i, 0]):
            break
        front.append({
            'alpha': float(df.iloc[i, 0]),
            'cov': float(str(df.iloc[i, 3]).replace('%', '')) / 100.0,
            'sat': float(df.iloc[i, 5]),
        })
    front.sort(key=lambda x: x['cov'], reverse=True)
    nondom = []
    max_sat = -1.0
    for p in front:
        if p['sat'] > max_sat:
            nondom.append(p)
            max_sat = p['sat']
    nondom.sort(key=lambda x: x['cov'])
    if len(nondom) < 3:
        return nondom[len(nondom) // 2]['alpha']
    best_score, best_alpha = -1, nondom[0]['alpha']
    for i in range(1, len(nondom) - 1):
        kl = (nondom[i]['sat'] - nondom[i - 1]['sat']) / (nondom[i]['cov'] - nondom[i - 1]['cov'] + 1e-10)
        kr = (nondom[i + 1]['sat'] - nondom[i]['sat']) / (nondom[i + 1]['cov'] - nondom[i]['cov'] + 1e-10)
        score = abs(kl - kr) / (1 + abs(kl) + abs(kr))
        if score > best_score:
            best_score, best_alpha = score, nondom[i]['alpha']
    return best_alpha


def get_S3_from_mult(mult):
    if mult <= 1.00:    return 1.00
    elif mult <= 1.10:  return 0.90
    elif mult <= 1.20:  return 0.75
    else:                return 0.60


def load_benchmark(revenue_df):
    """从附件2直接支出读取基准价"""
    return dict(zip(revenue_df['服务项目'], revenue_df['直接支出']))


def compute_community_demand(name, pop_results, demand_df, price_dict, df_pop, caps, year=5):
    _, c, s, d, _ = pop_results[name][year]
    pops = {'自理': float(c), '半失能': float(s), '失能': float(d)}
    income = float(df_pop.loc[df_pop['小区'] == name, '人均月收入'].iloc[0])

    demand_rate = {}
    for _, row in demand_df.iterrows():
        svc = row['服务项目']
        demand_rate[svc] = {
            '自理': float(row['自理']),
            '半失能': float(row['半失能']),
            '失能': float(row['失能']),
        }

    result = {svc: 0.0 for svc in SERVICES}
    for typ in ['自理', '半失能', '失能']:
        cap_amount = income * caps[typ]
        per_capita = sum(demand_rate[svc][typ] * price_dict[svc] for svc in SERVICES)
        factor = 1.0 if per_capita <= cap_amount else cap_amount / per_capita
        for svc in SERVICES:
            result[svc] += pops[typ] * demand_rate[svc][typ] * factor
    return {svc: max(round(v), 0) for svc, v in result.items()}


def compute_S3_community(summary_j, price_dict, benchmark):
    """Usage-weighted S3 for one community. Weighted by raw demand visits."""
    total_w = 0.0
    weighted = 0.0
    for svc in SERVICES:
        w = summary_j.get(svc, 0)
        if w == 0:
            continue
        if svc == '紧急救助':
            s3 = 1.0
        else:
            mult = price_dict[svc] / benchmark[svc]
            s3 = get_S3_from_mult(mult)
        weighted += w * s3
        total_w += w
    return weighted / total_w if total_w > 0 else 1.0


def converge_S2_for_station(assigned_comms, summary, S1_values, S3_values, capacity):
    S2 = 1.0
    for _ in range(20):
        total_eff_daily = 0.0
        for comm in assigned_comms:
            S1 = S1_values[comm]
            S3 = S3_values[comm]
            S = 0.2 * S1 + 0.3 * S2 + 0.5 * S3
            daily_total = sum(summary[comm][svc] for svc in SERVICES) / DAYS_MONTH
            total_eff_daily += daily_total * S
        util = total_eff_daily / capacity
        new_S2 = get_S2(min(util, 2.0))
        damped = S2_DAMP * S2 + (1 - S2_DAMP) * new_S2
        diff = abs(damped - S2)
        S2 = damped
        if diff < 0.0001 and _ > 0:
            break
    final_S = {}
    for comm in assigned_comms:
        S1 = S1_values[comm]
        S3 = S3_values[comm]
        final_S[comm] = 0.2 * S1 + 0.3 * S2 + 0.5 * S3
    return S2, final_S


def get_price_level_label(mult):
    if mult <= 1.00:    return '平价'
    elif mult <= 1.10:  return '微溢价'
    elif mult <= 1.20:  return '中溢价'
    else:                return '高溢价'


# ============================================================
# 动态定性分析生成
# ============================================================

def generate_geographic_text(S1_by_station, STATIONS, communities, pop_results, df_pop):
    lines = ['[地理可及性]']
    covered = set()
    for si, st in enumerate(STATIONS):
        for comm in st['comms']:
            covered.add(comm)
            s1 = S1_by_station[si][comm]
            lines.append(f'  站{st["loc"]}→{comm}: S1={s1:.2f}')
    all_comms = df_pop['小区'].tolist()
    uncovered = [c for c in all_comms if c not in covered]
    if uncovered:
        unc_pop = 0
        for c in uncovered:
            _, c5, s5, d5, _ = pop_results[c][5]
            unc_pop += c5 + s5 + d5
        total_unc = unc_pop
        total_pop = unc_pop
        for c in covered:
            _, c5, s5, d5, _ = pop_results[c][5]
            total_pop += c5 + s5 + d5
        pct = total_unc / total_pop * 100 if total_pop > 0 else 0
        lines.append(f'  未覆盖: {"、".join(uncovered)}({int(total_unc)}人, {pct:.2f}%)')
    return '\n'.join(lines)


def generate_information_text():
    return """[信息可及性]
不同自理能力老人获取服务信息的渠道和能力存在显著差异：
- 自理老人：行动便利、社交活跃，可通过社区公告栏、微信群、公众号等
  多种渠道主动获取价格与补贴信息，信息可及性最高。
- 半失能老人：出行受限，较少参与社区公共活动，更依赖社区专员电话
  通知、上门告知等被动渠道，需建立主动送达机制。
- 失能老人：基本居家不出，主要依靠家属或上门护理员获取信息。
  建议建立"家属-服务站"信息直通机制(短信/微信群)。"""


def generate_economic_text(df_acc_detail, df_acc_agg, grand_subsidy):
    lines = ['[经济可及性]']
    orig_mean = {typ: df_acc_detail[df_acc_detail['类型']==typ]['原价人均月消费'].mean()
                 for typ in ['自理', '半失能', '失能']}
    opt_mean = {typ: df_acc_detail[df_acc_detail['类型']==typ]['现价人均月消费'].mean()
                for typ in ['自理', '半失能', '失能']}
    cap_mean = {typ: df_acc_detail[df_acc_detail['类型']==typ]['月消费上限'].mean()
                for typ in ['自理', '半失能', '失能']}

    for typ in ['自理', '半失能', '失能']:
        sub = df_acc_detail[df_acc_detail['类型']==typ]
        n_touch = (sub['原价人均月消费'] > sub['月消费上限']).sum()
        n_total = len(sub)
        lines.append(
            f'- {typ}人均理论消费{orig_mean[typ]:.0f}元, 优化后{opt_mean[typ]:.0f}元'
            f'({"均低于" if n_touch==0 else f"{n_touch}/{n_total}触及"}上限{cap_mean[typ]:.0f}元)'
        )

    improved = df_acc_detail[df_acc_detail['因子变化'] > 0]
    if len(improved) > 0:
        orig_unconstrained = (df_acc_detail['原缩减因子'] >= 1.0).sum()
        now_unconstrained = (df_acc_detail['现缩减因子'] >= 1.0).sum()
        newly_unconstrained = now_unconstrained - orig_unconstrained
        lines.append(f'  缩减因子改善: {len(improved)}项（其中{newly_unconstrained}项完全解除约束, '
                     f'现共{now_unconstrained}项不受约束）')

    max_save = df_acc_detail.loc[df_acc_detail['月节省'].idxmax()] if len(df_acc_detail) > 0 else None
    if max_save is not None:
        lines.append(f'  最大月节省: {max_save["社区"]}-{max_save["类型"]}: {max_save["月节省"]:.1f}元/人')
        dis_opt = df_acc_detail[df_acc_detail['类型']=='失能']['现人均月支出']
        if len(dis_opt) > 0:
            lines.append(f'  失能人均月支出均值: {dis_opt.mean():.0f}元')

    lines.append('')
    lines.append('[补贴分析]')
    for _, r in df_acc_agg.iterrows():
        lines.append(f'  {r["类型"]}: 人口{r["总人口"]}, 总年省{r["总年节省(万元)"]}万元')

    subsidy_potential = df_acc_agg['补贴受益(万元)'].sum()
    lines.append(f'  潜在补贴{subsidy_potential:.2f}万, 实际{grand_subsidy/10000:.2f}万(经每日上限约束)')

    return '\n'.join(lines)


# ============================================================
# 可及性分析
# ============================================================

def analyze_accessibility(optimal_prices, S_final, S1_by_station, station_S2,
                          STATIONS, pop_results, demand_df, revenue_df, df_pop,
                          caps_dict, communities, benchmark):
    TYPES = ['自理', '半失能', '失能']
    original_prices = dict(zip(revenue_df['服务项目'], revenue_df['营收']))
    demand_rate = {}
    for _, row in demand_df.iterrows():
        svc = row['服务项目']
        demand_rate[svc] = {t: float(row[t]) for t in TYPES}

    rows = []
    for si, st in enumerate(STATIONS):
        assigned = st['comms']
        for comm in assigned:
            income = float(df_pop.loc[df_pop['小区'] == comm, '人均月收入'].iloc[0])
            _, c, s, d, _ = pop_results[comm][5]
            pops = {'自理': float(c), '半失能': float(s), '失能': float(d)}
            S_comm = S_final[si][comm]

            for typ in TYPES:
                cap = income * caps_dict[typ]
                pop = pops[typ]
                if pop == 0:
                    continue

                per_cap_orig = sum(demand_rate[svc][typ] * original_prices[svc] for svc in SERVICES)
                factor_orig = 1.0 if per_cap_orig <= cap else cap / per_cap_orig
                oop_orig = min(per_cap_orig, cap)

                per_cap_opt = sum(demand_rate[svc][typ] * optimal_prices[si][svc] for svc in SERVICES)
                factor_opt = 1.0 if per_cap_opt <= cap else cap / per_cap_opt
                oop_opt = min(per_cap_opt, cap)

                non_emerg_rate = sum(demand_rate[svc][typ] for svc in NON_EMERGENCY)
                non_emerg_eff_annual_per = non_emerg_rate * factor_opt * S_comm * 12

                rows.append({
                    '社区': comm, '类型': typ, '人口': int(pop), '收入': income,
                    '月消费上限': round(cap, 1),
                    '原价人均月消费': round(per_cap_orig, 1),
                    '现价人均月消费': round(per_cap_opt, 1),
                    '原缩减因子': round(factor_orig, 4),
                    '现缩减因子': round(factor_opt, 4),
                    '因子变化': round(factor_opt - factor_orig, 4),
                    '原人均月支出': round(oop_orig, 1),
                    '现人均月支出': round(oop_opt, 1),
                    '月节省': round(oop_orig - oop_opt, 1),
                    '年节省/人': round((oop_orig - oop_opt) * 12, 1),
                    '年有效非紧人次/人': round(non_emerg_eff_annual_per, 1),
                    'S(满意度)': round(S_comm, 4),
                })

    df_detail = pd.DataFrame(rows)

    agg_rows = []
    for typ in TYPES:
        sub = df_detail[df_detail['类型'] == typ]
        if len(sub) == 0:
            continue
        pop_total = sub['人口'].sum()
        month_save = (sub['月节省'] * sub['人口']).sum()
        non_emerg_total = (sub['年有效非紧人次/人'] * sub['人口']).sum()
        subsidy_wan = non_emerg_total * 2 / 10000
        agg_rows.append({
            '类型': typ, '总人口': int(pop_total),
            '人均月节省(元)': round(month_save / pop_total, 2) if pop_total > 0 else 0,
            '总年节省(万元)': round(month_save * 12 / 10000, 2),
            '年有效非紧总人次': round(non_emerg_total, 0),
            '补贴受益(万元)': round(subsidy_wan, 2),
            '人均年补贴(元)': round(subsidy_wan * 10000 / pop_total, 2) if pop_total > 0 else 0,
        })
    df_agg = pd.DataFrame(agg_rows)

    return df_detail, df_agg


# ============================================================
# 主流程
# ============================================================

def main():
    print('=' * 70)
    print('问题3: 服务定价与政府补贴优化 & 可及性分析')
    print('=' * 70)

    # ---- 1. 加载数据 ----
    print('\n[1/4] 加载数据...')
    df_pop, p1, p2 = read_data()
    demand_df = read_demand_rates()
    revenue_df = read_revenue()
    caps_dict = read_consumption_caps()
    station_df = read_station_data()
    dist_df = read_distance()
    communities = df_pop['小区'].tolist()

    pop_results = predict_population(df_pop, p1, p2, years=5)

    cover_mat, dist_values = build_cover_matrix(dist_df, communities)
    station_op_cost = get_station_op_cost(station_df)

    # 从附件2直接支出读取基准价（替代硬编码）
    benchmark = load_benchmark(revenue_df)

    # 读取帕累托前沿并自动选择最优α（膝点法）
    pareto_path = os.path.join(_p3_dir, '问题2_1_v3_帕累托前沿.xlsx')
    best_alpha = select_knee_from_excel(pareto_path)
    STATIONS = read_optimal_solution(pareto_path, alpha=best_alpha)
    print(f'\n膝点法选α={best_alpha:.2f}方案: {len(STATIONS)}站')
    for st in STATIONS:
        print(f'  {st["loc"]}({SIZE_LABEL[st["size"]]}): {", ".join(st["comms"])}')
    uncovered = [c for c in communities if all(c not in st['comms'] for st in STATIONS)]
    if uncovered:
        print(f'  未覆盖: {", ".join(uncovered)}')

    # ---- 老年人选择模拟（替换Excel固定分配） ----
    print('\n运行老年人选择模拟(替换Excel固定分配)...')
    elderly_pop = []
    for name in communities:
        _, c, s, d, _ = pop_results[name][5]
        elderly_pop.append(c + s + d)
    detail, summary, _ = compute_actual_demand_v2(
        pop_results, demand_df, revenue_df, df_pop, caps_dict, year=5)
    daily_demand = []
    for name in communities:
        total_m = sum(summary[name][svc] for svc in SERVICES)
        daily_demand.append(total_m / DAYS_MONTH)
    n_comm = len(communities)
    station_ids = [communities.index(st['loc']) for st in STATIONS]
    caps_plan = {sid: CAPACITY[st['size']] for sid, st in zip(station_ids, STATIONS)}
    ass, sat, S2_vals = simulate_elderly_choice(
        station_ids, daily_demand, cover_mat, dist_values, caps_plan, n_comm)
    for st in STATIONS:
        st['comms'] = []
    for j, i in enumerate(ass):
        if i >= 0:
            st_idx = station_ids.index(i)
            STATIONS[st_idx]['comms'].append(communities[j])
    print('老年人选择模拟完成:')
    for st in STATIONS:
        print(f'  {st["loc"]}({SIZE_LABEL[st["size"]]}): {", ".join(st["comms"])}')

    # ---- 2. 计算S1 ----
    print('[2/4] 计算距离满意度S1...')
    S1_by_station = {}
    for si, st in enumerate(STATIONS):
        loc_idx = communities.index(st['loc'])
        comm_S1 = {}
        for comm in st['comms']:
            c_idx = communities.index(comm)
            d = dist_values[loc_idx][c_idx]
            comm_S1[comm] = get_S1(d)
        S1_by_station[si] = comm_S1
        sname = f"{st['loc']}({SIZE_LABEL[st['size']]})"
        print(f'  站 {sname}: { {c:f" {v:.2f}" for c,v in comm_S1.items()} }')

    # ---- 3. 逐站定价优化 ----
    print('\n[3/4] 逐站定价优化(网格搜索)...')
    optimal_prices = {}
    best_station_S2 = {}
    best_community_S = {}
    best_community_S3 = {}
    # Cache: store optimal summary (demand) for each station to avoid re-computation
    optimal_summary = {}

    for si, st in enumerate(STATIONS):
        sname = f"{st['loc']}({SIZE_LABEL[st['size']]})"
        print(f'\n  --- 优化 {sname} ---')
        size = st['size']
        capacity = CAPACITY[size]
        assigned = st['comms']
        price_levels_list = list(PRICE_MULTIPLIERS)
        n_services = len(NON_EMERGENCY)

        best_avg_S = -1.0
        best_comb = None
        best_result = None
        total_combos = len(price_levels_list) ** n_services
        combo_iter = 0

        for combo in product(price_levels_list, repeat=n_services):
            combo_iter += 1
            if combo_iter % 300 == 0:
                print(f'    {combo_iter}/{total_combos}', end='\r')

            price_dict = dict(benchmark)
            price_dict['紧急救助'] = 0.0
            for k, svc_name in enumerate(NON_EMERGENCY):
                price_dict[svc_name] = benchmark[svc_name] * combo[k]

            summary_j = {}
            all_feasible = True
            for comm in assigned:
                sd = compute_community_demand(
                    comm, pop_results, demand_df, price_dict, df_pop, caps_dict, year=5
                )
                if sum(sd.values()) == 0:
                    all_feasible = False
                    break
                summary_j[comm] = sd
            if not all_feasible:
                continue

            S3_j = {}
            for comm in assigned:
                S3_j[comm] = compute_S3_community(summary_j[comm], price_dict, benchmark)
            S1_j = S1_by_station[si]
            S2, S_final = converge_S2_for_station(assigned, summary_j, S1_j, S3_j, capacity)

            annual_rev = 0.0
            annual_dir = 0.0
            annual_non_emerg_eff = 0.0
            for comm in assigned:
                S = S_final[comm]
                for svc in SERVICES:
                    svc_eff_annual = summary_j[comm][svc] * S * 12
                    if svc == '紧急救助':
                        annual_dir += svc_eff_annual * benchmark[svc]
                    else:
                        annual_rev += svc_eff_annual * price_dict[svc]
                        annual_dir += svc_eff_annual * benchmark[svc]
                        annual_non_emerg_eff += svc_eff_annual

            raw_subsidy = annual_non_emerg_eff * 2.0
            subsidy_cap = DAILY_SUBSIDY_CAP[size] * 365.0
            annual_subsidy = min(raw_subsidy, subsidy_cap)
            build_amortized = BUILD_COST[size] * 10000 / 20.0
            annual_fixed = station_op_cost[size] * 365 + build_amortized
            profit = annual_rev - annual_dir + annual_subsidy - annual_fixed
            margin = profit / annual_fixed if annual_fixed > 0 else -1.0

            if not (0.0 <= margin <= 0.08):
                continue

            avg_S = np.mean([S_final[comm] for comm in assigned])

            if avg_S > best_avg_S:
                best_avg_S = avg_S
                best_comb = combo
                best_result = {
                    'price_dict': price_dict,
                    'summary_j': summary_j,
                    'S2': S2,
                    'S_final': S_final,
                    'S3_j': S3_j,
                    'profit': profit,
                    'margin': margin,
                    'subsidy': annual_subsidy,
                    'annual_rev': annual_rev,
                    'annual_dir': annual_dir,
                    'annual_fixed': annual_fixed,
                }

        if best_result is None:
            print(f'    WARNING: no feasible combo for {sname}')
            price_dict = dict(benchmark)
            price_dict['紧急救助'] = 0.0
            for svc in NON_EMERGENCY:
                price_dict[svc] = benchmark[svc] * 1.0
            optimal_prices[si] = price_dict
            best_station_S2[si] = 1.0
            best_community_S[si] = {c: 0.0 for c in assigned}
            best_community_S3[si] = {c: 1.0 for c in assigned}
            optimal_summary[si] = {}
            continue

        print(f'    OK 满意度={best_avg_S:.4f} 利润={best_result["profit"]/10000:.2f}万 利润率={best_result["margin"]*100:.2f}%')
        detail_str = '  '.join(
            f'{svc_name}={best_comb[k]:.2f}x({get_price_level_label(best_comb[k])} S3={get_S3_from_mult(best_comb[k]):.2f})'
            for k, svc_name in enumerate(NON_EMERGENCY)
        )
        print(f'    {detail_str}')

        optimal_prices[si] = best_result['price_dict']
        best_station_S2[si] = best_result['S2']
        best_community_S[si] = best_result['S_final']
        best_community_S3[si] = best_result['S3_j']
        optimal_summary[si] = best_result['summary_j']

    # ---- 4. 最终汇总计算并输出 ----
    print('\n[4/4] 输出结果')
    print('=' * 70)

    elderly_pop = {}
    for name in communities:
        _, c, s, d, _ = pop_results[name][5]
        elderly_pop[name] = c + s + d

    grand_profit = 0.0
    grand_subsidy = 0.0
    grand_rev = 0.0
    grand_dir = 0.0
    station_details = []

    for si, st in enumerate(STATIONS):
        sname = f"{st['loc']}({SIZE_LABEL[st['size']]})"
        size = st['size']
        capacity = CAPACITY[size]
        assigned = st['comms']
        price_dict = optimal_prices[si]
        S_final = best_community_S[si]

        sum_rev = 0.0
        sum_dir = 0.0
        sum_non_emerg = 0.0

        # Use cached summary when available (avoids redundant demand computation)
        cached = optimal_summary.get(si)
        for comm in assigned:
            S = S_final[comm]
            if cached is not None and comm in cached:
                sd = cached[comm]
            else:
                sd = compute_community_demand(comm, pop_results, demand_df, price_dict, df_pop, caps_dict, year=5)
            for svc in SERVICES:
                monthly_v = sd[svc]
                eff_annual = monthly_v * S * 12
                if svc == '紧急救助':
                    sum_dir += eff_annual * benchmark[svc]
                else:
                    sum_rev += eff_annual * price_dict[svc]
                    sum_dir += eff_annual * benchmark[svc]
                    sum_non_emerg += eff_annual

        raw_sb = sum_non_emerg * 2.0
        sb_cap = DAILY_SUBSIDY_CAP[size] * 365.0
        annual_sb = min(raw_sb, sb_cap)
        fx = station_op_cost[size] * 365 + BUILD_COST[size] * 10000 / 20.0
        profit_val = sum_rev - sum_dir + annual_sb - fx
        margin_val = profit_val / fx

        grand_profit += profit_val
        grand_subsidy += annual_sb
        grand_rev += sum_rev
        grand_dir += sum_dir

        station_details.append({
            'name': sname,
            'loc': st['loc'],
            'size_label': SIZE_LABEL[st['size']],
            'si': si,
            'rev_wan': sum_rev / 10000,
            'dir_wan': sum_dir / 10000,
            'sb_wan': annual_sb / 10000,
            'sb_cap_wan': sb_cap / 10000,
            'fx_wan': fx / 10000,
            'profit_wan': profit_val / 10000,
            'margin_pct': margin_val * 100,
            'S2': best_station_S2[si],
            'assigned': assigned,
        })

    covered_pop = sum(elderly_pop[name] for st in STATIONS for name in st['comms'])
    total_pop = sum(elderly_pop.values())
    coverage = covered_pop / total_pop * 100

    all_S = [best_community_S[sd['si']][comm]
             for sd in station_details for comm in sd['assigned']]
    avg_all_S = np.mean(all_S) if all_S else 0

    # ---- 打印定价表 ----
    print(f'\n{"="*70}')
    print('最优定价方案')
    print(f'{"="*70}')
    print(f"{'站':<12} {'服务':<10} {'基准价':<8} {'最优价':<8} {'乘子':<6} {'级别':<8} {'S3':<6}")
    print('-' * 58)
    for si, st in enumerate(STATIONS):
        sname = f"{st['loc']}({SIZE_LABEL[st['size']]})"
        price_dict = optimal_prices[si]
        for svc in SERVICES:
            bp = benchmark[svc]
            if svc == '紧急救助':
                print(f"{sname:<12} {svc:<10} {bp:<8.0f} {'0.0':<8} {'免费':<6} {'-':<8} {'1.00':<6}")
            else:
                op = price_dict[svc]
                mult = op / bp
                s3 = get_S3_from_mult(mult)
                lname = get_price_level_label(mult)
                print(f"{sname:<12} {svc:<10} {bp:<8.0f} {op:<8.1f} {mult:<6.2f} {lname:<8} {s3:<6.2f}")

    # ---- 打印利润明细 ----
    print()
    for sd in station_details:
        print(f'  {sd["name"]}:')
        print(f'    年营收:    {sd["rev_wan"]:.2f} 万元')
        print(f'    年直接支出: {sd["dir_wan"]:.2f} 万元')
        print(f'    年补贴:    {sd["sb_wan"]:.2f} 万元 (上限{sd["sb_cap_wan"]:.1f}万)')
        print(f'    固定成本:  {sd["fx_wan"]:.2f} 万元(日管理+摊销)')
        print(f'    年利润:    {sd["profit_wan"]:.2f} 万元')
        print(f'    利润率:    {sd["margin_pct"]:.2f}%')
        print(f'    S2: {sd["S2"]:.4f}')
        print(f'    社区满意度:')
        for comm in sd['assigned']:
            si = sd['si']
            s3 = best_community_S3[si][comm]
            s_tot = best_community_S[si][comm]
            s1 = S1_by_station[si][comm]
            print(f'      {comm}: S={s_tot:.4f} (S1={s1:.2f} S2={sd["S2"]:.4f} S3={s3:.4f})')
        print()

    print('-' * 50)
    print(f'  汇总:')
    print(f'    总年营收:   {grand_rev/10000:.2f} 万元')
    print(f'    总直接支出: {grand_dir/10000:.2f} 万元')
    print(f'    总政府补贴: {grand_subsidy/10000:.2f} 万元')
    print(f'    总年利润:   {grand_profit/10000:.2f} 万元')
    print(f'    覆盖率:     {coverage:.2f}%')
    print(f'    平均满意度: {avg_all_S:.4f}')
    for sd in station_details:
        print(f'    {sd["name"]} 覆盖: {", ".join(sd["assigned"])}')

    # ---- S3明细 ----
    print()
    print('=' * 70)
    print('价格满意度(S3)明细')
    print('=' * 70)
    print(f"{'社区':<6} {'归属站':<10} {'S3(价)':<8} {'S2(响应)':<10} {'S1(距)':<8} {'S(总)':<8}")
    print('-' * 50)
    for sd in station_details:
        for comm in sd['assigned']:
            si = sd['si']
            print(f"{comm:<6} {sd['loc']:<10} {best_community_S3[si][comm]:<8.4f} {sd['S2']:<10.4f} {S1_by_station[si][comm]:<8.2f} {best_community_S[si][comm]:<8.4f}")

    # ---- 3.2: 可及性分析 ----
    print(f'\n{"="*70}')
    print('3.2: 定价与补贴对不同类型老人服务可及性的影响')
    print(f'{"="*70}')

    df_acc_detail, df_acc_agg = analyze_accessibility(
        optimal_prices, best_community_S, S1_by_station, best_station_S2,
        STATIONS, pop_results, demand_df, revenue_df, df_pop, caps_dict,
        communities, benchmark
    )

    print(f'\n{"社区":<5} {"类型":<8} {"人口":<6} {"收入":<6} {"上限":<8} '
          f'{"原消费":<8} {"现消费":<8} {"原因子":<8} {"现因子":<8} {"变化":<8} '
          f'{"原月支":<8} {"现月支":<8} {"月省":<6}')
    print('-' * 110)
    for _, r in df_acc_detail.iterrows():
        print(f"{r['社区']:<5} {r['类型']:<8} {int(r['人口']):<6} {int(r['收入']):<6} "
              f"{r['月消费上限']:<8} {r['原价人均月消费']:<8} {r['现价人均月消费']:<8} "
              f"{r['原缩减因子']:<8} {r['现缩减因子']:<8} {r['因子变化']:<8} "
              f"{r['原人均月支出']:<8} {r['现人均月支出']:<8} {r['月节省']:<6}")

    print(f'\n{"类型":<8} {"总人口":<8} {"人均月省":<10} {"总年省(万)":<12} '
          f'{"有效非紧人次":<14} {"补贴(万)":<10} {"人均年补贴":<10}')
    print('-' * 80)
    for _, r in df_acc_agg.iterrows():
        print(f"{r['类型']:<8} {int(r['总人口']):<8} {r['人均月节省(元)']:<10} "
              f"{r['总年节省(万元)']:<12} {r['年有效非紧总人次']:<14.0f} "
              f"{r['补贴受益(万元)']:<10} {r['人均年补贴(元)']:<10}")

    # ---- 按服务类型的定价变动分析 ----
    print(f'\n{"="*70}')
    print('各服务定价变动（对比原营收价）')
    print(f'{"="*70}')
    rev_prices = dict(zip(revenue_df['服务项目'], revenue_df['营收']))
    header = f'{"服务":<12} {"基准价":<8} {"原营收价":<10} '
    for st in STATIONS:
        label = f'{st["loc"]}({SIZE_LABEL[st["size"]]})'
        header += f'{label:<22} '
    print(header)
    print('-' * (12 + 8 + 10 + 22 * len(STATIONS)))
    for svc in SERVICES:
        bp = benchmark[svc]
        rp = rev_prices[svc]
        if svc == '紧急救助':
            line = f'{svc:<12} {bp:<8.0f} {rp:<10.0f} '
            for _ in STATIONS:
                line += f'{"公益免费":<22} '
            print(line)
        else:
            line = f'{svc:<12} {bp:<8.0f} {rp:<10.0f} '
            for si, st in enumerate(STATIONS):
                op = optimal_prices[si][svc]
                mult = op / bp
                lname = get_price_level_label(mult)
                line += f'{op:<6.1f}({mult:.2f}x){lname:<8} '
            print(line)

    # ---- 动态生成定性分析 ----
    print(f'\n{"="*70}')
    print('定性分析')
    print(f'{"="*70}')
    print()
    print(generate_economic_text(df_acc_detail, df_acc_agg, grand_subsidy))
    print()
    print(generate_geographic_text(S1_by_station, STATIONS, communities, pop_results, df_pop))
    print()
    print(generate_information_text())

    # ---- Excel输出 ----
    excel_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '问题3_定价优化结果.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Sheet 1: 定价表
        rows_pricing = []
        for si, st in enumerate(STATIONS):
            price_dict = optimal_prices[si]
            for svc in SERVICES:
                bp = benchmark[svc]
                if svc == '紧急救助':
                    op, mult, lname, s3 = 0.0, 0.0, '免费', 1.0
                else:
                    op = price_dict[svc]
                    mult = round(op / bp, 2)
                    s3 = get_S3_from_mult(mult)
                    lname = get_price_level_label(mult)
                rows_pricing.append({
                    '服务站': f"{st['loc']}({SIZE_LABEL[st['size']]})",
                    '服务项目': svc, '基准价(元/次)': bp,
                    '最优定价(元/次)': op, '价格乘子': mult,
                    '价格级别': lname, 'S3(价格满意度)': s3,
                })
        pd.DataFrame(rows_pricing).to_excel(writer, sheet_name='最优定价表', index=False)

        # Sheet 2: 利润明细
        rows_profit = [{
            '服务站': sd['name'],
            '年营收(万元)': round(sd['rev_wan'], 2),
            '年直接支出(万元)': round(sd['dir_wan'], 2),
            '年政府补贴(万元)': round(sd['sb_wan'], 2),
            '固定成本(万元)': round(sd['fx_wan'], 2),
            '年利润(万元)': round(sd['profit_wan'], 2),
            '利润率(%)': round(sd['margin_pct'], 2),
            'S2(响应满意度)': round(sd['S2'], 4),
        } for sd in station_details]
        pd.DataFrame(rows_profit).to_excel(writer, sheet_name='利润明细', index=False)

        # Sheet 3: 满意度明细
        rows_sat = [{
            '小区': comm, '归属服务站': sd['name'],
            'S1(距离)': round(S1_by_station[sd['si']][comm], 2),
            'S2(响应)': round(sd['S2'], 4),
            'S3(价格)': round(best_community_S3[sd['si']][comm], 4),
            'S(总满意度)': round(best_community_S[sd['si']][comm], 4),
        } for sd in station_details for comm in sd['assigned']]
        pd.DataFrame(rows_sat).to_excel(writer, sheet_name='满意度明细', index=False)

        # Sheet 4: 经济汇总
        coverage_detail = '; '.join(
            f'{st["loc"]}({SIZE_LABEL[st["size"]]})→{",".join(st["comms"])}'
            for st in STATIONS
        )
        summary_data = [
            ('总年营收(万元)', round(grand_rev/10000, 2)),
            ('总直接支出(万元)', round(grand_dir/10000, 2)),
            ('总政府补贴(万元)', round(grand_subsidy/10000, 2)),
            ('总年利润(万元)', round(grand_profit/10000, 2)),
            ('服务覆盖率(%)', round(coverage, 2)),
            ('平均满意度', round(avg_all_S, 4)),
            ('覆盖详情', coverage_detail),
            ('未覆盖社区', ', '.join(uncovered) if uncovered else '无'),
            ('建设总预算(万元)', 120),
        ]
        pd.DataFrame(summary_data, columns=['指标', '数值']).to_excel(
            writer, sheet_name='经济汇总', index=False
        )

        # Sheet 5: 可及性分析
        df_acc_detail_out = df_acc_detail.copy()
        df_acc_detail_out.columns = [
            '社区','类型','人口','收入','月消费上限',
            '原价人均月消费','现价人均月消费',
            '原缩减因子','现缩减因子','因子变化',
            '原人均月支出','现人均月支出','月节省','年节省每人均',
            '年有效非紧人次每人均','S满意度'
        ]
        df_acc_detail_out.to_excel(writer, sheet_name='可及性分析_明细', index=False)
        df_acc_agg.to_excel(writer, sheet_name='可及性分析_汇总', index=False)

    print(f'\nExcel已导出: {excel_path}')


if __name__ == '__main__':
    main()
