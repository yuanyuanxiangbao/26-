# -*- coding: utf-8 -*-
"""
问题4：灵敏度分析与方案比较（完全自包含，不调用任何现有 .py 文件）
- S1: 人口参数变化 (new_rate=8%, p_stos=5.5%, p_tod=9.5%)
- S2: 日管理成本+20%
- S3: 预算140万, k=2..7
"""

import os, sys, time, random, re
from functools import lru_cache
from itertools import combinations, product
from multiprocessing import Pool, cpu_count
from collections import OrderedDict
from dataclasses import dataclass
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================
# 常量
# ==============================
SERVICES = ['助餐', '日间照料', '上门护理', '康复理疗', '助浴', '紧急救助']
NON_EMERGENCY = ['助餐', '日间照料', '上门护理', '康复理疗', '助浴']
TYPES = ['自理', '半失能', '失能']
CAPACITY = [0, 1000, 2000, 3000]
BUILD_COST = [0, 18, 32, 45]
SIZE_LABEL = ['', '小', '中', '大']
RADIUS = 1000
DAYS_MONTH = 30
S3_DEFAULT = 1.0
ALPHAS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
          0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
CLUSTER_THRESHOLD = 700
CLUSTER_BUDGET_RATIO = 1.5
DAILY_SUBSIDY_CAP = {1: 1000, 2: 1800, 3: 2600}
PRICE_MULTIPLIERS = [0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.30, 1.50]
N_CORES = 16

# 模块级缓存（跨场景共享不变数据）
_demand_rate_cache = None
_income_dict_cache = None

def _get_demand_rate(demand_df):
    global _demand_rate_cache
    if _demand_rate_cache is not None:
        return _demand_rate_cache
    result = {}
    for _, row in demand_df.iterrows():
        svc = row['服务项目']
        result[svc] = {
            '自理': float(row['自理']),
            '半失能': float(row['半失能']),
            '失能': float(row['失能']),
        }
    _demand_rate_cache = result
    return result


def _get_income_dict(pop_df):
    global _income_dict_cache
    if _income_dict_cache is not None:
        return _income_dict_cache
    result = dict(zip(pop_df['小区'], pop_df['人均月收入']))
    _income_dict_cache = result
    return result

# ==============================
# 场景定义
# ==============================
SCENARIOS = OrderedDict([
    ('baseline', {
        'label': '基线方案',
        'budget': 120, 'k_min': 2, 'k_max': 6,
        'op_cost_mult': 1.0,
        'new_rate': None, 'p_stos': None, 'p_tod': None,
    }),
    ('S1', {
        'label': 'S1:人口参数变化',
        'budget': 120, 'k_min': 2, 'k_max': 6,
        'op_cost_mult': 1.0,
        'new_rate': 0.08, 'p_stos': 0.055, 'p_tod': 0.095,
    }),
    ('S2', {
        'label': 'S2:日管理成本+20%',
        'budget': 120, 'k_min': 2, 'k_max': 6,
        'op_cost_mult': 1.2,
        'new_rate': None, 'p_stos': None, 'p_tod': None,
    }),
    ('S3', {
        'label': 'S3:预算140万(k=2..7)',
        'budget': 140, 'k_min': 2, 'k_max': 7,
        'op_cost_mult': 1.0,
        'new_rate': None, 'p_stos': None, 'p_tod': None,
    }),
])


# ==============================
# [1] 数据加载
# ==============================

@dataclass
class ModelConfig:
    days_per_year: int = 365
    mortality: float = 0.05
    new_rate: float = 0.07


@lru_cache(maxsize=None)
def read_population_data():
    path = os.path.join(BASE_DIR, '附件1：小区基础数据.xlsx')
    df = pd.read_excel(path, sheet_name=0, header=1)
    df.columns = ['小区', '总人口', '60+老人数', '自理', '半失能', '失能', '人均月收入']
    df_prob = pd.read_excel(path, sheet_name=1, header=1)
    df_prob.columns = ['场景', '概率']
    p1 = float(df_prob.loc[df_prob['场景'].str.contains('自理.*半失能', na=False), '概率'].values[0])
    p2 = float(df_prob.loc[df_prob['场景'].str.contains('半失能.*失能', na=False), '概率'].values[0])
    return df, p1, p2


@lru_cache(maxsize=None)
def read_demand_rates():
    path = os.path.join(BASE_DIR, '附件2：服务需求数据.xlsx')
    df = pd.read_excel(path, sheet_name=0, header=1)
    df.columns = ['服务项目', '自理', '半失能', '失能']
    return df


@lru_cache(maxsize=None)
def read_revenue():
    path = os.path.join(BASE_DIR, '附件2：服务需求数据.xlsx')
    df = pd.read_excel(path, sheet_name=1, header=1)
    df.columns = ['服务项目', '营收', '直接支出']
    df['营收'] = df['营收'].apply(
        lambda x: float(str(x).split('（')[0]) if '（' in str(x) else float(x))
    return df


@lru_cache(maxsize=None)
def read_consumption_caps():
    path = os.path.join(BASE_DIR, '附件2：服务需求数据.xlsx')
    df = pd.read_excel(path, sheet_name=2, header=None)
    caps = {}
    for _, rows in df.iterrows():
        val = str(rows.iloc[0]).strip() if pd.notna(rows.iloc[0]) else ''
        cap_str = str(rows.iloc[1]) if pd.notna(rows.iloc[1]) else ''
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


@lru_cache(maxsize=None)
def read_station_data():
    path = os.path.join(BASE_DIR, '附件3：服务站建设与运营成本.xlsx')
    return pd.read_excel(path, sheet_name=0, header=1,
                         names=['规模', '建设成本_万元', '日管理成本_元', '日容量'],
                         usecols=[0, 1, 2, 3])


@lru_cache(maxsize=None)
def read_distance():
    path = os.path.join(BASE_DIR, '附件4：小区间距离矩阵.xlsx')
    df = pd.read_excel(path, sheet_name=0, header=1)
    df = df.set_index('组别')
    df.index.name = None
    df.columns = [c.strip() for c in df.columns]
    return df


def get_station_op_cost(station_df, multiplier=1.0):
    op_cost = [0.0]
    for _, row in station_df.iterrows():
        val = row.iloc[2]
        if pd.isna(val):
            continue
        op_cost.append(float(val) * multiplier)
    return op_cost


def get_price_data(revenue_df):
    rev_dict, cost_dict = {}, {}
    for _, row in revenue_df.iterrows():
        svc = row['服务项目']
        rev_str = str(row['营收'])
        cost_str = str(row['直接支出'])
        rev_dict[svc] = float(rev_str.split('（')[0]) if '（' in rev_str else float(rev_str)
        cost_dict[svc] = float(cost_str.split('（')[0]) if '（' in cost_str else float(cost_str)
    return rev_dict, cost_dict


def load_benchmark(revenue_df):
    return dict(zip(revenue_df['服务项目'], revenue_df['直接支出']))


# ==============================
# [2] 人口预测 - 与problem1_1.py保持一致的年级递推模型
# ==============================

def predict_population(df_pop, p_self_to_semi, p_semi_to_dis, years=5,
                       config=None, new_rate_override=None):
    """
    年级递推模型（与problem1_1.py一致）
    时序（每年）：
      1. 年初：加入新增人员（年初人口 × new_rate），新增人全部归自理类
      2. 全年：逐日处理死亡与转移（365日）
         - 死亡：既存人口按5%年率扣除（日化）
         - 转移：自理→半失能，半失能→失能（日化）
      3. 年末：记录各类人口数
    """
    if config is None:
        config = ModelConfig()
    
    eff_new_rate = new_rate_override if new_rate_override is not None else config.new_rate
    
    # 日化参数
    mu = 1 - (1 - config.mortality) ** (1 / config.days_per_year)
    t1 = 1 - (1 - p_self_to_semi) ** (1 / config.days_per_year)
    t2 = 1 - (1 - p_semi_to_dis) ** (1 / config.days_per_year)
    
    results = {}
    for _, row in df_pop.iterrows():
        name = row['小区']
        c0, s0, d0 = float(row['自理']), float(row['半失能']), float(row['失能'])
        
        records = [(0, c0, s0, d0, c0 + s0 + d0)]
        
        for year in range(1, years + 1):
            # 第1步：年初加入新增人员（基于上年末总人口）
            prev_total = c0 + s0 + d0
            new_people = prev_total * eff_new_rate  # 年初新增人数
            c0 += new_people  # 新增人全部入自理
            
            # 第2步：全年逐日处理死亡与转移（365日）
            for day in range(1, config.days_per_year + 1):
                # 死亡（作用于所有人：年初既存+年初新增）
                c0 *= (1 - mu)
                s0 *= (1 - mu)
                d0 *= (1 - mu)
                
                # 转移（作用于存活人口）
                to_semi = c0 * t1
                to_dis = s0 * t2
                c0 -= to_semi
                s0 += to_semi - to_dis
                d0 += to_dis
            
            # 第3步：年末记录
            total = c0 + s0 + d0
            records.append((year, c0, s0, d0, total))
        
        results[name] = records
    
    return results


# ==============================
# [3] 需求计算
# ==============================

def compute_actual_demand(pop_results, demand_df, revenue_df, pop_df, caps, year=5):
    rev_dict = dict(zip(revenue_df['服务项目'], revenue_df['营收']))
    income_dict = _get_income_dict(pop_df)
    demand_rate = _get_demand_rate(demand_df)
    summary = {name: {svc: 0 for svc in SERVICES} for name in pop_df['小区']}
    for name, records in pop_results.items():
        _, c, s, d, _ = records[year]
        pop = {'自理': float(c), '半失能': float(s), '失能': float(d)}
        income = float(income_dict[name])
        for typ in TYPES:
            cap = income * caps[typ]
            per_cap = sum(demand_rate[svc][typ] * rev_dict[svc] for svc in SERVICES)
            factor = 1.0 if per_cap <= cap else cap / per_cap
            for svc in SERVICES:
                summary[name][svc] += round(pop[typ] * demand_rate[svc][typ] * factor)
    return summary


def compute_demand_at_price(name, pop_results, demand_df, price_dict, pop_df, caps, year=5):
    _, c, s, d, _ = pop_results[name][year]
    pops = {'自理': float(c), '半失能': float(s), '失能': float(d)}
    income = _get_income_dict(pop_df)[name]
    demand_rate = _get_demand_rate(demand_df)
    result = {svc: 0.0 for svc in SERVICES}
    for typ in TYPES:
        cap_amount = income * caps[typ]
        per_capita = sum(demand_rate[svc][typ] * price_dict[svc] for svc in SERVICES)
        factor = 1.0 if per_capita <= cap_amount else cap_amount / per_capita
        for svc in SERVICES:
            result[svc] += pops[typ] * demand_rate[svc][typ] * factor
    return {svc: max(round(v), 0) for svc, v in result.items()}


# ==============================
# [4] 满意度函数
# ==============================

def get_S1(dist):
    if dist <= 300:
        return 1.00
    elif dist <= 500:
        return 0.90
    elif dist <= 650:
        return 0.75
    elif dist <= 1000:
        return 0.60
    return 0.0


def get_S2(util):
    if util <= 0.60:
        return 1.00
    elif util <= 0.75:
        return 0.93
    elif util <= 0.85:
        return 0.85
    elif util <= 0.95:
        return 0.72
    else:
        return 0.60


def get_S3_from_mult(mult):
    if mult <= 1.00:
        return 1.00
    elif mult <= 1.10:
        return 0.90
    elif mult <= 1.20:
        return 0.75
    else:
        return 0.60


def get_price_level_label(mult):
    if mult <= 1.00:
        return '平价'
    elif mult <= 1.10:
        return '微溢价'
    elif mult <= 1.20:
        return '中溢价'
    else:
        return '高溢价'


def compute_S3_community(summary_j, price_dict, benchmark):
    total_w, weighted = 0.0, 0.0
    for svc in SERVICES:
        w = summary_j.get(svc, 0)
        if w == 0:
            continue
        s3 = 1.0 if svc == '紧急救助' else get_S3_from_mult(price_dict[svc] / benchmark[svc])
        weighted += w * s3
        total_w += w
    return weighted / total_w if total_w > 0 else 1.0


# ==============================
# [5] 覆盖矩阵 + 聚类
# ==============================

def build_cover_matrix(dist_df, communities, radius=RADIUS):
    n = len(communities)
    cover = [[False] * n for _ in range(n)]
    dist_values = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            d = float(dist_df.iloc[i, j])
            dist_values[i][j] = d
            if d <= radius:
                cover[i][j] = True
    return cover, dist_values


def perform_clustering(communities, dist_values, elderly_pop, daily_demand, budget):
    n = len(communities)
    try:
        import scipy.cluster.hierarchy as sch
        import scipy.spatial.distance as ssd
        dist_array = ssd.squareform(np.array(dist_values, dtype=float))
        Z = sch.linkage(dist_array, method='average')
        labels = sch.fcluster(Z, t=CLUSTER_THRESHOLD, criterion='distance')
        labels = [int(x) - 1 for x in labels]
    except ImportError:
        selected = [0]
        farthest = max(range(n), key=lambda j: dist_values[selected[0]][j])
        selected.append(farthest)
        labels = [0 if dist_values[i][selected[0]] <= dist_values[i][selected[1]] else 1
                  for i in range(n)]
    unique_labels = sorted(set(labels))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    labels = [label_map[l] for l in labels]
    n_clusters = max(labels) + 1
    cluster_info = {}
    for cid in range(n_clusters):
        members = [i for i, l in enumerate(labels) if l == cid]
        cluster_info[cid] = {
            'members': members,
            'total_pop': sum(elderly_pop[i] for i in members),
            'total_demand': sum(daily_demand[i] for i in members),
        }
    return labels, cluster_info


def filter_plan_by_cluster(plan, cluster_labels, cluster_info, budget):
    n = len(plan)
    n_clusters = max(cluster_labels) + 1
    station_in = [False] * n_clusters
    cost_in = [0.0] * n_clusters
    for i in range(n):
        if plan[i] > 0:
            cid = cluster_labels[i]
            station_in[cid] = True
            cost_in[cid] += BUILD_COST[plan[i]]
    total_pop = sum(ci['total_pop'] for ci in cluster_info.values())
    for cid in range(n_clusters):
        if not station_in[cid]:
            return False
        pop_ratio = cluster_info[cid]['total_pop'] / max(total_pop, 1)
        if cost_in[cid] > pop_ratio * budget * CLUSTER_BUDGET_RATIO:
            return False
    return True


# ==============================
# [6] 方案枚举
# ==============================

def enumerate_plans(n, budget, k_min=2, k_max=6,
                    cluster_labels=None, cluster_info=None):
    plans = []
    for k in range(k_min, k_max + 1):
        if k * BUILD_COST[1] > budget:
            continue
        for locs in combinations(range(n), k):
            for sizes in product([1, 2, 3], repeat=k):
                cost = sum(BUILD_COST[sz] for sz in sizes)
                if cost <= budget:
                    plan = [0] * n
                    for idx, sz in zip(locs, sizes):
                        plan[idx] = sz
                    plans.append(tuple(plan))
    if cluster_labels is not None and cluster_info is not None:
        plans = [p for p in plans
                 if filter_plan_by_cluster(p, cluster_labels, cluster_info, budget)]
    random.Random(42).shuffle(plans)
    return plans


# ==============================
# [7] 贪心种子 + 爬山
# ==============================

def _compute_S1_matrix(station_ids, cover_mat, dist_values, n):
    n_sta = len(station_ids)
    S1_mat = np.zeros((n_sta, n))
    for idx, i in enumerate(station_ids):
        for j in range(n):
            if cover_mat[i][j]:
                S1_mat[idx, j] = get_S1(dist_values[i][j])
    return S1_mat


def simulate_elderly_choice(station_ids, daily_demand, cover_mat, dist_mat,
                            cap_of_plan, n, S3_by_station=None, max_rounds=20):
    if S3_by_station is None:
        S3_by_station = {i: 1.0 for i in station_ids}
    n_sta = len(station_ids)
    S1_mat = np.zeros((n_sta, n))
    for idx, i in enumerate(station_ids):
        for j in range(n):
            if cover_mat[i][j]:
                S1_mat[idx, j] = get_S1(dist_mat[i][j])
    ass = np.full(n, -1, dtype=np.int32)
    sat = np.zeros(n)
    S2 = np.ones(n_sta)
    for _round in range(max_rounds):
        pairs = []
        for j in range(n):
            for idx, i in enumerate(station_ids):
                if cover_mat[i][j]:
                    s3 = S3_by_station.get(i, 1.0)
                    S = 0.2 * S1_mat[idx, j] + 0.3 * S2[idx] + 0.5 * s3
                    pairs.append((j, idx, S))
        pairs.sort(key=lambda x: -x[2])
        new_ass = np.full(n, -1, dtype=np.int32)
        new_sat = np.zeros(n)
        remain = {i: cap_of_plan[i] for i in station_ids}
        for j, idx, _ in pairs:
            if new_ass[j] >= 0:
                continue
            i = station_ids[idx]
            s3 = S3_by_station.get(i, 1.0)
            S_val = 0.2 * S1_mat[idx, j] + 0.3 * S2[idx] + 0.5 * s3
            if daily_demand[j] * S_val <= remain[i] + 1e-9:
                new_ass[j] = i
                new_sat[j] = S_val
                remain[i] -= daily_demand[j] * S_val
        loads = {i: sum(daily_demand[j] * new_sat[j] for j in range(n) if new_ass[j] == i)
                 for i in station_ids}
        new_S2 = np.zeros(n_sta)
        for idx, i in enumerate(station_ids):
            util = loads[i] / cap_of_plan[i] if cap_of_plan[i] > 0 else 0
            new_S2[idx] = get_S2(util)
        changed = not np.array_equal(new_ass, ass)
        S2_diff = np.max(np.abs(new_S2 - S2))
        ass, sat, S2 = new_ass, new_sat, new_S2
        if not changed and S2_diff < 0.01:
            break
    return ass, sat, S2


def _hill_climb_hybrid(plan, station_ids, caps, daily_demand,
                       cover_mat, dist_values, elderly_pop, alpha, n,
                       max_rounds=10, budget=120):
    """
    方案C: 配置优化爬山
    完全符合"老人只选择满意度最高的服务站"约束
    
    核心原则:
    1. 只改变 plan（站点配置）
    2. 每次 plan 改变后，重新运行 simulate_elderly_choice
    3. 不直接操作 ass/sat（由老人选择决定）
    """
    current_plan = list(plan)
    
    def _evaluate_plan(trial_plan):
        """评估某个配置的得分（符合约束）"""
        st_ids = [i for i, sz in enumerate(trial_plan) if sz > 0]
        if len(st_ids) < 2:
            return -1.0, None, None, None
        cp = {i: CAPACITY[trial_plan[i]] for i in st_ids}
        a, s, s2 = simulate_elderly_choice(
            st_ids, daily_demand, cover_mat, dist_values, cp, n)
        covered = [j for j in range(n) if a[j] >= 0]
        if not covered:
            return -1.0, a, s, s2
        total_S = float(np.sum(s[covered]))
        cov_r = sum(elderly_pop[j] for j in covered) / max(sum(elderly_pop), 1)
        score = alpha * cov_r + (1 - alpha) * (total_S / n)
        return score, a, s, s2
    
    # 初始评估
    best_score, best_ass, best_sat, best_S2 = _evaluate_plan(current_plan)
    best_plan = list(current_plan)
    
    for _round in range(max_rounds):
        improved = False
        
        # 操作1: 改变现有站点规模
        for i in range(n):
            if current_plan[i] == 0:
                continue
            for new_size in [1, 2, 3]:
                if new_size == current_plan[i]:
                    continue
                trial_plan = list(current_plan)
                trial_plan[i] = new_size
                if sum(BUILD_COST[sz] for sz in trial_plan if sz > 0) > budget:
                    continue
                score, a, s, s2 = _evaluate_plan(trial_plan)
                if score > best_score + 1e-9:
                    best_score = score
                    best_ass, best_sat, best_S2 = a, s, s2
                    best_plan = trial_plan
                    current_plan = trial_plan
                    improved = True
        
        # 操作2: 添加新站点
        for i in range(n):
            if current_plan[i] > 0:
                continue
            for new_size in [1, 2, 3]:
                trial_plan = list(current_plan)
                trial_plan[i] = new_size
                if sum(BUILD_COST[sz] for sz in trial_plan if sz > 0) > budget:
                    continue
                score, a, s, s2 = _evaluate_plan(trial_plan)
                if score > best_score + 1e-9:
                    best_score = score
                    best_ass, best_sat, best_S2 = a, s, s2
                    best_plan = trial_plan
                    current_plan = trial_plan
                    improved = True
        
        # 操作3: 移除站点
        if sum(1 for sz in current_plan if sz > 0) > 2:
            for i in range(n):
                if current_plan[i] == 0:
                    continue
                trial_plan = list(current_plan)
                trial_plan[i] = 0
                score, a, s, s2 = _evaluate_plan(trial_plan)
                if score > best_score + 1e-9:
                    best_score = score
                    best_ass, best_sat, best_S2 = a, s, s2
                    best_plan = trial_plan
                    current_plan = trial_plan
                    improved = True
        
        if not improved:
            break
    
    # 最终评估（确保符合约束）
    if best_ass is None:
        return None, None, None, best_plan
    
    return best_ass, best_sat, best_S2, best_plan


def select_knee_alpha(pareto, alphas):
    front = []
    for a in alphas:
        r = pareto.get(a)
        if r is None or r['plan'] is None:
            continue
        front.append({'alpha': a, 'cov': r['cov_ratio'], 'sat': r['avg_s']})
    if not front:
        return None
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
        kl = (nondom[i]['sat'] - nondom[i - 1]['sat']) / (
            nondom[i]['cov'] - nondom[i - 1]['cov'] + 1e-10)
        kr = (nondom[i + 1]['sat'] - nondom[i]['sat']) / (
            nondom[i + 1]['cov'] - nondom[i]['cov'] + 1e-10)
        score = abs(kl - kr) / (1 + abs(kl) + abs(kr))
        if score > best_score:
            best_score, best_alpha = score, nondom[i]['alpha']
    return best_alpha


# ==============================
# [8] 方案求解管线
# ==============================

def solve_plan(args):
    (plan, alphas, daily_demand, elderly_pop, cover_mat, dist_values, n, budget) = args
    station_ids = [i for i, sz in enumerate(plan) if sz > 0]
    if not station_ids or len(station_ids) < 2:
        return None
    caps = {i: CAPACITY[plan[i]] for i in station_ids}
    
    results = {}
    for alpha in sorted(alphas):
        # 直接调用混合爬山（内部会重新评估）
        ass, sat, S2_arr, final_plan = _hill_climb_hybrid(
            plan, station_ids, caps, daily_demand,
            cover_mat, dist_values, elderly_pop, alpha, n,
            max_rounds=10, budget=budget)
        
        if ass is None or np.all(ass < 0):
            continue
        
        final_station_ids = [i for i, sz in enumerate(final_plan) if sz > 0]
        cov_r = (sum(elderly_pop[j] for j in range(n) if ass[j] >= 0)
                 / max(sum(elderly_pop), 1))
        total_S = float(np.sum(sat[ass >= 0]))
        avg_s_global = total_S / n
        score = alpha * cov_r + (1 - alpha) * avg_s_global
        
        results[alpha] = {
            'plan': tuple(final_plan),
            'assignment': {int(j): int(ass[j]) for j in range(n) if ass[j] >= 0},
            'sat_dict': {int(j): float(sat[j]) for j in range(n) if ass[j] >= 0},
            'S2': {int(i): float(S2_arr[idx])
                   for idx, i in enumerate(final_station_ids)},
            'cov_ratio': cov_r,
            'geo_cov_ratio': sum(1 for j in range(n) if ass[j] >= 0) / n,
            'avg_s': avg_s_global,
            'score': score,
        }
    
    return results if results else None


def solve_all_plans(all_plans, alphas, daily_demand, elderly_pop,
                    cover_mat, dist_values, n, budget=120):
    import gc
    pareto = {a: {'plan': None, 'score': -1.0, 'cov_ratio': 0.0,
                  'geo_cov_ratio': 0.0, 'avg_s': 0.0, 'assignment': {},
                  'sat_dict': {}, 'S2': {}} for a in alphas}
    worker_args = [(plan, alphas, daily_demand, elderly_pop,
                    cover_mat, dist_values, n, budget) for plan in all_plans]
    t0 = time.time()
    solved = 0
    pool = Pool(N_CORES)
    try:
        for plan_results in pool.imap_unordered(solve_plan, worker_args):
            if plan_results is None:
                continue
            solved += 1
            for alpha, result in plan_results.items():
                if result['score'] > pareto[alpha]['score']:
                    pareto[alpha] = {
                        'plan': result['plan'], 'score': result['score'],
                        'cov_ratio': result['cov_ratio'],
                        'geo_cov_ratio': result['geo_cov_ratio'],
                        'avg_s': result['avg_s'],
                        'assignment': result['assignment'],
                        'sat_dict': result['sat_dict'], 'S2': result['S2'],
                    }
            if solved % 500 == 0:
                elapsed = time.time() - t0
                rate = elapsed / solved
                remaining = rate * (len(worker_args) - solved)
                print(f'    已解 {solved}/{len(worker_args)}  '
                      f'耗时 {elapsed:.0f}s  预计剩余 {remaining:.0f}s')
    finally:
        pool.close()
        pool.join()
    del worker_args
    gc.collect()
    print(f'  求解完成: {solved}方案, {time.time() - t0:.1f}s')
    return pareto


# ==============================
# [9] 利润计算
# ==============================

def compute_station_profit(plan, assignment, communities, summary,
                           rev_dict, cost_dict, station_op_cost, sat_dict=None):
    n = len(plan)
    profit_by, rev_by, cost_by = {}, {}, {}
    for i in range(n):
        if plan[i] == 0:
            continue
        assigned = [j for j in range(n) if assignment.get(j) == i]
        annual_rev = annual_dir = 0.0
        for j in assigned:
            cname = communities[j]
            S_j = sat_dict.get(j, 1.0) if sat_dict else 1.0
            for svc in SERVICES:
                monthly = summary[cname].get(svc, 0) * S_j
                annual_rev += monthly * rev_dict[svc] * 12
                annual_dir += monthly * cost_dict[svc] * 12
        build = BUILD_COST[plan[i]] * 10000
        daily_op = station_op_cost[plan[i]]
        amortized = build / 20.0
        annual_fixed = daily_op * 365 + amortized
        profit_by[i] = annual_rev - annual_dir - annual_fixed
        rev_by[i] = annual_rev
        cost_by[i] = annual_dir + annual_fixed
    return profit_by, rev_by, cost_by


def compute_profit_for_pareto(pareto, alphas, communities, summary,
                              rev_dict, cost_dict, station_op_cost,
                              daily_demand, n):
    for a in alphas:
        r = pareto[a]
        if r['plan'] is None:
            continue
        plan = r['plan']
        assignment = r['assignment']
        profit, rev, cost = compute_station_profit(
            plan, assignment, communities, summary,
            rev_dict, cost_dict, station_op_cost, sat_dict=r.get('sat_dict'))
        r['station_profit'] = profit
        r['station_rev'] = rev
        r['station_cost'] = cost
        r['total_profit'] = sum(profit.values())
        r['cost'] = sum(BUILD_COST[x] for x in plan if x > 0)
        r['n_stations'] = sum(1 for x in plan if x > 0)


# ==============================
# [10] 定价优化
# ==============================

def converge_S2_for_station(assigned_comms, summary, S1_values, S3_values, capacity):
    S2 = 1.0
    for _ in range(20):
        total_eff_daily = 0.0
        for comm in assigned_comms:
            S = 0.2 * S1_values[comm] + 0.3 * S2 + 0.5 * S3_values[comm]
            total_eff_daily += sum(summary[comm][svc] for svc in SERVICES) / DAYS_MONTH * S
        util = total_eff_daily / capacity
        new_S2 = get_S2(min(util, 2.0))
        damped = 0.5 * S2 + 0.5 * new_S2
        diff = abs(damped - S2)
        S2 = damped
        if diff < 0.0001 and _ > 0:
            break
    final_S = {}
    for comm in assigned_comms:
        final_S[comm] = 0.2 * S1_values[comm] + 0.3 * S2 + 0.5 * S3_values[comm]
    return S2, final_S


# ==============================
# 迭代辅助函数
# ==============================

def compute_S3_by_station_p4(station_config, station_ids, optimal_prices,
                             optimal_summary, benchmark, communities):
    """计算每个站的加权S3（按需求量加权）"""
    S3_by_station = {}
    for si, st in enumerate(station_config):
        sid = station_ids[si]
        price_dict = optimal_prices.get(si, dict(benchmark))
        total_w = 0.0
        weighted_s3 = 0.0
        for comm in st['comms']:
            sd = optimal_summary.get(si, {}).get(comm, {})
            if not sd:
                continue
            w = sum(sd.get(svc, 0) for svc in SERVICES)
            s3 = compute_S3_community(sd, price_dict, benchmark)
            weighted_s3 += w * s3
            total_w += w
        S3_by_station[sid] = weighted_s3 / total_w if total_w > 0 else 1.0
    return S3_by_station


def compute_S1_by_station_p4(station_config, communities, dist_df):
    """计算每个站的距离满意度S1"""
    S1_by_station = {}
    for si, st in enumerate(station_config):
        loc_idx = communities.index(st['loc'])
        S1_by_station[si] = {}
        for comm in st['comms']:
            c_idx = communities.index(comm)
            S1_by_station[si][comm] = get_S1(float(dist_df.iloc[c_idx, loc_idx]))
    return S1_by_station


def update_assignment_p4(station_config, station_ids, daily_demand, cover_mat,
                         dist_values, caps_plan, n_comm, communities,
                         S3_by_station=None):
    """运行老年人选择模拟，更新station_config中的comms分配"""
    ass, sat, S2_vals = simulate_elderly_choice(
        station_ids, daily_demand, cover_mat, dist_values,
        caps_plan, n_comm, S3_by_station=S3_by_station)
    for st in station_config:
        st['comms'] = []
    for j, i in enumerate(ass):
        if i >= 0:
            st_idx = station_ids.index(i)
            station_config[st_idx]['comms'].append(communities[j])
    return ass, sat, S2_vals


def update_daily_demand_p4(n_comm, communities, ass, station_ids,
                           optimal_summary, original_daily_demand):
    """定价后更新daily_demand"""
    daily_demand = list(original_daily_demand)
    for j in range(n_comm):
        si = ass[j]
        if si >= 0:
            si_idx = station_ids.index(int(si))
            sd = optimal_summary.get(si_idx, {}).get(communities[j], {})
            if sd:
                daily_demand[j] = sum(sd[svc] for svc in SERVICES) / DAYS_MONTH
    return daily_demand


def run_pricing_optimization(station_config, communities, pop_results,
                             summary, demand_df, revenue_df, df_pop,
                             caps_dict, dist_df, station_op_cost, benchmark):
    n_comm = len(communities)
    station_ids = [communities.index(st['loc']) for st in station_config]
    caps_plan = {sid: CAPACITY[st['size']] for sid, st in zip(station_ids, station_config)}

    # 计算原始 daily_demand
    original_daily_demand = []
    for name in communities:
        total_m = sum(summary[name][svc] for svc in SERVICES)
        original_daily_demand.append(total_m / DAYS_MONTH)
    daily_demand = list(original_daily_demand)

    # 读取距离矩阵和覆盖矩阵
    cover_mat = [[False] * n_comm for _ in range(n_comm)]
    dist_values_mat = [[0.0] * n_comm for _ in range(n_comm)]
    for i in range(n_comm):
        for j in range(n_comm):
            d = float(dist_df.iloc[i, j])
            dist_values_mat[i][j] = d
            if d <= RADIUS:
                cover_mat[i][j] = True

    MAX_ITER = 5
    best_avg_S = -1.0
    best_state = None

    # 迭代历史记录（振荡检测 + 定价收敛）
    comms_history = []       # 分配历史（振荡检测）
    prev_optimal_prices = None  # 上轮定价（定价收敛检查）

    for iteration in range(MAX_ITER):
        # Step 1: 计算 S3_by_station
        if iteration == 0:
            S3_by_station = None
        else:
            S3_by_station = compute_S3_by_station_p4(
                station_config, station_ids, optimal_prices,
                optimal_summary, benchmark, communities)

        # 保存旧分配
        old_comms = {st['loc']: sorted(st['comms']) for st in station_config}

        # Step 2: 运行老年人选择
        ass, sat, S2_vals = update_assignment_p4(
            station_config, station_ids, daily_demand, cover_mat,
            dist_values_mat, caps_plan, n_comm, communities, S3_by_station)

        # Step 3: 计算 S1
        S1_by_station = compute_S1_by_station_p4(station_config, communities, dist_df)

        # Step 4: 逐站定价优化
        optimal_prices = {}
        best_station_S2 = {}
        best_community_S = {}
        best_community_S3 = {}
        optimal_summary = {}

        for si, st in enumerate(station_config):
            size = st['size']
            capacity = CAPACITY[size]
            assigned = st['comms']
            best_avg_S_inner = -1.0
            best_result = None
            n_services = len(NON_EMERGENCY)

            for combo in product(PRICE_MULTIPLIERS, repeat=n_services):
                price_dict = dict(benchmark)
                price_dict['紧急救助'] = 0.0
                for k, svc_name in enumerate(NON_EMERGENCY):
                    price_dict[svc_name] = benchmark[svc_name] * combo[k]

                summary_j = {}
                all_feasible = True
                for comm in assigned:
                    sd = compute_demand_at_price(
                        comm, pop_results, demand_df, price_dict, df_pop, caps_dict)
                    if sum(sd.values()) == 0:
                        all_feasible = False
                        break
                    summary_j[comm] = sd
                if not all_feasible:
                    continue

                S3_j = {comm: compute_S3_community(summary_j[comm], price_dict, benchmark)
                        for comm in assigned}
                S1_j = S1_by_station.get(si, {})
                if not S1_j:
                    continue
                S2_val, S_final = converge_S2_for_station(
                    assigned, summary_j, S1_j, S3_j, capacity)

                annual_rev = annual_dir = annual_non_emerg_eff = 0.0
                for comm in assigned:
                    S = S_final[comm]
                    for svc in SERVICES:
                        eff = summary_j[comm][svc] * S * 12
                        if svc == '紧急救助':
                            annual_dir += eff * benchmark[svc]
                        else:
                            annual_rev += eff * price_dict[svc]
                            annual_dir += eff * benchmark[svc]
                            annual_non_emerg_eff += eff

                raw_subsidy = annual_non_emerg_eff * 2.0
                subsidy_cap = DAILY_SUBSIDY_CAP[size] * 365.0
                annual_subsidy = min(raw_subsidy, subsidy_cap)
                build_amortized = BUILD_COST[size] * 10000 / 20.0
                annual_fixed = station_op_cost[size] * 365 + build_amortized
                annual_total_cost = annual_dir + annual_fixed
                profit = annual_rev + annual_subsidy - annual_total_cost
                margin = profit / annual_total_cost if annual_total_cost > 0 else -1.0

                if not (0.0 <= margin <= 0.08):
                    continue

                avg_S = np.mean([S_final[comm] for comm in assigned])
                if avg_S > best_avg_S_inner:
                    best_avg_S_inner = avg_S
                    best_result = {
                        'price_dict': price_dict, 'summary_j': summary_j,
                        'S2': S2_val, 'S_final': S_final, 'S3_j': S3_j,
                        'profit': profit, 'margin': margin, 'subsidy': annual_subsidy,
                    }

            if best_result is None:
                optimal_prices[si] = {svc: benchmark[svc] for svc in SERVICES}
                best_station_S2[si] = 1.0
                best_community_S[si] = {c: 0.6 for c in assigned}
                best_community_S3[si] = {c: 1.0 for c in assigned}
                optimal_summary[si] = {}
            else:
                optimal_prices[si] = best_result['price_dict']
                best_station_S2[si] = best_result['S2']
                best_community_S[si] = best_result['S_final']
                best_community_S3[si] = best_result['S3_j']
                optimal_summary[si] = best_result['summary_j']

        # Step 5: 更新 daily_demand
        daily_demand = update_daily_demand_p4(
            n_comm, communities, ass, station_ids,
            optimal_summary, original_daily_demand)

        # Step 6: 跟踪最佳结果
        all_S = [best_community_S[si][comm]
                 for si, st in enumerate(station_config) for comm in st['comms']]
        avg_S_global = np.mean(all_S) if all_S else 0

        if avg_S_global > best_avg_S:
            best_avg_S = avg_S_global
            best_state = {
                'optimal_prices': dict(optimal_prices),
                'best_station_S2': dict(best_station_S2),
                'best_community_S': {k: dict(v) for k, v in best_community_S.items()},
                'best_community_S3': {k: dict(v) for k, v in best_community_S3.items()},
                'optimal_summary': {k: dict(v) for k, v in optimal_summary.items()},
                'S1_by_station': {k: dict(v) for k, v in S1_by_station.items()},
                'comms': {st['loc']: list(st['comms']) for st in station_config},
            }

        # Step 7: 检查收敛（振荡检测 + 定价收敛 + 分配稳定）
        new_comms = {st['loc']: sorted(st['comms']) for st in station_config}
        new_comms_key = frozenset(
            (loc, tuple(comms)) for loc, comms in new_comms.items()
        )

        # 振荡检测：当前分配与历史中某次相同 → 周期循环
        oscillated = False
        if new_comms_key in comms_history:
            cycle_start = comms_history.index(new_comms_key)
            cycle_len = len(comms_history) - cycle_start
            print(f'  检测到振荡! 周期={cycle_len}, 从第{cycle_start+1}轮开始')
            oscillated = True
        else:
            comms_history.append(new_comms_key)

        # 定价收敛：连续两轮价格完全一致
        price_converged = False
        if prev_optimal_prices is not None:
            price_unchanged = True
            for si in optimal_prices:
                for svc in NON_EMERGENCY:
                    old_p = prev_optimal_prices.get(si, {}).get(svc, -1)
                    new_p = optimal_prices[si].get(svc, -1)
                    if abs(old_p - new_p) > 0.01:
                        price_unchanged = False
                        break
                if not price_unchanged:
                    break
            if price_unchanged:
                price_converged = True
                print(f'  定价收敛! (价格未变化)')
        prev_optimal_prices = {si: dict(prices) for si, prices in optimal_prices.items()}

        # 综合判断：振荡 / 定价收敛 / 分配稳定 → 终止
        if oscillated or price_converged or (old_comms == new_comms and iteration > 0):
            reason = '振荡' if oscillated else ('定价收敛' if price_converged else '分配稳定')
            print(f'  迭代终止 ({reason}), 共{iteration+1}轮')
            break
    else:
        # 未收敛，使用最佳结果
        if best_state:
            optimal_prices = best_state['optimal_prices']
            best_station_S2 = best_state['best_station_S2']
            best_community_S = best_state['best_community_S']
            best_community_S3 = best_state['best_community_S3']
            optimal_summary = best_state['optimal_summary']
            S1_by_station = best_state['S1_by_station']
            for st in station_config:
                st['comms'] = best_state['comms'].get(st['loc'], st['comms'])

    # ---- 最终汇总 ----
    elder_pop_dict = {}
    for name in communities:
        _, c, s, d, _ = pop_results[name][5]
        elder_pop_dict[name] = c + s + d

    grand_profit = grand_subsidy = grand_rev = grand_dir = 0.0
    station_details = []
    for si, st in enumerate(station_config):
        size = st['size']
        assigned = st['comms']
        price_dict = optimal_prices[si]
        S_final = best_community_S[si]
        sum_rev = sum_dir = sum_non_emerg = 0.0
        for comm in assigned:
            S = S_final[comm]
            sd = compute_demand_at_price(
                comm, pop_results, demand_df, price_dict, df_pop, caps_dict)
            for svc in SERVICES:
                eff = sd[svc] * S * 12
                if svc == '紧急救助':
                    sum_dir += eff * benchmark[svc]
                else:
                    sum_rev += eff * price_dict[svc]
                    sum_dir += eff * benchmark[svc]
                    sum_non_emerg += eff
        raw_sb = sum_non_emerg * 2.0
        sb_cap = DAILY_SUBSIDY_CAP[size] * 365.0
        annual_sb = min(raw_sb, sb_cap)
        fx = station_op_cost[size] * 365 + BUILD_COST[size] * 10000 / 20.0
        total_cost = sum_dir + fx
        profit_val = sum_rev + annual_sb - total_cost
        margin_val = profit_val / total_cost if total_cost > 0 else 0
        grand_profit += profit_val
        grand_subsidy += annual_sb
        grand_rev += sum_rev
        grand_dir += sum_dir
        station_details.append({
            'name': f"{st['loc']}({SIZE_LABEL[size]})", 'si': si, 'size': size,
            'assigned': assigned, 'profit_wan': profit_val / 10000,
            'margin_pct': margin_val * 100, 'subsidy_wan': annual_sb / 10000,
            'S2': best_station_S2[si],
        })

    covered_pop = sum(elder_pop_dict[name]
                      for st in station_config for name in st['comms'])
    total_pop = sum(elder_pop_dict.values())
    coverage = covered_pop / total_pop * 100 if total_pop > 0 else 0
    all_S = [best_community_S[sd['si']][comm]
             for sd in station_details for comm in sd['assigned']]
    return {
        'optimal_prices': optimal_prices, 'station_details': station_details,
        'grand_profit': grand_profit, 'grand_subsidy': grand_subsidy,
        'coverage': coverage, 'avg_satisfaction': np.mean(all_S) if all_S else 0,
        'S1_by_station': S1_by_station, 'best_station_S2': best_station_S2,
        'best_community_S': best_community_S, 'best_community_S3': best_community_S3,
    }


# ==============================
# [11] 场景编排
# ==============================

def build_station_config_from_pareto(r, communities, n, daily_demand,
                                     cover_mat, dist_values):
    plan = r['plan']
    station_ids = [i for i in range(n) if plan[i] > 0]
    caps = {i: CAPACITY[plan[i]] for i in station_ids}
    ass, sat, S2_vals = simulate_elderly_choice(
        station_ids, daily_demand, cover_mat, dist_values, caps, n)
    stations = []
    for idx, i in enumerate(station_ids):
        comms = [communities[j] for j in range(n) if ass[j] == i]
        stations.append({
            'loc': communities[i], 'size': plan[i],
            'label': f"{communities[i]}({SIZE_LABEL[plan[i]]})",
            'comms': comms, 'size_label': SIZE_LABEL[plan[i]],
        })
    return stations


def run_scenario(scenario_key, params, n_cores=N_CORES):
    print(f'\n{"=" * 70}')
    print(f'  场景: {params["label"]}')
    print(f'  budget={params["budget"]}万, k={params["k_min"]}..{params["k_max"]}')
    if params['new_rate']:
        print(f'  new_rate={params["new_rate"]}, '
              f'p_stos={params["p_stos"]}, p_tod={params["p_tod"]}')
    print(f'  op_cost_mult={params["op_cost_mult"]}')
    print(f'{"=" * 70}')

    # 1. 数据加载
    df_pop, p1_orig, p2_orig = read_population_data()
    demand_df = read_demand_rates()
    revenue_df = read_revenue()
    caps_dict = read_consumption_caps()
    station_df = read_station_data()
    dist_df = read_distance()
    communities = df_pop['小区'].tolist()
    n = len(communities)

    # 2. 人口预测
    print('\n  [1/5] 人口预测...')
    p1 = params['p_stos'] if params['p_stos'] is not None else p1_orig
    p2 = params['p_tod'] if params['p_tod'] is not None else p2_orig
    if params['new_rate']:
        config = ModelConfig(new_rate=params['new_rate'])
    else:
        config = ModelConfig()
    pop_results = predict_population(df_pop, p1, p2, years=5, config=config)
    elderly_pop = [sum(pop_results[name][5][1:4]) for name in communities]

    # 3. 需求计算
    print('  [2/5] 需求计算...')
    summary = compute_actual_demand(pop_results, demand_df, revenue_df,
                                    df_pop, caps_dict)
    daily_demand = []
    for name in communities:
        total_m = sum(summary[name][svc] for svc in SERVICES)
        daily_demand.append(total_m / DAYS_MONTH)

    # 4. 成本
    station_op_cost = get_station_op_cost(station_df,
                                          multiplier=params['op_cost_mult'])
    rev_dict, cost_dict = get_price_data(revenue_df)
    benchmark = load_benchmark(revenue_df)

    # 5. 覆盖矩阵 + 聚类
    print('  [3/5] 枚举...')
    cover_mat, dist_values = build_cover_matrix(dist_df, communities)
    cluster_labels, cluster_info = perform_clustering(
        communities, dist_values, elderly_pop, daily_demand, params['budget'])

    # 6. 枚举
    all_plans = enumerate_plans(n, budget=params['budget'],
                                k_min=params['k_min'], k_max=params['k_max'],
                                cluster_labels=cluster_labels,
                                cluster_info=cluster_info)
    print(f'  枚举方案数: {len(all_plans)}')

    # 7. 并行求解
    print('  [4/5] 并行求解...')
    pareto = solve_all_plans(all_plans, ALPHAS, daily_demand, elderly_pop,
                             cover_mat, dist_values, n, budget=params['budget'])

    # 8. 利润计算
    print('  [5/5] 利润计算...')
    compute_profit_for_pareto(pareto, ALPHAS, communities, summary,
                              rev_dict, cost_dict, station_op_cost,
                              daily_demand, n)

    return {
        'scenario_key': scenario_key, 'label': params['label'],
        'params': params, 'pareto': pareto, 'alphas': ALPHAS,
        'communities': communities, 'n': n, 'elderly_pop': elderly_pop,
        'daily_demand': daily_demand, 'summary': summary,
        'station_op_cost': station_op_cost, 'rev_dict': rev_dict,
        'cost_dict': cost_dict, 'pop_results': pop_results,
        'demand_df': demand_df, 'revenue_df': revenue_df,
        'df_pop': df_pop, 'caps_dict': caps_dict,
        'station_df': station_df, 'dist_df': dist_df,
        'cover_mat': cover_mat, 'dist_values': dist_values,
        'benchmark': benchmark,
        'p1_orig': p1_orig, 'p2_orig': p2_orig,  # 保存实际基线转移概率
    }


# ==============================
# [12] 对比输出
# ==============================

def output_comparison(all_results, filepath=None):
    if filepath is None:
        filepath = os.path.join(BASE_DIR, '问题4_灵敏度对比.xlsx')

    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        # Sheet 1: 全量 Pareto 对比
        overview_rows = []
        for key, res in all_results.items():
            for alpha in res['alphas']:
                r = res['pareto'].get(alpha)
                if r is None or r['plan'] is None:
                    continue
                overview_rows.append({
                    '场景': res['label'], 'α': alpha,
                    '站数': sum(1 for x in r['plan'] if x > 0),
                    '建设成本(万)': r.get('cost', 0),
                    '覆盖率': round(r['cov_ratio'], 4),
                    '满意度': round(r['avg_s'], 4),
                    '总利润(万)': round(r.get('total_profit', 0) / 10000, 2),
                })
        pd.DataFrame(overview_rows).to_excel(
            writer, sheet_name='Pareto全量对比', index=False)

        # Sheet 2: 代表方案对比 (P2 + P3)
        rep_rows = []
        pricing_rows = []
        p3_results = {}
        for key, res in all_results.items():
            best_alpha = select_knee_alpha(res['pareto'], res['alphas'])
            if best_alpha is None:
                continue
            r = res['pareto'][best_alpha]
            plan = r['plan']
            locs = [f"{res['communities'][i]}({SIZE_LABEL[plan[i]]})"
                    for i in range(res['n']) if plan[i] > 0]

            # 定价优化 (Problem 3)
            station_config = build_station_config_from_pareto(
                r, res['communities'], res['n'],
                res['daily_demand'], res['cover_mat'], res['dist_values'])
            p3 = run_pricing_optimization(
                station_config, res['communities'], res['pop_results'],
                res['summary'], res['demand_df'], res['revenue_df'],
                res['df_pop'], res['caps_dict'], res['dist_df'],
                res['station_op_cost'], res['benchmark'])
            p3_results[key] = {'p3': p3, 'station_config': station_config, 'res': res}

            # 计算平均定价乘子
            all_mults = []
            for si, st in enumerate(station_config):
                for svc in NON_EMERGENCY:
                    bp = res['benchmark'][svc]
                    op = p3['optimal_prices'][si][svc]
                    all_mults.append(op / bp)
            avg_mult = np.mean(all_mults) if all_mults else 1.0

            # P2 + P3 对比行
            rep_rows.append({
                '场景': res['label'],
                '代表α(膝点法)': best_alpha,
                '站数': r['n_stations'],
                '站点配置': '+'.join(locs),
                '建设成本(万)': r.get('cost', 0),
                'P2覆盖率(%)': round(r['cov_ratio'] * 100, 2),
                'P2满意度': round(r['avg_s'], 4),
                'P2利润(万)': round(r.get('total_profit', 0) / 10000, 2),
                'P3补贴总额(万)': round(p3['grand_subsidy'] / 10000, 2),
                'P3利润(万)': round(p3['grand_profit'] / 10000, 2),
                'P3覆盖率(%)': round(p3['coverage'], 2),
                'P3满意度': round(p3['avg_satisfaction'], 4),
                '平均定价乘子': round(avg_mult, 3),
            })

            # 定价明细行
            for si, st in enumerate(station_config):
                for svc in SERVICES:
                    bp = res['benchmark'][svc]
                    if svc == '紧急救助':
                        pricing_rows.append({
                            '场景': res['label'], '站': st['label'],
                            '服务': svc, '基准价': bp, '最优价': 0.0,
                            '乘子': 0.0, '级别': '免费', 'S3': 1.0,
                        })
                    else:
                        op = p3['optimal_prices'][si][svc]
                        mult = op / bp
                        pricing_rows.append({
                            '场景': res['label'], '站': st['label'],
                            '服务': svc, '基准价': bp,
                            '最优价': round(op, 1),
                            '乘子': round(mult, 2),
                            '级别': get_price_level_label(mult),
                            'S3': get_S3_from_mult(mult),
                        })

        pd.DataFrame(rep_rows).to_excel(
            writer, sheet_name='代表方案对比', index=False)
        pd.DataFrame(pricing_rows).to_excel(
            writer, sheet_name='定价对比', index=False)

        # Sheet 3: 可及性分析 (Problem 3.3 对每个场景)
        for key, p3_info in p3_results.items():
            p3 = p3_info['p3']
            station_config = p3_info['station_config']
            res = p3_info['res']
            acc_df = run_accessibility_analysis(
                p3, station_config, res['communities'], res['pop_results'],
                res['demand_df'], res['revenue_df'], res['df_pop'],
                res['caps_dict'], res['benchmark'])
            sheet_name = f'可及性_{key}'
            acc_df.to_excel(writer, sheet_name=sheet_name, index=False)

        # Sheet 4: 敏感性分析
        analyze_sensitivity(all_results, writer)
        # Sheet 5: 鲁棒性分析
        analyze_robustness(all_results, writer)

    print(f'\n对比Excel已保存: {filepath}')

    # 绘制对比图
    plot_comparison_charts(all_results, p3_results)


# ==============================
# [13] 敏感性 + 鲁棒性分析
# ==============================

def analyze_sensitivity(all_results, writer):
    baseline = all_results.get('baseline')
    if baseline is None:
        return
    best_alpha_base = select_knee_alpha(baseline['pareto'], baseline['alphas'])
    if best_alpha_base is None:
        return
    r_base = baseline['pareto'][best_alpha_base]
    rows = []
    for key in ['S1', 'S2', 'S3']:
        scenario = all_results.get(key)
        if scenario is None:
            continue
        best_alpha_scen = select_knee_alpha(scenario['pareto'], scenario['alphas'])
        if best_alpha_scen is None:
            continue
        r_scen = scenario['pareto'][best_alpha_scen]
        if key == 'S1':
            # 使用实际基线转移概率而非硬编码值
            p1_base = baseline.get('p1_orig', 0.045)
            p2_base = baseline.get('p2_orig', 0.100)
            delta_params = {
                'new_rate': (0.08 - 0.07) / 0.07,
                'p_stos': (0.055 - p1_base) / p1_base,
                'p_tod': (0.095 - p2_base) / p2_base,
            }
        elif key == 'S2':
            delta_params = {'op_cost': 0.2}
        else:
            delta_params = {'budget': (140 - 120) / 120}
        max_delta = max(abs(v) for v in delta_params.values())
        cov_base, sat_base = r_base['cov_ratio'], r_base['avg_s']
        cov_scen, sat_scen = r_scen['cov_ratio'], r_scen['avg_s']
        cov_elast = ((cov_scen - cov_base) / cov_base) / max_delta \
            if cov_base > 0 and max_delta > 0 else 0
        sat_elast = ((sat_scen - sat_base) / sat_base) / max_delta \
            if sat_base > 0 and max_delta > 0 else 0
        rows.append({
            '场景': key,
            '参数变化': ', '.join(f'{k}={v:.2%}' for k, v in delta_params.items()),
            '基线覆盖率': round(cov_base, 4),
            '场景覆盖率': round(cov_scen, 4),
            '覆盖率变化': round(cov_scen - cov_base, 4),
            '覆盖率弹性': round(cov_elast, 4),
            '基线满意度': round(sat_base, 4),
            '场景满意度': round(sat_scen, 4),
            '满意度变化': round(sat_scen - sat_base, 4),
            '满意度弹性': round(sat_elast, 4),
        })
    pd.DataFrame(rows).to_excel(writer, sheet_name='敏感性分析', index=False)


def analyze_robustness(all_results, writer):
    baseline = all_results.get('baseline')
    if baseline is None:
        return
    best_alpha_base = select_knee_alpha(baseline['pareto'], baseline['alphas'])
    if best_alpha_base is None:
        return
    r_base = baseline['pareto'][best_alpha_base]
    rows = []
    for key in ['S1', 'S2', 'S3']:
        scenario = all_results.get(key)
        if scenario is None:
            continue
        best_alpha_scen = select_knee_alpha(scenario['pareto'], scenario['alphas'])
        if best_alpha_scen is None:
            continue
        r_scen = scenario['pareto'][best_alpha_scen]
        cov_base, sat_base = r_base['cov_ratio'], r_base['avg_s']
        cov_scen, sat_scen = r_scen['cov_ratio'], r_scen['avg_s']
        cov_pct = abs(cov_scen - cov_base) / cov_base * 100 if cov_base > 0 else 0
        sat_pct = abs(sat_scen - sat_base) / sat_base * 100 if sat_base > 0 else 0

        def level(p):
            if p < 5:
                return '高'
            elif p < 15:
                return '中等'
            else:
                return '低'

        rows.append({
            '场景': key,
            '覆盖率变化(%)': round(cov_pct, 2),
            '覆盖率鲁棒性': level(cov_pct),
            '满意度变化(%)': round(sat_pct, 2),
            '满意度鲁棒性': level(sat_pct),
            '综合评价': '整体稳定' if max(cov_pct, sat_pct) < 10 else '需要关注',
        })
    pd.DataFrame(rows).to_excel(writer, sheet_name='鲁棒性分析', index=False)


# ==============================
# 可及性分析 (Problem 3.3)
# ==============================

def run_accessibility_analysis(p3, station_config, communities, pop_results,
                               demand_df, revenue_df, df_pop, caps_dict, benchmark):
    """对每个场景的代表方案运行可及性分析"""
    original_prices = dict(zip(revenue_df['服务项目'], revenue_df['营收']))
    demand_rate = {}
    for _, row in demand_df.iterrows():
        svc = row['服务项目']
        demand_rate[svc] = {
            '自理': float(row['自理']),
            '半失能': float(row['半失能']),
            '失能': float(row['失能']),
        }

    rows = []
    for si, st in enumerate(station_config):
        assigned = st['comms']
        price_dict = p3['optimal_prices'][si]
        S_final = p3['best_community_S'][si]
        for comm in assigned:
            income = float(df_pop.loc[df_pop['小区'] == comm, '人均月收入'].iloc[0])
            _, c, s, d, _ = pop_results[comm][5]
            pops = {'自理': float(c), '半失能': float(s), '失能': float(d)}
            S_comm = S_final.get(comm, 0)
            for typ in TYPES:
                cap = income * caps_dict[typ]
                pop = pops[typ]
                if pop == 0:
                    continue
                per_cap_orig = sum(demand_rate[svc][typ] * original_prices[svc] for svc in SERVICES)
                factor_orig = 1.0 if per_cap_orig <= cap else cap / per_cap_orig
                oop_orig = min(per_cap_orig, cap)
                per_cap_opt = sum(demand_rate[svc][typ] * price_dict[svc] for svc in SERVICES)
                factor_opt = 1.0 if per_cap_opt <= cap else cap / per_cap_opt
                oop_opt = min(per_cap_opt, cap)
                rows.append({
                    '社区': comm, '类型': typ, '人口': int(pop), '收入': income,
                    '月消费上限': round(cap, 1),
                    '原价月消费': round(per_cap_orig, 1),
                    '现价月消费': round(per_cap_opt, 1),
                    '原缩减因子': round(factor_orig, 4),
                    '现缩减因子': round(factor_opt, 4),
                    '原月支出': round(oop_orig, 1),
                    '现月支出': round(oop_opt, 1),
                    '月节省': round(oop_orig - oop_opt, 1),
                    'S满意度': round(S_comm, 4),
                })
    return pd.DataFrame(rows)


# ==============================
# 对比图表
# ==============================

def plot_comparison_charts(all_results, p3_results):
    """绘制4张对比图"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except ImportError:
        print('  [警告] matplotlib 未安装，跳过绘图')
        return

    # 收集数据
    scenario_keys = []
    scenario_labels = []
    cov_p2 = []
    sat_p2 = []
    profit_p2 = []
    subsidy_p3 = []
    profit_p3 = []
    cov_p3 = []
    sat_p3 = []
    pricing_data = {svc: [] for svc in NON_EMERGENCY}

    for key, res in all_results.items():
        best_alpha = select_knee_alpha(res['pareto'], res['alphas'])
        if best_alpha is None:
            continue
        r = res['pareto'][best_alpha]
        scenario_keys.append(key)
        scenario_labels.append(key)
        cov_p2.append(r['cov_ratio'] * 100)
        sat_p2.append(r['avg_s'])
        profit_p2.append(r.get('total_profit', 0) / 10000)

        if key in p3_results:
            p3 = p3_results[key]['p3']
            subsidy_p3.append(p3['grand_subsidy'] / 10000)
            profit_p3.append(p3['grand_profit'] / 10000)
            cov_p3.append(p3['coverage'])
            sat_p3.append(p3['avg_satisfaction'])
            station_config = p3_results[key]['station_config']
            benchmark = p3_results[key]['res']['benchmark']
            for svc in NON_EMERGENCY:
                mults = []
                for si in range(len(station_config)):
                    bp = benchmark[svc]
                    op = p3['optimal_prices'][si][svc]
                    mults.append(op / bp)
                pricing_data[svc].append(np.mean(mults))
        else:
            subsidy_p3.append(0)
            profit_p3.append(0)
            cov_p3.append(0)
            sat_p3.append(0)
            for svc in NON_EMERGENCY:
                pricing_data[svc].append(1.0)

    n_scenarios = len(scenario_keys)
    if n_scenarios == 0:
        return

    # ---- 图1: 帕累托前沿对比 ----
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']
    markers = ['o', 's', '^', 'D']
    for idx, (key, res) in enumerate(all_results.items()):
        covs = []
        sats = []
        for alpha in res['alphas']:
            r = res['pareto'].get(alpha)
            if r is None or r['plan'] is None:
                continue
            covs.append(r['cov_ratio'] * 100)
            sats.append(r['avg_s'])
        if covs:
            ax.scatter(covs, sats, c=colors[idx % 4], marker=markers[idx % 4],
                       label=key, s=60, alpha=0.8, edgecolors='white', linewidth=0.5)
    ax.set_xlabel('覆盖率 (%)', fontsize=12)
    ax.set_ylabel('满意度评分', fontsize=12)
    ax.set_title('帕累托前沿对比', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(BASE_DIR, '问题4_帕累托前沿对比.png'), dpi=150)
    plt.close(fig)

    # ---- 图2: 关键指标对比柱状图 ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    x = np.arange(n_scenarios)
    width = 0.35

    # P2 覆盖率
    axes[0, 0].bar(x, cov_p2, width, color='#2196F3', label='P2')
    if any(c > 0 for c in cov_p3):
        axes[0, 0].bar(x + width, cov_p3, width, color='#FF5722', label='P3')
    axes[0, 0].set_title('覆盖率 (%)', fontsize=12)
    axes[0, 0].set_xticks(x + width / 2)
    axes[0, 0].set_xticklabels(scenario_labels, fontsize=9)
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)

    # P2 满意度
    axes[0, 1].bar(x, sat_p2, width, color='#2196F3', label='P2')
    if any(s > 0 for s in sat_p3):
        axes[0, 1].bar(x + width, sat_p3, width, color='#FF5722', label='P3')
    axes[0, 1].set_title('满意度评分', fontsize=12)
    axes[0, 1].set_xticks(x + width / 2)
    axes[0, 1].set_xticklabels(scenario_labels, fontsize=9)
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)

    # P3 补贴总额
    axes[1, 0].bar(x, subsidy_p3, width, color='#4CAF50')
    axes[1, 0].set_title('P3政府补贴 (万元)', fontsize=12)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(scenario_labels, fontsize=9)
    axes[1, 0].grid(axis='y', alpha=0.3)

    # 利润对比
    axes[1, 1].bar(x, profit_p2, width, color='#2196F3', label='P2')
    axes[1, 1].bar(x + width, profit_p3, width, color='#FF5722', label='P3')
    axes[1, 1].set_title('利润 (万元)', fontsize=12)
    axes[1, 1].set_xticks(x + width / 2)
    axes[1, 1].set_xticklabels(scenario_labels, fontsize=9)
    axes[1, 1].legend()
    axes[1, 1].grid(axis='y', alpha=0.3)

    fig.suptitle('各场景关键指标对比', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(BASE_DIR, '问题4_关键指标对比.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---- 图3: 定价对比柱状图 ----
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(n_scenarios)
    width = 0.15
    svc_colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800']
    for idx, svc in enumerate(NON_EMERGENCY):
        ax.bar(x + idx * width, pricing_data[svc], width,
               color=svc_colors[idx], label=svc)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='基准价格')
    ax.set_xlabel('场景', fontsize=12)
    ax.set_ylabel('价格乘子', fontsize=12)
    ax.set_title('各场景定价对比', fontsize=14)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(scenario_labels, fontsize=10)
    ax.legend(fontsize=9, ncol=3)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(BASE_DIR, '问题4_定价对比.png'), dpi=150)
    plt.close(fig)

    # ---- 图4: 敏感性弹性系数 ----
    baseline = all_results.get('baseline')
    if baseline is not None:
        best_alpha_base = select_knee_alpha(baseline['pareto'], baseline['alphas'])
        if best_alpha_base is not None:
            r_base = baseline['pareto'][best_alpha_base]
            elasticity_data = []
            for key in ['S1', 'S2', 'S3']:
                scenario = all_results.get(key)
                if scenario is None:
                    continue
                best_alpha_scen = select_knee_alpha(scenario['pareto'], scenario['alphas'])
                if best_alpha_scen is None:
                    continue
                r_scen = scenario['pareto'][best_alpha_scen]
                if key == 'S1':
                    p1_base = baseline.get('p1_orig', 0.045)
                    p2_base = baseline.get('p2_orig', 0.100)
                    max_delta = max(abs((0.08 - 0.07) / 0.07),
                                    abs((0.055 - p1_base) / p1_base),
                                    abs((0.095 - p2_base) / p2_base))
                elif key == 'S2':
                    max_delta = 0.2
                else:
                    max_delta = (140 - 120) / 120
                cov_base, sat_base = r_base['cov_ratio'], r_base['avg_s']
                cov_scen, sat_scen = r_scen['cov_ratio'], r_scen['avg_s']
                cov_elast = ((cov_scen - cov_base) / cov_base) / max_delta if cov_base > 0 and max_delta > 0 else 0
                sat_elast = ((sat_scen - sat_base) / sat_base) / max_delta if sat_base > 0 and max_delta > 0 else 0
                elasticity_data.append({'scenario': key, 'cov': cov_elast, 'sat': sat_elast})

            if elasticity_data:
                fig, ax = plt.subplots(figsize=(8, 5))
                x = np.arange(len(elasticity_data))
                width = 0.35
                labels = [d['scenario'] for d in elasticity_data]
                ax.bar(x - width / 2, [d['cov'] for d in elasticity_data],
                       width, color='#2196F3', label='覆盖率弹性')
                ax.bar(x + width / 2, [d['sat'] for d in elasticity_data],
                       width, color='#FF5722', label='满意度弹性')
                ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
                ax.axhline(y=1, color='gray', linestyle='--', alpha=0.3, label='弹性=1')
                ax.axhline(y=-1, color='gray', linestyle='--', alpha=0.3)
                ax.set_xlabel('场景', fontsize=12)
                ax.set_ylabel('弹性系数', fontsize=12)
                ax.set_title('敏感性分析：关键指标弹性系数', fontsize=14)
                ax.set_xticks(x)
                ax.set_xticklabels(labels, fontsize=10)
                ax.legend(fontsize=10)
                ax.grid(axis='y', alpha=0.3)
                fig.tight_layout()
                fig.savefig(os.path.join(BASE_DIR, '问题4_敏感性弹性.png'), dpi=150)
                plt.close(fig)

    print('  对比图表已保存')


# ==============================
# [14] 不确定因素讨论
# ==============================

def discuss_uncertainties():
    return [
        ('土地与场地获取难度',
         '选定小区内可能无可用场地建设服务站，导致选址方案失效。',
         '建立备选站点库：为每个主选站点配备1-2个邻近小区作为备选。'),
        ('政府补贴政策调整',
         '补贴标准降低或利润率限制收紧会减少可行定价空间，影响服务站可持续运营。',
         '设计弹性定价区间：预留±15%定价调整余量，应对补贴政策变动。'),
        ('老年人消费习惯/需求变化',
         '实际服务使用率可能低于理论需求，导致服务站收入不足。',
         '分期建设：首期按预测的80%容量配置，预留扩展空间。建立需求监测反馈机制。'),
        ('竞争对手/其他养老设施',
         '其他养老机构进入市场会分流需求，降低本方案覆盖率。',
         '竞争因子建模：在需求预测中引入市场占有率因子，定期评估竞争态势。'),
        ('通胀与工资上涨',
         '运营成本随通胀上升，可能突破利润率约束。',
         '年度动态调整机制：将成本指数纳入定价模型，自动触发价格调整。'),
    ]


# ==============================
# [15] 主入口
# ==============================

def main():
    print('=' * 70)
    print('问题4：灵敏度分析与方案比较（完全自包含）')
    print('=' * 70)

    all_results = {}
    for key, params in SCENARIOS.items():
        result = run_scenario(key, params)
        all_results[key] = result

    # 输出对比表
    output_comparison(all_results)

    # 不确定因素
    factors = discuss_uncertainties()
    print('\n' + '=' * 70)
    print('不确定因素及应对策略')
    print('=' * 70)
    for i, (name, impact, strategy) in enumerate(factors, 1):
        print(f'\n{i}. {name}')
        print(f'   影响: {impact}')
        print(f'   策略: {strategy}')

    # ---- Excel输出（参考output_comparison的正确数据获取方式）----
    excel_path = os.path.join(BASE_DIR, '问题4_灵敏度对比_场景分析.xlsx')
    try:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Sheet 1: 场景汇总对比
            summary_rows = []
            pricing_rows = []
            
            for key, res in all_results.items():
                params = SCENARIOS[key]
                best_alpha = select_knee_alpha(res['pareto'], res['alphas'])
                if best_alpha is None:
                    continue
                r = res['pareto'][best_alpha]
                plan = r['plan']
                locs = [f"{res['communities'][i]}({SIZE_LABEL[plan[i]]})"
                        for i in range(res['n']) if plan[i] > 0]
                
                # 定价优化
                station_config = build_station_config_from_pareto(
                    r, res['communities'], res['n'],
                    res['daily_demand'], res['cover_mat'], res['dist_values'])
                p3 = run_pricing_optimization(
                    station_config, res['communities'], res['pop_results'],
                    res['summary'], res['demand_df'], res['revenue_df'],
                    res['df_pop'], res['caps_dict'], res['dist_df'],
                    res['station_op_cost'], res['benchmark'])
                
                # 计算平均定价乘子
                all_mults = []
                for si, st in enumerate(station_config):
                    for svc in NON_EMERGENCY:
                        bp = res['benchmark'][svc]
                        op = p3['optimal_prices'][si][svc]
                        all_mults.append(op / bp)
                avg_mult = np.mean(all_mults) if all_mults else 1.0
                
                # 站点配置字符串
                station_config_str = '; '.join(
                    f"{st['label']}→{','.join(st['comms'])}"
                    for st in station_config
                )
                
                # 汇总行
                summary_rows.append({
                    '场景': params['label'],
                    '代表α(膝点法)': best_alpha,
                    '站数': r['n_stations'],
                    '站点配置': station_config_str,
                    '建设成本(万)': r.get('cost', 0),
                    'P2覆盖率(%)': round(r['cov_ratio'] * 100, 2),
                    'P2满意度': round(r['avg_s'], 4),
                    'P2利润(万)': round(r.get('total_profit', 0) / 10000, 2),
                    'P3补贴总额(万)': round(p3['grand_subsidy'] / 10000, 2),
                    'P3利润(万)': round(p3['grand_profit'] / 10000, 2),
                    'P3覆盖率(%)': round(p3['coverage'], 2),
                    'P3满意度': round(p3['avg_satisfaction'], 4),
                    '平均定价乘子': round(avg_mult, 3),
                })
                
                # 定价明细行
                for si, st in enumerate(station_config):
                    for svc in SERVICES:
                        bp = res['benchmark'][svc]
                        if svc == '紧急救助':
                            pricing_rows.append({
                                '场景': params['label'], '站': st['label'],
                                '服务': svc, '基准价': bp, '最优价': 0.0,
                                '乘子': 0.0, '级别': '免费', 'S3': 1.0,
                            })
                        else:
                            op = p3['optimal_prices'][si][svc]
                            mult = op / bp
                            pricing_rows.append({
                                '场景': params['label'], '站': st['label'],
                                '服务': svc, '基准价': bp,
                                '最优价': round(op, 1),
                                '乘子': round(mult, 2),
                                '级别': get_price_level_label(mult),
                                'S3': get_S3_from_mult(mult),
                            })
            
            # 写入Excel
            if summary_rows:
                pd.DataFrame(summary_rows).to_excel(writer, sheet_name='场景汇总对比', index=False)
            if pricing_rows:
                pd.DataFrame(pricing_rows).to_excel(writer, sheet_name='定价对比', index=False)
        
        print(f'\nExcel已导出: {excel_path}')
    except Exception as e:
        print(f'\nExcel导出失败: {e}')

    print(f'\n{"=" * 70}')
    print('问题4完成!')
    print(f'{"=" * 70}')


if __name__ == '__main__':
    main()
