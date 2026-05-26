"""
问题2.1: 服务站选址与分配优化
- 枚举: k=2..6 按站数枚举，预算≤120万
- 求解: 枚举+多策略种子(4策略贪心+S0)+α感知爬山
- α: 20个 (自适应步长, 0.0~1.0 含0.45/0.55)
- S2: 基于利用率分段函数，每3轮爬山重收敛
- 并行: 16进程
"""
import sys, time, os, random
from itertools import combinations, product
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from problem1_1 import read_data, predict_population
from problem1_3 import (
    read_demand_rates, read_revenue, read_consumption_caps,
    compute_actual_demand, SERVICES,
)

# ==============================
# 常量
# ==============================
CAPACITY = [0, 1000, 2000, 3000]
BUILD_COST = [0, 18, 32, 45]
RADIUS = 1000
DAYS_MONTH = 30
SIZE_LABEL = ['', '小', '中', '大']
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '题目')
BUDGET = 120
S3 = 1.0  # 问题2假设平价策略
ALPHAS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
          0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
KEY_ALPHAS = sorted([a for a in [0.0, 0.25, 0.5, 0.75, 1.0] if a in ALPHAS])
N_CORES = 16
USE_CLUSTER_FILTER = True
CLUSTER_THRESHOLD = 700
CLUSTER_BUDGET_RATIO = 1.5


# ==============================
from problem2_1_plot import (
    _setup_chinese_font, plot_connectivity_graph, plot_dendrogram,
    plot_mds_coverage, plot_pareto_scatter,
)


# ==============================
# Phase 0a: 聚类过滤
# ==============================

def _simple_distance_clustering(communities, dist_values, elderly_pop, daily_demand):
    n = len(communities)
    if n < 2:
        labels = [0] * n
        cluster_info = _build_cluster_info(labels, communities, elderly_pop, daily_demand)
        return labels, cluster_info

    selected = [0]
    farthest = max(range(n), key=lambda j: dist_values[selected[0]][j])
    selected.append(farthest)

    labels = []
    for i in range(n):
        d0 = dist_values[i][selected[0]]
        d1 = dist_values[i][selected[1]]
        labels.append(0 if d0 <= d1 else 1)

    unique_labels = sorted(set(labels))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    labels = [label_map[l] for l in labels]

    cluster_info = _build_cluster_info(labels, communities, elderly_pop, daily_demand)
    return labels, cluster_info


def perform_clustering(communities, dist_values, elderly_pop, daily_demand):
    try:
        import scipy.cluster.hierarchy as sch
        import scipy.spatial.distance as ssd
    except ImportError:
        print('  [聚类分析] 缺少 scipy，使用基于距离矩阵的连通分量聚类')
        return _simple_distance_clustering(communities, dist_values, elderly_pop, daily_demand)

    n = len(communities)
    dist_array = ssd.squareform(np.array(dist_values, dtype=float))
    Z = sch.linkage(dist_array, method='average')
    labels = sch.fcluster(Z, t=CLUSTER_THRESHOLD, criterion='distance')
    labels = [int(x) - 1 for x in labels]
    unique_labels = sorted(set(labels))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    labels = [label_map[l] for l in labels]

    cluster_info = _build_cluster_info(labels, communities, elderly_pop, daily_demand)
    return labels, cluster_info


def _build_cluster_info(labels, communities, elderly_pop, daily_demand):
    n_clusters = max(labels) + 1
    cluster_info = {}
    for cid in range(n_clusters):
        members = [i for i, l in enumerate(labels) if l == cid]
        cluster_info[cid] = {
            'members': members,
            'communities': [communities[i] for i in members],
            'total_pop': sum(elderly_pop[i] for i in members),
            'total_demand': sum(daily_demand[i] for i in members),
            'n_members': len(members),
        }
    return cluster_info


def _filter_plan_by_cluster(plan, cluster_labels, cluster_info):
    n = len(plan)
    n_clusters = max(cluster_labels) + 1

    station_in_cluster = [False] * n_clusters
    cost_in_cluster = [0.0] * n_clusters

    for i in range(n):
        if plan[i] > 0:
            cid = cluster_labels[i]
            station_in_cluster[cid] = True
            cost_in_cluster[cid] += BUILD_COST[plan[i]]

    for cid in range(n_clusters):
        if not station_in_cluster[cid]:
            return False
        pop_ratio = cluster_info[cid]['total_pop'] / max(
            sum(ci['total_pop'] for ci in cluster_info.values()), 1)
        max_cost = pop_ratio * BUDGET * CLUSTER_BUDGET_RATIO
        if cost_in_cluster[cid] > max_cost:
            return False

    return True


# ==============================
# Phase 1: 枚举
# ==============================

def enumerate_plans(n, min_k=2, max_k=6, cluster_labels=None, cluster_info=None):
    """按站数 k=2..6 枚举方案，预算过滤"""
    plans = []

    for k in range(min_k, max_k + 1):
        valid_sizes = [s for s in product([1, 2, 3], repeat=k)
                       if sum(BUILD_COST[sz] for sz in s) <= BUDGET]
        for locs in combinations(range(n), k):
            for sizes in valid_sizes:
                plan = [0] * n
                for idx, sz in zip(locs, sizes):
                    plan[idx] = sz
                plans.append(tuple(plan))

    if USE_CLUSTER_FILTER and cluster_labels is not None and cluster_info is not None:
        filtered = []
        for plan in plans:
            if _filter_plan_by_cluster(plan, cluster_labels, cluster_info):
                filtered.append(plan)
        plans = filtered

    random.Random(42).shuffle(plans)
    return plans


# ==============================
# Phase 2: 多策略贪心种子 + α感知爬山
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


def read_station_data():
    path = os.path.join(DATA_DIR, '附件3：服务站建设与运营成本.xlsx')
    return pd.read_excel(path, sheet_name=0, header=1,
                         names=['规模', '建设成本_万元', '日管理成本_元', '日容量'],
                         usecols=[0, 1, 2, 3])


def read_distance():
    path = os.path.join(DATA_DIR, '附件4：小区间距离矩阵.xlsx')
    df = pd.read_excel(path, sheet_name=0, header=1)
    df = df.set_index('组别')
    df.index.name = None
    df.columns = [c.strip() for c in df.columns]
    return df


def build_cover_matrix(dist_df, communities, radius=RADIUS):
    """
    构建覆盖矩阵与距离值矩阵。
    - 尝试按 communities 顺序重排 dist_df 的行/列；若无法匹配则抛出有说明性的错误。
    - 缺失或不可转为数值的距离视为极大值（不可覆盖）。
    """
    n = len(communities)

    # 尝试直接按名称重排（容忍字符串首尾空格）
    idx = [str(x).strip() for x in dist_df.index]
    cols = [str(x).strip() for x in dist_df.columns]

    if set(communities).issubset(set(idx)) and set(communities).issubset(set(cols)):
        tmp = dist_df.copy()
        tmp.index = idx
        tmp.columns = cols
        df = tmp.reindex(index=communities, columns=communities)
    else:
        raise ValueError(
            '距离矩阵的行/列名无法与小区名称对齐。请检查 附件4：小区间距离矩阵.xlsx 的行列标签是否为小区名（A..J）并且顺序/拼写一致。')

    cover = [[False] * n for _ in range(n)]
    dist_values = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            val = df.iat[i, j]
            try:
                d = float(val)
            except Exception:
                d = 1e9
            dist_values[i][j] = d
            if d <= radius:
                cover[i][j] = True
    return cover, dist_values



def simulate_elderly_choice(station_ids, daily_demand, cover_mat, dist_mat,
                             cap_of_plan, n, S3_by_station=None, max_rounds=20):
    """
    老年人选择模拟器
    每个社区从覆盖它的服务站中选择满意度最高的站
    延迟接受匹配 + S2迭代收敛
    S3_by_station: dict {station_id: S3_value}，默认1.0
    """
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

        loads = {i: sum(daily_demand[j] * new_sat[j] for j in range(n) if new_ass[j] == i) for i in station_ids}
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


# ==============================
# Phase 2a: 多策略种子辅助函数
# ==============================

def _compute_S1_matrix(station_ids, cover_mat, dist_values, n):
    n_sta = len(station_ids)
    S1_mat = np.zeros((n_sta, n))
    for idx, i in enumerate(station_ids):
        for j in range(n):
            if cover_mat[i][j]:
                S1_mat[idx, j] = get_S1(dist_values[i][j])
    return S1_mat


def _greedy_one_pass(order, station_ids, caps, daily_demand, cover_mat, S1_mat, n):
    """单趟贪心分配：按顺序遍历社区，选满意度最高的可用站"""
    ass = np.full(n, -1, dtype=np.int32)
    sat = np.zeros(n)
    remain = {i: caps[i] for i in station_ids}
    for j in order:
        best_S, best_i, best_idx = -1.0, -1, -1
        for idx, i in enumerate(station_ids):
            if not cover_mat[i][j]:
                continue
            S = 0.2 * S1_mat[idx, j] + 0.3 * 1.0 + 0.5 * 1.0
            if daily_demand[j] * S <= remain[i] + 1e-9:
                if S > best_S:
                    best_S, best_i, best_idx = S, i, idx
        if best_i >= 0:
            ass[j] = best_i
            sat[j] = best_S
            remain[best_i] -= daily_demand[j] * best_S
    return ass, sat


def _converge_S2(ass, sat, station_ids, caps, daily_demand, cover_mat, dist_values, n, max_rounds=20, S1_mat=None):
    """从初始分配开始迭代收敛S2，返回的sat基于最终S2重算"""
    if S1_mat is None:
        S1_mat = _compute_S1_matrix(station_ids, cover_mat, dist_values, n)
    n_sta = len(station_ids)
    station_to_idx = {i: idx for idx, i in enumerate(station_ids)}
    S2 = np.ones(n_sta)
    for _round in range(max_rounds):
        loads = {i: sum(daily_demand[j] * sat[j] for j in range(n) if ass[j] == i) for i in station_ids}
        new_S2 = np.zeros(n_sta)
        for idx, i in enumerate(station_ids):
            util = loads[i] / caps[i] if caps[i] > 0 else 0
            new_S2[idx] = get_S2(util)
        S2_diff = np.max(np.abs(new_S2 - S2))
        S2 = new_S2
        if S2_diff < 0.01:
            break
        pairs = []
        for j in range(n):
            for idx, i in enumerate(station_ids):
                if cover_mat[i][j]:
                    S = 0.2 * S1_mat[idx, j] + 0.3 * S2[idx] + 0.5 * 1.0
                    pairs.append((j, idx, S))
        pairs.sort(key=lambda x: -x[2])
        new_ass = np.full(n, -1, dtype=np.int32)
        new_sat = np.zeros(n)
        remain = {i: caps[i] for i in station_ids}
        for j, idx, _ in pairs:
            if new_ass[j] >= 0:
                continue
            i = station_ids[idx]
            S_val = 0.2 * S1_mat[idx, j] + 0.3 * S2[idx] + 0.5 * 1.0
            if daily_demand[j] * S_val <= remain[i] + 1e-9:
                new_ass[j] = i
                new_sat[j] = S_val
                remain[i] -= daily_demand[j] * S_val
        if np.array_equal(new_ass, ass):
            ass, sat = new_ass, new_sat
            break
        ass, sat = new_ass, new_sat
    for j in range(n):
        if ass[j] >= 0:
            idx = station_to_idx[ass[j]]
            sat[j] = 0.2 * S1_mat[idx, j] + 0.3 * S2[idx] + 0.5 * 1.0
    loads = {i: sum(daily_demand[j] * sat[j] for j in range(n) if ass[j] == i) for i in station_ids}
    for idx, i in enumerate(station_ids):
        util = loads[i] / caps[i] if caps[i] > 0 else 0
        S2[idx] = get_S2(util)
    for j in range(n):
        if ass[j] >= 0:
            idx = station_to_idx[ass[j]]
            sat[j] = 0.2 * S1_mat[idx, j] + 0.3 * S2[idx] + 0.5 * 1.0
    return ass, sat, S2


def _seed_strategy_cover_desc(n, station_ids, daily_demand, cover_mat):
    """策略1: 可覆盖站数升序 → 需求降序"""
    n_cover = [sum(1 for i in station_ids if cover_mat[i][j]) for j in range(n)]
    return sorted(range(n), key=lambda j: (n_cover[j], -daily_demand[j]))


def _seed_strategy_demand_desc(n, station_ids, daily_demand, cover_mat):
    """策略2: 需求降序 → 可覆盖站数降序"""
    n_cover = [sum(1 for i in station_ids if cover_mat[i][j]) for j in range(n)]
    return sorted(range(n), key=lambda j: (-daily_demand[j], -n_cover[j]))


def _seed_strategy_dist_asc(n, station_ids, daily_demand, cover_mat, dist_values):
    """策略3: 最近站距离升序"""
    min_dist = [min(dist_values[i][j] for i in station_ids if cover_mat[i][j])
                if any(cover_mat[i][j] for i in station_ids) else 9999 for j in range(n)]
    return sorted(range(n), key=lambda j: min_dist[j])


def _seed_strategy_random(n, station_ids, daily_demand, cover_mat):
    """策略4: 随机排列"""
    order = list(range(n))
    random.Random(42).shuffle(order)
    return order


# ==============================
# Phase 2b: 方案C混合爬山（配置优化）
# ==============================


def _verify_constraint(ass, sat, station_ids, caps, daily_demand,
                       cover_mat, dist_values, n):
    """
    验证分配方案是否符合"老人只选择满意度最高的服务站"约束
    返回: (is_valid, true_ass, true_sat)
    """
    true_ass, true_sat, true_S2 = simulate_elderly_choice(
        station_ids, daily_demand, cover_mat, dist_values, caps, n)
    is_valid = np.array_equal(ass, true_ass) and np.allclose(sat, true_sat, atol=1e-6)
    return is_valid, true_ass, true_sat


def _hill_climb_hybrid(plan, station_ids, caps, daily_demand,
                       cover_mat, dist_values, elderly_pop, alpha, n,
                       max_rounds=10, budget=BUDGET):
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


def solve_plan(args):
    """
    多进程worker: 多策略种子 + α感知爬山
    args = (plan, alphas, daily_demand, elderly_pop,
            cover_mat, dist_values, n)
    """
    (plan, alphas, daily_demand, elderly_pop,
     cover_mat, dist_values, n) = args

    station_ids = [i for i, sz in enumerate(plan) if sz > 0]
    if not station_ids or len(station_ids) < 2:
        return None
    caps = {i: CAPACITY[plan[i]] for i in station_ids}

    # Phase 2b: α感知爬山（每个α独立调用混合爬山）
    results = {}
    for alpha in sorted(alphas):
        # 直接调用混合爬山（内部会重新评估）
        ass, sat, S2_arr, final_plan = _hill_climb_hybrid(
            plan, station_ids, caps, daily_demand,
            cover_mat, dist_values, elderly_pop, alpha, n,
            max_rounds=10, budget=BUDGET)
        
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


# ==============================
# Phase 3: 利润计算
# ==============================

def get_price_data(revenue_df):
    """读取营收与直接支出，按题意将“紧急救助”视为公益免费（营收=0），但保留直接支出。"""
    rev_dict = {}
    cost_dict = {}
    for _, row in revenue_df.iterrows():
        svc = row['服务项目']
        rev_str = str(row['营收'])
        cost_str = str(row['直接支出'])
        try:
            rev = float(rev_str.split('（')[0]) if '（' in rev_str else float(rev_str)
        except Exception:
            rev = 0.0
        try:
            cost = float(cost_str.split('（')[0]) if '（' in cost_str else float(cost_str)
        except Exception:
            cost = 0.0

        # 紧急救助为公益免费（题目说明）——营收强制为0，但直接支出仍计入
        if '紧急' in str(svc) or '救助' in str(svc):
            rev = 0.0
        rev_dict[svc] = rev
        cost_dict[svc] = cost
    return rev_dict, cost_dict


def get_station_op_cost(station_df):
    """返回日管理费列表，索引=规模(0/1/2/3)"""
    op_cost = [0.0]
    for _, row in station_df.iterrows():
        val = row.iloc[2]
        if pd.isna(val):
            continue
        op_cost.append(float(val))
    return op_cost


def compute_station_profit(plan, assignment, communities, summary,
                           rev_dict, cost_dict, station_op_cost,
                           sat_dict=None):
    """计算每站年度利润（元）并返回利润率估计（不含政府补贴）。
    返回: (profit_by_station, rev_by_station, cost_by_station, profit_rate_by_station)
    profit_rate按公式近似计算为: profit / annual_fixed *100 (%)（年固定成本作为分母的简化近似）
    """
    n = len(plan)
    profit_by_station = {}
    rev_by_station = {}
    cost_by_station = {}
    profit_rate_by_station = {}

    for i in range(n):
        if plan[i] == 0:
            continue
        assigned = [j for j in range(n) if assignment.get(j) == i]
        annual_rev = 0.0
        annual_dir = 0.0
        for j in assigned:
            cname = communities[j]
            S_j = sat_dict.get(j, 1.0) if sat_dict else 1.0
            for svc in SERVICES:
                monthly = summary[cname].get(svc, 0) * S_j
                annual_rev += monthly * rev_dict.get(svc, 0.0) * 12
                annual_dir += monthly * cost_dict.get(svc, 0.0) * 12

        build = BUILD_COST[plan[i]] * 10000
        daily_op = station_op_cost[plan[i]]
        amortized = build / 20.0
        annual_fixed = daily_op * 365 + amortized

        profit = annual_rev - annual_dir - annual_fixed
        profit_by_station[i] = profit
        rev_by_station[i] = annual_rev
        cost_by_station[i] = annual_dir + annual_fixed

        # 利润率 = 利润 / 年运营成本总额（不含政府补贴）
        try:
            total_cost = annual_dir + annual_fixed
            profit_rate = (profit / total_cost) * 100 if total_cost > 0 else 0.0
        except Exception:
            profit_rate = 0.0
        profit_rate_by_station[i] = profit_rate

    return profit_by_station, rev_by_station, cost_by_station, profit_rate_by_station


# ==============================
# Phase 4: 输出
# ==============================

def export_pareto_excel_v2(pareto, alphas, communities, summary,
                           elderly_pop, rev_dict, cost_dict,
                           station_op_cost, daily_demand, filepath,
                           alpha_knee=None):
    """Excel输出: Overview + 关键α详细 + 膝点α详细"""
    n = len(communities)
    size_label = ['', '小', '中', '大']

    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        # Sheet 1: Overview
        overview = []
        for a in alphas:
            r = pareto.get(a)
            if r is None or r['plan'] is None:
                overview.append({
                    'α': a, '站数': '-', '成本(万)': '-',
                    '覆盖率': '-', '满意度': '-', '得分': '-',
                    '总利润(万)': '-', '未覆盖区': '-',
                })
                continue
            plan = r['plan']
            uncovered = [communities[j] for j in range(n)
                         if j not in r['assignment']]
            locs = [f'{communities[i]}({size_label[plan[i]]})'
                    for i in range(n) if plan[i] > 0]
            overview.append({
                'α': a,
                '站数': sum(1 for x in plan if x > 0),
                '成本(万)': r.get(
                    'cost',
                    sum(BUILD_COST[x] for x in plan if x > 0)),
                '服务覆盖率': f"{r['cov_ratio']*100:.2f}%",
                '地理覆盖率': f"{r.get('geo_cov_ratio',r['cov_ratio'])*100:.2f}%",
                '满意度(全局)': round(r['avg_s'], 4),
                '得分': round(r['score'], 4),
                '总利润(万)': round(
                    r.get('total_profit', 0) / 10000, 2),
                '未覆盖区': ', '.join(uncovered) if uncovered else '无',
                '站点配置': '+'.join(locs),
            })
        pd.DataFrame(overview).to_excel(
            writer, sheet_name='Pareto Overview', index=False)

        # Sheet 2-...: 关键α详细 + 膝点α
        export_alphas = list(KEY_ALPHAS)
        if alpha_knee is not None and alpha_knee not in export_alphas:
            export_alphas.append(alpha_knee)
        for a in export_alphas:
            r = pareto.get(a)
            if r is None or r['plan'] is None:
                continue
            plan = r['plan']
            assignment = r['assignment']
            station_ids = [i for i in range(n) if plan[i] > 0]
            sat_dict = r['sat_dict']

            uncovered = [communities[j] for j in range(n)
                         if j not in assignment]
            locs = [f'{communities[i]}({size_label[plan[i]]})'
                    for i in range(n) if plan[i] > 0]

            info = [
                ['α (Pareto权重)', a],
                ['站数', len(station_ids)],
                ['建设成本(万元)',
                 r.get('cost',
                       sum(BUILD_COST[x] for x in plan if x > 0))],
                ['服务覆盖率', f"{r['cov_ratio']*100:.2f}%"],
                ['地理覆盖率', f"{r.get('geo_cov_ratio',r['cov_ratio'])*100:.2f}%"],
                ['满意度(全局均)', round(r['avg_s'], 4)],
                ['综合得分', round(r['score'], 4)],
                ['总年收入(万元)',
                 round(sum(r.get('station_rev', {}).values())
                       / 10000, 2)],
                ['总年成本(万元)',
                 round(sum(r.get('station_cost', {}).values())
                       / 10000, 2)],
                ['总年利润(万元)',
                 round(r.get('total_profit', 0) / 10000, 2)],
                ['未覆盖社区',
                 ', '.join(uncovered) if uncovered else '无'],
                ['站点配置', '+'.join(locs)],
            ]
            df_info = pd.DataFrame(info, columns=['指标', '值'])

            rows_st = []
            for si in station_ids:
                assigned = [j for j in range(n)
                            if assignment.get(j) == si]
                sz = plan[si]
                rows_st.append({
                    '服务站': communities[si],
                    '规模': size_label[sz],
                    '日容量': CAPACITY[sz],
                    '建设成本(万)': BUILD_COST[sz],
                    '日管理成本(元)': station_op_cost[sz],
                    '覆盖社区': ', '.join(
                        [communities[j] for j in assigned]),
                    '年营收(万)': round(
                        r['station_rev'].get(si, 0) / 10000, 2),
                    '年成本(万)': round(
                        r['station_cost'].get(si, 0) / 10000, 2),
                    '原始负载': f"{sum(daily_demand[j] for j in range(n) if assignment.get(j)==si):.0f}",
                    '有效负载/容量': f"{sum(daily_demand[j] * sat_dict.get(j, 1.0) for j in range(n) if assignment.get(j)==si):.0f}/{CAPACITY[sz]}",
                    '年利润(万)': round(
                        r['station_profit'].get(si, 0) / 10000, 2),
                    '利润率(%)': round(r.get('station_profit_rate', {}).get(si, 0.0), 2),
                    'S2': round(r['S2'].get(si, 0), 4),
                })
            df_st = pd.DataFrame(rows_st)

            rows_cm = []
            for j in range(n):
                si = assignment.get(j)
                rows_cm.append({
                    '社区': communities[j],
                    '归属站': communities[si] if si is not None else '-',
                    'S(总满意度)': round(sat_dict.get(j, 0), 4),
                    '是否覆盖': '是' if si is not None else '否',
                    '老年人口': int(elderly_pop[j]),
                })
            df_cm = pd.DataFrame(rows_cm)

            sheet_name = f'α={a:.2f}'
            startrow = 0
            df_info.to_excel(writer, sheet_name=sheet_name,
                             index=False, startrow=startrow)
            startrow += len(df_info) + 2
            df_st.to_excel(writer, sheet_name=sheet_name,
                            index=False, startrow=startrow)
            startrow += len(df_st) + 2
            df_cm.to_excel(writer, sheet_name=sheet_name,
                           index=False, startrow=startrow)

    print(f'  Excel已保存: {filepath}')


def select_knee_alpha(pareto, alphas):
    """膝点法：从帕累托前沿自动选择最优α"""
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
        kl = (nondom[i]['sat'] - nondom[i - 1]['sat']) / (nondom[i]['cov'] - nondom[i - 1]['cov'] + 1e-10)
        kr = (nondom[i + 1]['sat'] - nondom[i]['sat']) / (nondom[i + 1]['cov'] - nondom[i]['cov'] + 1e-10)
        score = abs(kl - kr) / (1 + abs(kl) + abs(kr))
        if score > best_score:
            best_score, best_alpha = score, nondom[i]['alpha']
    return best_alpha


def print_problem2_3_analysis(pareto, alphas, communities, elderly_pop, daily_demand):
    """问题2.3：最优方案详细分析"""
    n = len(communities)

    alpha_opt = select_knee_alpha(pareto, alphas)
    if alpha_opt is None:
        print('  [问题2.3] 无可行方案')
        return
    r = pareto.get(alpha_opt)

    plan = r['plan']
    assignment = r['assignment']
    sat_dict = r['sat_dict']
    station_ids = [i for i in range(n) if plan[i] > 0]

    print()
    print('=' * 80)
    print('  问题2.3  最优选址方案分析')
    print('=' * 80)
    print(f'\n  最优权重 α = {alpha_opt}（均衡覆盖与满意度）')
    print(f'  {"站点数量:":16} {len(station_ids)} 个')
    print(f'  {"建设总成本:":16} {r.get("cost", sum(BUILD_COST[x] for x in plan if x > 0))} 万元')
    print(f'  {"服务覆盖率:":16} {r["cov_ratio"]*100:.2f}%')
    print(f'  {"地理覆盖率:":16} {r.get("geo_cov_ratio",r["cov_ratio"])*100:.2f}%')
    print(f'  {"平均满意度:":16} {r["avg_s"]:.4f}')
    print(f'  {"综合得分:":16} {r["score"]:.4f}')
    print(f'  {"总年利润:":16} {r.get("total_profit", 0)/10000:.2f} 万元')

    print(f'\n  {"="*62}')
    print(f'  {"  各服务站配置与经营明细":^60}')
    print(f'  {"="*62}')

    for si in station_ids:
        assigned_j = [j for j in range(n) if assignment.get(j) == si]
        covered_names = [communities[j] for j in assigned_j]
        sz = plan[si]
        s2_val = r['S2'].get(si, 0)
        profit_wan = r.get('station_profit', {}).get(si, 0) / 10000
        rev_wan = r.get('station_rev', {}).get(si, 0) / 10000
        cost_wan = r.get('station_cost', {}).get(si, 0) / 10000

        print(f'\n  >> 站点 {communities[si]}（{SIZE_LABEL[sz]}）')
        raw_load = sum(daily_demand[j] for j in range(n) if assignment.get(j) == si)
        eff_load = sum(daily_demand[j] * sat_dict.get(j, 1.0) for j in range(n) if assignment.get(j) == si)
        eff_util = eff_load / CAPACITY[sz]
        status = '高负载' if eff_util > 0.95 else ('中负载' if eff_util > 0.75 else '正常' if eff_util > 0.6 else '轻载')
        print(f'    日容量: {CAPACITY[sz]} 人次')
        print(f'    原始负载: {raw_load:.0f} 人次  |  有效负载: {eff_load:.0f}/{CAPACITY[sz]} ({eff_util*100:.1f}% · {status})')
        print(f'    S2(运营压力): {s2_val:.4f}')
        print(f'    覆盖小区: {", ".join(covered_names) if assigned_j else "无"}')
        print(f'    年收入: {rev_wan:>8.2f}万元  |  年成本: {cost_wan:>8.2f}万元  |  年利润: {profit_wan:>8.2f}万元')

        if assigned_j:
            print(f'    各小区满意度:')
            for j in assigned_j:
                print(f'      - {communities[j]}: 满意度={sat_dict.get(j, 0):.4f}  （老年人口 {int(elderly_pop[j])} 人）')

    uncovered_j = [j for j in range(n) if assignment.get(j) is None]
    if uncovered_j:
        print(f'\n  未覆盖小区（共 {len(uncovered_j)} 个）:')
        for j in uncovered_j:
            print(f'    X {communities[j]}: 老年人口 {int(elderly_pop[j])} 人')

    # 全α帕累托对比
    print(f'\n  {"="*74}')
    print(f'  {"帕累托前沿对比（所有α方案）":^72}')
    print(f'  {"="*74}')
    print(f'  {"α":>5} | {"站数":>4} | {"成本":>6} | {"服务覆盖":>8} | {"地理覆盖":>8} | {"满意度":>8} | {"得分":>8} | {"利润(万)":>10}')
    print(f'  {"-"*84}')
    for a in alphas:
        rr = pareto.get(a)
        if rr is None or rr['plan'] is None:
            continue
        cnt = sum(1 for x in rr['plan'] if x > 0)
        cst = rr.get('cost', 0)
        cov = rr['cov_ratio'] * 100
        geo = rr.get('geo_cov_ratio', rr['cov_ratio']) * 100
        sat = rr['avg_s']
        sc = rr['score']
        prof = rr.get('total_profit', 0) / 10000
        sel = '<--' if a == alpha_opt else ''
        print(f'  {a:>5.2f} | {cnt:>4} | {cst:>6} | {cov:>7.2f}% | {geo:>7.2f}% | {sat:>8.4f} | {sc:>8.4f} | {prof:>10.2f}  {sel}')


# ==============================
# 主流程
# ==============================

def main():
    print('=' * 60)
    print('问题2: 服务站选址与规模优化（完整版）')
    print('=' * 60)

    # ---- 0. 数据加载 ----
    print('\n[0/4] 数据加载...')
    df_pop, p1, p2 = read_data()
    demand_df = read_demand_rates()
    revenue_df = read_revenue()
    caps_dict = read_consumption_caps()
    station_df = read_station_data()
    dist_df = read_distance()

    # 确保全局 BUILD_COST 和 CAPACITY 与 附件3 保持一致（BUILD_COST 单位: 万元）
    global BUILD_COST, CAPACITY, N_CORES
    try:
        bc = [0]
        cap = [0]
        for _, row in station_df.iterrows():
            # 读取时可能存在 NaN 或字符串，做安全转换
            try:
                bc.append(float(row['建设成本_万元']))
            except Exception:
                bc.append(0.0)
            try:
                cap.append(int(row['日容量']))
            except Exception:
                try:
                    cap.append(int(float(row.iloc[3])))
                except Exception:
                    cap.append(0)
        BUILD_COST = bc
        CAPACITY = cap
    except Exception:
        print('  [警告] 从附件3读取建设成本/容量失败，使用代码中默认常量')

    # 自动调整并行进程数为可用CPU核数上限
    try:
        N_CORES = min(N_CORES, cpu_count())
    except Exception:
        pass

    communities = df_pop['小区'].tolist()
    n = len(communities)

    pop_results = predict_population(df_pop, p1, p2, years=5)
    elderly_pop = []
    for name in communities:
        _, c, s, d, _ = pop_results[name][5]
        elderly_pop.append(c + s + d)

    detail, summary, _ = compute_actual_demand(
        pop_results, demand_df, revenue_df, df_pop, caps_dict,
        year=5)

    daily_demand = []
    for name in communities:
        total_m = sum(summary[name][svc] for svc in SERVICES)
        daily_demand.append(total_m / DAYS_MONTH)

    cover_mat, dist_values = build_cover_matrix(
        dist_df, communities, radius=RADIUS)
    total_elderly_pop = sum(elderly_pop)

    print(f'  小区: {communities}')
    print(f'  总老年人口: {total_elderly_pop:.0f}')
    print(f'  α: {len(ALPHAS)}个 (步长自适应)')

    print('\n[Phase 0a] 聚类分析...')
    cluster_labels, cluster_info = perform_clustering(
        communities, dist_values, elderly_pop, daily_demand)
    n_clusters = max(cluster_labels) + 1
    print(f'  聚类群组数: {n_clusters}')
    for cid in range(n_clusters):
        info = cluster_info[cid]
        print(f'  群组{cid+1}: {info["communities"]}, '
              f'人口={info["total_pop"]:.0f}, '
              f'日需求={info["total_demand"]:.1f}')

    # ---- Phase 0: 图分析 ----
    print('\n[Phase 0] 图分析...')
    for i in range(n):
        reachable = [communities[j] for j in range(n)
                     if cover_mat[i][j] and i != j]
        print(f'  {communities[i]}: 可达 {reachable}')

    for j in range(n):
        east = [i for i in range(n) if cover_mat[i][j]
                and communities[i] in {'C', 'E', 'F', 'G', 'I'}]
        west = [i for i in range(n) if cover_mat[i][j]
                and communities[i] in {'A', 'B', 'D', 'H', 'J'}]
        if not east:
            print(f'  ** {communities[j]} 只能被西组站覆盖')
        elif not west:
            print(f'  ** {communities[j]} 只能被东组站覆盖')

    # 画图（先设中文字体避免CJK警告泛滥拖慢速度）
    _setup_chinese_font()
    try:
        plot_connectivity_graph(
            communities, dist_values, elderly_pop,
            os.path.join(BASE_DIR, '问题2_1_v3_可达图.png'))
        plot_dendrogram(
            communities, dist_values,
            os.path.join(BASE_DIR, '问题2_1_v3_聚类树状图.png'))
        plot_mds_coverage(
            communities, dist_values, elderly_pop,
            os.path.join(BASE_DIR, '问题2_1_v3_MDS覆盖圈.png'))
    except Exception as e:
        print(f'  绘图异常: {e}')

    # ---- Phase 1: 枚举 ----
    print(f'\n[Phase 1] 枚举 (k=2..6)...')
    # 先算原始总数（无聚类过滤）
    raw_plans = enumerate_plans(n, min_k=2, max_k=6)
    n_plans_total = len(raw_plans)
    all_plans = enumerate_plans(n, min_k=2, max_k=6,
                                cluster_labels=cluster_labels,
                                cluster_info=cluster_info)
    filter_tag = '(聚类过滤)' if USE_CLUSTER_FILTER else ''
    print(f'  枚举方案数: {len(all_plans)} {filter_tag}')
    print(f'  原始方案数（无过滤）: {n_plans_total}')

    # ---- Phase 2a: 方案排序（按容量降序） ----
    print('\n[Phase 2a] 方案排序（按总容量降序）...')
    plan_data = sorted(all_plans, key=lambda p: sum(CAPACITY[sz] for sz in p if sz > 0),
                       reverse=True)
    if not plan_data:
        print('  错误: 无有效方案, 退出')
        sys.exit(1)

    # ---- Phase 2b: 多策略种子+爬山 并行求解 ----
    est_s = len(plan_data) * 0.5 / N_CORES
    print(f'\n[Phase 2b] 多策略种子+α爬山 并行 ({N_CORES}进程, 预计{est_s:.0f}s)...')
    worker_args = [
        (plan, ALPHAS, daily_demand, elderly_pop, cover_mat,
         dist_values, n)
        for plan in plan_data
    ]

    pareto = {}
    for a in ALPHAS:
        pareto[a] = {
            'plan': None, 'score': -1.0, 'cov_ratio': 0.0, 'geo_cov_ratio': 0.0,
            'avg_s': 0.0, 'assignment': {}, 'sat_dict': {}, 'S2': {},
        }

    t0 = time.time()
    solved = 0

    with Pool(N_CORES) as pool:
        for plan_results in pool.imap_unordered(solve_plan,
                                                 worker_args):
            if plan_results is None:
                continue
            solved += 1
            for alpha, result in plan_results.items():
                if result['score'] > pareto[alpha]['score']:
                    pareto[alpha] = {
                        'plan': result['plan'],
                        'score': result['score'],
                        'cov_ratio': result['cov_ratio'],
                        'geo_cov_ratio': result['geo_cov_ratio'],
                        'avg_s': result['avg_s'],
                        'assignment': result['assignment'],
                        'sat_dict': result['sat_dict'],
                        'S2': result['S2'],
                    }
            if solved % 500 == 0:
                elapsed = time.time() - t0
                rate = elapsed / solved
                remaining = rate * (len(worker_args) - solved)
                print(f'    已解 {solved}/{len(worker_args)}  '
                      f'耗时 {elapsed:.0f}s  '
                      f'预计剩余 {remaining:.0f}s')

    t1 = time.time()
    print(f'  多策略种子+爬山完成: {solved}方案, 总耗时 {t1-t0:.1f}s')
    print(f'  方案平均耗时: {(t1-t0)/max(solved,1)*1000:.0f}ms')

    # ---- Phase 3: 利润 ----
    print('\n[Phase 3] 利润计算...')
    rev_dict, cost_dict = get_price_data(revenue_df)
    station_op_cost = get_station_op_cost(station_df)

    n_profit_plans = sum(1 for a in ALPHAS if pareto[a]['plan'] is not None)
    profit_count = 0
    for a in ALPHAS:
        r = pareto[a]
        if r['plan'] is None:
            continue
        plan = r['plan']
        assignment = r['assignment']
        profit, rev, cost, profit_rate = compute_station_profit(
            plan, assignment, communities, summary,
            rev_dict, cost_dict, station_op_cost,
            sat_dict=r.get('sat_dict'))

        r['station_profit'] = profit
        r['station_rev'] = rev
        r['station_cost'] = cost
        r['station_profit_rate'] = profit_rate
        r['total_profit'] = sum(profit.values())
        r['cost'] = sum(BUILD_COST[x] for x in plan if x > 0)
        r['n_stations'] = sum(1 for x in plan if x > 0)
        profit_count += 1
        if profit_count % 2000 == 0:
            print(f'    利润计算: {profit_count}/{n_profit_plans}')
        # 检查利润率是否满足题目中第3问的提示（<=8%）
        # 当前不在选址阶段强制约束，仅记录并在输出中提示
        for si, pr in profit_rate.items():
            if pr > 8.0:
                # 在调试/输出阶段记录警告
                r.setdefault('profit_rate_warnings', []).append((si, pr))

    # ---- Phase 4: 输出 ----
    print('\n[Phase 4] 输出...')

    print('\n' + '=' * 110)
    print('帕累托前沿')
    print('=' * 110)
    h = (f'{"α":>5} | {"站数":>4} | {"成本":>6} | '
         f'{"服务覆盖":>8} | {"地理覆盖":>8} | {"满意度*":>8} | {"得分":>8} | '
         f'{"利润(万)":>10} | 站点配置')
    print(h)
    print('-' * len(h))
    size_label = ['', '小', '中', '大']
    for a in ALPHAS:
        r = pareto[a]
        if r['plan'] is None:
            print(f'{a:>5.2f} | --- 无可行解 ---')
            continue
        plan = r['plan']
        locs = [f'{communities[i]}({size_label[plan[i]]})'
                for i in range(n) if plan[i] > 0]
        print(f'{a:>5.2f} | {r["n_stations"]:>4} | {r["cost"]:>6} | '
              f'{r["cov_ratio"]*100:>7.2f}% | {r.get("geo_cov_ratio",r["cov_ratio"])*100:>7.2f}% | {r["avg_s"]:>8.4f} | '
              f'{r["score"]:>8.4f} | '
              f'{r["total_profit"]/10000:>9.2f} | '
              f'{", ".join(locs)}')

    alpha_knee = select_knee_alpha(pareto, ALPHAS)
    export_pareto_excel_v2(
        pareto, ALPHAS, communities, summary, elderly_pop,
        rev_dict, cost_dict, station_op_cost, daily_demand,
        os.path.join(BASE_DIR, '问题2_1_v3_帕累托前沿.xlsx'),
        alpha_knee=alpha_knee)

    plot_pareto_scatter(
        pareto, ALPHAS, communities,
        os.path.join(BASE_DIR, '问题2_1_v3_α满意度散点图.png'))

    print_problem2_3_analysis(pareto, ALPHAS, communities, elderly_pop, daily_demand)

    print(f'\n完成! 总耗时 {time.time()-t0:.1f}s')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', action='store_true',
                        help='仅枚举+种子测试')
    args = parser.parse_args()

    if args.benchmark:
        t0 = time.time()
        test_comms = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        plans = enumerate_plans(len(test_comms), min_k=2, max_k=6)
        print(f'枚举 {len(plans)} 方案, 耗时 {time.time()-t0:.3f}s')
        sys.exit(0)

    main()
