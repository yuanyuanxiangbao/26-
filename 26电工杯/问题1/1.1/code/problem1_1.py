"""
问题1.1：未来五年老人数量预测（年级递推模型）
时序(每年)：年初新增(全归自理) → 全年逐日死亡+转移(自理→半失能, 半失能→失能) → 年末记录
新增人员从加入当年起即面临死亡率
"""
import os
from dataclasses import dataclass
import pandas as pd
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
PROJECT_ROOT = os.path.join(BASE_DIR, '..', '..', '..')
DATA_FILE_DIR = os.path.join(PROJECT_ROOT, '题目')
@dataclass
class ModelConfig:
    days_per_year: int = 365
    mortality: float = 0.05
    new_rate: float = 0.07
def read_data():
    path = os.path.join(DATA_FILE_DIR, '附件1：小区基础数据.xlsx')
    df_pop = pd.read_excel(path, sheet_name=0, header=1)
    df_pop.columns = ['小区', '总人口', '60+老人数', '自理', '半失能', '失能', '人均月收入']
    df_prob = pd.read_excel(path, sheet_name=1, header=1)
    df_prob.columns = ['场景', '概率']
    
    # 增强转移概率读取：使用更健壮的方式（精确列名+验证）
    try:
        p_self_to_semi = float(df_prob.loc[df_prob['场景'].str.contains('自理.*半失能', na=False), '概率'].values[0])
    except Exception as e:
        raise ValueError(f'转移概率读取失败（自理→半失能）：{e}。请检查 附件1 Sheet2 的场景列名')
    
    try:
        p_semi_to_dis  = float(df_prob.loc[df_prob['场景'].str.contains('半失能.*失能', na=False), '概率'].values[0])
    except Exception as e:
        raise ValueError(f'转移概率读取失败（半失能→失能）：{e}。请检查 附件1 Sheet2 的场景列名')
    
    # 验证转移概率合理性
    if not (0 < p_self_to_semi < 1):
        raise ValueError(f'转移概率自理→半失能={p_self_to_semi} 不在(0,1)范围内')
    if not (0 < p_semi_to_dis < 1):
        raise ValueError(f'转移概率半失能→失能={p_semi_to_dis} 不在(0,1)范围内')
    
    return df_pop, p_self_to_semi, p_semi_to_dis
def predict_population(df_pop, p_self_to_semi, p_semi_to_dis, years=5, 
                       config=None, new_rate_override=None):
    """
    修正递推模型：年级递推而非日级递推
    
    时序（每年）：
      1. 年初：加入新增人员（年初人口 × 7%），新增人全部归自理类
      2. 全年：逐日处理死亡与转移（365日）
         - 死亡：既存人口按5%年率扣除（日化）
         - 转移：自理→半失能，半失能→失能（日化）
      3. 年末：记录各类人口数
    
    关键说明：
    - 新增人员在年初一次性加入，不是每日加入
    - 新增人员从加入当年起即面临死亡率（与既存人口同等处理）
    
    参数：
    - new_rate_override: 可选，覆盖config中的new_rate（用于灵敏度分析）
    """
    if config is None:
        config = ModelConfig()
    
    # 日化参数
    mu = 1 - (1 - config.mortality) ** (1 / config.days_per_year)
    t1 = 1 - (1 - p_self_to_semi) ** (1 / config.days_per_year)
    t2 = 1 - (1 - p_semi_to_dis) ** (1 / config.days_per_year)
    growth_rate = new_rate_override if new_rate_override is not None else config.new_rate
    
    results = {}
    for _, row in df_pop.iterrows():
        name = row['小区']
        c0, s0, d0 = float(row['自理']), float(row['半失能']), float(row['失能'])
        
        records = [(0, c0, s0, d0, c0 + s0 + d0)]
        
        for year in range(1, years + 1):
            # 第1步：年初加入新增人员（基于上年末总人口）
            prev_total = c0 + s0 + d0
            new_people = prev_total * growth_rate  # 年初新增人数
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

def print_results(results):
    for name, records in results.items():
        print(f'\n=== 小区 {name} ===')
        print(f'{"年份":>4} | {"自理":>6} | {"半失能":>6} | {"失能":>6} | {"总计":>6}')
        print('-' * 40)
        for y, c, s, d, t in records:
            print(f'{y:>4} | {c:.4f} | {s:.4f} | {d:.4f} | {t:.4f}')
def verify(results):
    print('\n' + '='*80)
    print('模型验证与增长率分析')
    print('='*80)
    for name, records in results.items():
        t0, t1, t5 = records[0][4], records[1][4], records[5][4]
        growth_y1 = (t1 - t0) / t0 * 100
        cagr_5 = (t5 / t0) ** (1/5) - 1
        cagr_14 = (t5 / t1) ** (1/4) - 1
        print(f'\n小区 {name}:')
        print(f'   第0年(初)总人口: {t0:.4f}')
        print(f'   第1年末总人口: {t1:.4f}  (较第0年 +{growth_y1:.4f}%)')
        print(f'   第5年末总人口: {t5:.4f}')
        print(f'   5年整体CAGR: {cagr_5*100:.4f}%')
        print(f'   第1→5年CAGR: {cagr_14*100:.4f}%')
        
        # 逐年详细输出
        print(f'   逐年详细:')
        for year, c, s, d, total in records:
            print(f'      年{year}: 自理={c:.2f}, 半失能={s:.2f}, 失能={d:.2f}, 合计={total:.2f}')
        
        if name == 'A':
            _, c1, s1, d1, _ = records[1]
            print(f'   A第1年结构: 自理={c1:.4f} | 半失能={s1:.4f} | 失能={d1:.4f}')



def output_excel(results, filepath=None):
    """输出各小区逐年数据到Excel，每个小区一个sheet"""
    if filepath is None:
        filepath = os.path.join(DATA_DIR, '问题1_1_人口预测结果.xlsx')

    if os.path.exists(filepath):
        try:
            os.remove(filepath)
        except PermissionError:
            from datetime import datetime
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = os.path.join(BASE_DIR, f'问题1_1_人口预测结果.xlsx')
            print(f'[警告] 文件被占用，已自动切换保存到: {filepath}')
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        for name, records in results.items():
            df = pd.DataFrame(records, columns=['年份', '自理', '半失能', '失能', '总计'])
            df.to_excel(writer, sheet_name=name, index=False)
    print(f'\nExcel已保存: {filepath}')
    return filepath
if __name__ == '__main__':
    config = ModelConfig()
    df_pop, p1, p2 = read_data()
    
    print('='*80)
    print('问题1.1：未来五年老人数量预测（年级递推模型）')
    print('='*80)
    print(f'\n参数设置:')
    print(f'  年均死亡率: {config.mortality*100:.1f}%')
    print(f'  年均新增率: {config.new_rate*100:.1f}% (年初一次性加入)')
    print(f'  转移概率（自理→半失能）: {p1*100:.2f}%')
    print(f'  转移概率（半失能→失能）: {p2*100:.2f}%')
    print(f'  递推周期: {config.days_per_year}天 × 5年')
    
    print('\n运行年级递推模型...')
    results = predict_population(df_pop, p1, p2, years=5, config=config)
    print_results(results)
    verify(results)
 
    output_excel(results)

