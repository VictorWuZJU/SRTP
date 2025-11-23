#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import zipfile
import io
import numpy as np
import faiss
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from pathlib import Path

# ================== 1.路径配置 (Pathlib 版) ==================
# 1. 自动定位项目根目录 (.../SRTP/)
#    解释: 当前脚本 -> matching -> src -> SRTP
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 2. 定义各级数据目录
DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"

# 3. 具体文件路径
ARCHIVE_PATH = DATA_DIR / "kt3.tar.gz"
QUESTION_FILE = DATA_DIR / "questions.csv"

DIFF_FILE = PROCESSED_DIR / "question_difficulty.csv"
OUT_FILE = RESULTS_DIR / "matched_difficulty.csv"

# (可选) 自动创建结果目录，防止报错
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
# ==========================================================
# ==========================================
# 2. 核心参数配置
# ==========================================
LIMIT_FILES = 1000        # 读取多少个用户文件 (调试时设小，跑全量设为 None)
HINT_WINDOW_MS = 10 * 60 * 1000  # 10分钟内的学习行为算作看提示
SEARCH_CANDIDATES = 50    # FAISS 初搜寻找 50 个 PS 最近邻
PS_CALIPER = 0.05         # PS 分数差异容忍度 (能力差异)
DIFF_CALIPER = 0.1        # 题目难度差异容忍度 (任务差异) <--- 核心改进
# ==========================================

def load_resources():
    """加载题目答案和难度表"""
    print(f"📖 [1/5] 加载资源文件...")
    
    # 1. 加载题目答案
    q_map = {}
    if os.path.exists(QUESTION_FILE):
        with open(QUESTION_FILE, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                q_map[row['question_id']] = row['correct_answer']
        print(f"   -> 已加载 {len(q_map)} 道题目答案")
    else:
        print(f"   -> 错误: 找不到 {QUESTION_FILE}")
        return None, None

    # 2. 加载题目难度
    d_map = {}
    if os.path.exists(DIFF_FILE):
        with open(DIFF_FILE, 'r') as f:
            for row in csv.DictReader(f):
                d_map[row['item_id']] = float(row['avg_correctness'])
        print(f"   -> 已加载 {len(d_map)} 条难度数据")
    else:
        print(f"   -> 警告: 找不到难度表 {DIFF_FILE}，将无法进行难度对齐！")
        # 为了代码不崩，给个空字典，后续会默认 difficulty=0.5
    
    return q_map, d_map

def extract_features(q_map, d_map):
    """从压缩包流式提取特征"""
    print(f"[2/5] 正在从 {os.path.basename(ARCHIVE_PATH)} 提取特征...")
    
    data_rows = []
    
    if not os.path.exists(ARCHIVE_PATH):
        print(f"找不到数据包: {ARCHIVE_PATH}")
        return []

    with zipfile.ZipFile(ARCHIVE_PATH, 'r') as zf:
        files = [f for f in zf.namelist() if f.endswith('.csv') and not f.endswith('/')]
        if LIMIT_FILES: 
            files = files[:LIMIT_FILES]
            print(f"   -> 采样模式: 仅处理前 {LIMIT_FILES} 个用户")
        
        for member_name in tqdm(files, desc="Processing Users"):
            with zf.open(member_name) as f:
                with io.TextIOWrapper(f, encoding='utf-8', errors='replace') as tf:
                    reader = csv.DictReader(tf)
                    
                    # 用户状态变量
                    history_correct = 0
                    history_total = 0
                    last_action_ts = 0
                    learning_timestamps = [] # 记录看 e/l 的时间
                    
                    for row in reader:
                        try:
                            ts = int(row['timestamp'])
                            action = row['action_type']
                            item = row['item_id']
                            
                            # 记录学习行为 (Explanation / Lecture)
                            if action == 'enter' and (item.startswith('e') or item.startswith('l')):
                                learning_timestamps.append(ts)
                                last_action_ts = ts
                                continue
                            
                            # 处理答题行为
                            if action == 'respond' and item.startswith('q'):
                                # 1. 计算响应时间
                                elapsed = ts - last_action_ts if last_action_ts > 0 else 0
                                elapsed = max(0, min(elapsed, 300000)) # 截断异常值
                                
                                # 2. 判断是否看过提示 (Recall Window)
                                hint_used = 0
                                # 倒序检查最近的学习记录
                                for l_ts in reversed(learning_timestamps):
                                    if ts - l_ts > HINT_WINDOW_MS: break # 超时了
                                    if ts - l_ts >= 0:
                                        hint_used = 1
                                        break
                                
                                # 3. 计算 Outcome
                                correct_ans = q_map.get(item)
                                if not correct_ans: continue # 没答案的题跳过
                                is_correct = 1 if row.get('user_answer') == correct_ans else 0
                                
                                # 4. 历史正确率 (Ability Proxy)
                                acc_rate = history_correct / history_total if history_total > 0 else 0.0
                                
                                # 5. 题目难度
                                diff_val = d_map.get(item, 0.5) # 默认中等难度

                                # 收集一行数据
                                data_rows.append([
                                    acc_rate,                   # Feature 0: 历史正确率
                                    np.log1p(history_total),    # Feature 1: 做题经验(log)
                                    np.log1p(elapsed),          # Feature 2: 耗时(log)
                                    diff_val,                   # Feature 3: 题目难度 (作为特征也要放进模型)
                                    hint_used,                  # Treatment
                                    is_correct                  # Outcome
                                ])
                                
                                # 更新状态
                                history_total += 1
                                history_correct += is_correct
                                last_action_ts = ts
                                
                                # 限制内存，只保留最近的学习记录
                                if len(learning_timestamps) > 20:
                                    learning_timestamps = learning_timestamps[-10:]
                            else:
                                last_action_ts = ts
                                
                        except ValueError: continue
                        
    return np.array(data_rows)

def main():
    # 1. 加载资源
    q_map, d_map = load_resources()
    if not q_map: return

    # 2. 提取特征
    data = extract_features(q_map, d_map)
    if len(data) == 0:
        print("没有提取到任何数据，请检查原始数据包。")
        return

    # 数据切分
    # X: [acc, log_count, log_time, difficulty]
    X = data[:, :4] 
    W = data[:, 4]   # Treatment (Hint)
    Y = data[:, 5]   # Outcome
    DIFF_COL = data[:, 3] # 单独拿出来方便后面筛选
    
    print(f"[3/5] 数据统计: 总样本 {len(data)}")
    print(f"   - Treated (Hint): {int(sum(W))}")
    print(f"   - Control (No Hint): {len(W) - int(sum(W))}")

    # 3. 计算 Propensity Score
    print("[4/5] 训练 LightGBM 计算倾向性得分 (PS Score)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    clf = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    clf.fit(X_scaled, W)
    ps_scores = clf.predict_proba(X_scaled)[:, 1]

    # 4. 执行 1:1 双重卡尺匹配
    print(f"[5/5] 执行 1:1 精确匹配 (PS < {PS_CALIPER} & Diff < {DIFF_CALIPER})...")
    
    treated_indices = np.where(W == 1)[0]
    control_indices = np.where(W == 0)[0]
    
    # 建立索引 (用 Treated 做库)
    treated_ps_vec = ps_scores[treated_indices].reshape(-1, 1).astype('float32')
    index = faiss.IndexFlatL2(1) # L2 距离其实就是 1D 的绝对差的平方
    index.add(treated_ps_vec)
    
    control_ps_vec = ps_scores[control_indices].reshape(-1, 1).astype('float32')
    
    # 搜索 K 个最近邻 (K 要大一点，因为我们要从中筛选符合难度要求的)
    D, I = index.search(control_ps_vec, SEARCH_CANDIDATES)
    
    matched_pairs = []
    used_treated_indices = set() # 保证 1:1，不重复使用 Treated 样本
    
    # 遍历每一个 Control 样本寻找它的 Soul Mate
    for i in range(len(control_indices)):
        c_real_idx = control_indices[i]
        c_diff = DIFF_COL[c_real_idx] # Control 做的题的难度
        
        # 遍历它的 K 个 PS 邻居
        best_match = None
        
        for k in range(SEARCH_CANDIDATES):
            dist_sq = D[i][k]
            t_idx_subset = I[i][k]
            
            if t_idx_subset == -1: continue
            
            # 1. 检查 PS 卡尺
            if dist_sq > PS_CALIPER**2: 
                continue # PS 差太远
            
            t_real_idx = treated_indices[t_idx_subset]
            
            # 2. 检查是否已经被用过了 (1:1 约束)
            if t_real_idx in used_treated_indices:
                continue
            
            # 3. 检查难度卡尺 (核心逻辑!)
            t_diff = DIFF_COL[t_real_idx]
            if abs(t_diff - c_diff) > DIFF_CALIPER:
                continue # 题目难度差太远 (苹果 vs 橘子)，跳过
            
            # 如果都满足，这才是我们要的人
            best_match = t_real_idx
            used_treated_indices.add(t_real_idx) # 标记为已用
            break # 找到一个就够了，退出内层循环
        
        if best_match is not None:
            # 记录匹配对
            t_real_idx = best_match
            matched_pairs.append({
                't_idx': t_real_idx,
                'c_idx': c_real_idx,
                'ps_t': ps_scores[t_real_idx],
                'ps_c': ps_scores[c_real_idx],
                'diff_t': DIFF_COL[t_real_idx],
                'diff_c': DIFF_COL[c_real_idx],
                'outcome_t': Y[t_real_idx],
                'outcome_c': Y[c_real_idx]
            })

    # 5. 保存结果
    print(f"保存结果至 {OUT_FILE}...")
    print(f"   -> 原始 Control 数: {len(control_indices)}")
    print(f"   -> 匹配成功对数: {len(matched_pairs)} (丢弃率: {100 - len(matched_pairs)/len(control_indices)*100:.1f}%)")
    
    if matched_pairs:
        keys = matched_pairs[0].keys()
        with open(OUT_FILE, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(matched_pairs)
        print("全部完成！去画图吧！")
    else:
        print("匹配失败，没有找到任何符合条件的样本对。试着放宽 CALIPER？")

if __name__ == '__main__':
    main()