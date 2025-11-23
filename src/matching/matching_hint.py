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

# ================== 路径配置 ==================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
IN_DIR = PROJECT_ROOT / "data" / "raw"
OUT_FILE = PROJECT_ROOT / "data" / "results" / "matched_kt3_hint.csv"
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
# ==========================================================
LIMIT_FILES = 3000  # 如果想调试只处理部分文件，可以设置这个值为整数
HINT_WINDOW_MS = 10 * 60 * 1000 # 窗口期
N_NEIGHBORS = 4      # 反向匹配 1:4 (Control去找4个Treated)
PS_CALIPER = 0.05    # 卡尺
# ==============================

def load_question_map(data_dir):
    q_map = {}
    q_path = os.path.join(data_dir, 'questions.csv')
    if not os.path.exists(q_path):
        q_path = os.path.join(data_dir, 'contents', 'questions.csv')
    if not os.path.exists(q_path):
        print("错误: 找不到 questions.csv")
        return {}
    print(f"正在加载答案表: {q_path} ...")
    with open(q_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            q_map[row['question_id']] = row['correct_answer']
    return q_map

def iter_kt3_features(in_dir, q_map):
    archive_path = os.path.join(in_dir, "kt3.tar.gz")
    if not os.path.exists(archive_path):
        print(f"找不到 {archive_path}")
        return

    print(f"正在处理 KT3 数据流 (窗口期={HINT_WINDOW_MS}ms)...")
    with zipfile.ZipFile(archive_path, 'r') as zf:
        files = [f for f in zf.namelist() if f.endswith('.csv') and not f.endswith('/')]
        if LIMIT_FILES: files = files[:LIMIT_FILES]
        
        for member_name in tqdm(files, desc="Parsing Users"):
            with zf.open(member_name) as f:
                with io.TextIOWrapper(f, encoding='utf-8', errors='replace') as tf:
                    reader = csv.DictReader(tf)
                    history_correct = 0
                    history_total = 0
                    last_action_ts = 0
                    learning_timestamps = []
                    
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
                            
                            # 处理答题
                            if action == 'respond' and item.startswith('q'):
                                elapsed = ts - last_action_ts if last_action_ts > 0 else 0
                                if elapsed < 0: elapsed = 0
                                if elapsed > 300_000: elapsed = 300_000 

                                # 判断 Treatment
                                hint_used = 0
                                for l_ts in reversed(learning_timestamps):
                                    if ts - l_ts > HINT_WINDOW_MS: break
                                    if ts - l_ts >= 0:
                                        hint_used = 1
                                        break
                                
                                # 判断 Outcome
                                correct_ans = q_map.get(item)
                                user_ans = row.get('user_answer')
                                is_correct = 1 if (correct_ans and user_ans == correct_ans) else 0
                                
                                acc_rate = history_correct / history_total if history_total > 0 else 0.0
                                
                                yield {
                                    'acc_rate': acc_rate,
                                    'n_count': history_total,
                                    'ms_response': elapsed,
                                    'hint_used': hint_used,
                                    'outcome': is_correct
                                }
                                
                                history_total += 1
                                history_correct += is_correct
                                last_action_ts = ts
                                if len(learning_timestamps) > 20:
                                    learning_timestamps = learning_timestamps[-10:]
                            else:
                                last_action_ts = ts
                        except ValueError: continue

def main():
    q_map = load_question_map(IN_DIR)
    if not q_map: return

    data_rows = []
    print("1/4 扫描数据并提取特征...")
    for row in iter_kt3_features(IN_DIR, q_map):
        data_rows.append([
            row['acc_rate'],
            np.log1p(row['n_count']),
            np.log1p(row['ms_response']),
            row['hint_used'],
            row['outcome']
        ])
    
    if not data_rows:
        print("没有提取到数据")
        return

    data = np.array(data_rows)
    X = data[:, :3] 
    W = data[:, 3]  
    Y = data[:, 4]  
    
    n_treated = int(sum(W))
    n_control = len(W) - n_treated
    print(f"数据概览: 总数 {len(data)}")
    print(f"   Treated (看提示): {n_treated} (占比 {n_treated/len(data)*100:.1f}%)")
    print(f"   Control (未看):   {n_control} (占比 {n_control/len(data)*100:.1f}%)")
    
    if n_control == 0:
        print("Control 样本为 0，无法进行反向匹配。")
        return

    # 2. 训练 PS 模型
    print("2/4 训练倾向性得分模型 (LightGBM)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    clf = LGBMClassifier(n_estimators=100, verbose=-1)
    clf.fit(X_scaled, W)
    ps_scores = clf.predict_proba(X_scaled)[:, 1]

    # === 诊断代码 ===
    print("\n[诊断] PS 分数分布检查:")
    unique_scores = np.unique(ps_scores)
    print(f"   唯一分数数量: {len(unique_scores)} / {len(ps_scores)}")
    
    # 3. 执行反向匹配 (Control -> Treated)
    print(f"\n3/4 执行反向匹配 (Control -> Treated, 1:{N_NEIGHBORS})...")
    print("   策略: 为每个稀缺的 '未看提示者' 寻找 4 个相似的 '看提示者'")
    
    treated_indices = np.where(W == 1)[0]
    control_indices = np.where(W == 0)[0]
    
    # 建立索引库 (Database = Treated, 因为他们人多)
    treated_ps_vec = ps_scores[treated_indices].reshape(-1, 1).astype('float32')
    index = faiss.IndexFlatL2(1)
    index.add(treated_ps_vec)

    control_ps_vec = ps_scores[control_indices].reshape(-1, 1).astype('float32')
    D, I = index.search(control_ps_vec, N_NEIGHBORS)

    print(f"4/4 保存结果至 {OUT_FILE}...")
    with open(OUT_FILE, 'w') as f:
        f.write("treated_idx,control_idx,ps_t,ps_c,dist,rank,outcome_t,outcome_c\n")
        
        match_count = 0
        for i, c_real_idx in enumerate(control_indices):
            ps_c = control_ps_vec[i][0]
            outcome_c = Y[c_real_idx]
            
            for k in range(N_NEIGHBORS):
                dist = D[i][k]
                t_idx_in_subset = I[i][k] # 找到的 Treated 在子集中的下标
                
                if t_idx_in_subset == -1: continue
                if dist > PS_CALIPER ** 2: continue
                
                # 还原 Treated 的真实 ID
                t_real_idx = treated_indices[t_idx_in_subset]
                ps_t = treated_ps_vec[t_idx_in_subset][0]
                outcome_t = Y[t_real_idx]
                
                # 写入 (保持 Treated在前, Control在后，方便分析)
                f.write(f"{t_real_idx},{c_real_idx},{ps_t:.8f},{ps_c:.8f},{dist:.5e},{k+1},{outcome_t},{outcome_c}\n")
                match_count += 1

    print(f"反向匹配完成! 共生成 {match_count} 对匹配关系。")

if __name__ == '__main__':
    main()