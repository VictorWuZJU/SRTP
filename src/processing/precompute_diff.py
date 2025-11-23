import csv
import zipfile
import io
import os
from tqdm import tqdm
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
IN_DIR = PROJECT_ROOT / "data" / "raw"
OUT_FILE = PROJECT_ROOT / "data" / "processed" / "question_difficulty.csv"
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
LIMIT_FILES = 2000 

def load_question_map(data_dir):
    q_map = {}
    q_path = os.path.join(data_dir, 'questions.csv')
    if not os.path.exists(q_path):
        q_path = os.path.join(data_dir, 'contents', 'questions.csv')
    if not os.path.exists(q_path):
        print("找不到 questions.csv")
        return {}
    with open(q_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            q_map[row['question_id']] = row['correct_answer']
    return q_map

def main():
    q_map = load_question_map(IN_DIR)
    if not q_map: return

    # 统计每道题的 [答对次数, 总次数]
    item_stats = {} 

    archive_path = os.path.join(IN_DIR, "kt3.tar.gz")
    print(f"正在扫描数据以计算题目难度 (采样 {LIMIT_FILES} 个用户)...")
    
    with zipfile.ZipFile(archive_path, 'r') as zf:
        files = [f for f in zf.namelist() if f.endswith('.csv') and not f.endswith('/')]
        if LIMIT_FILES: files = files[:LIMIT_FILES]
        
        for member_name in tqdm(files):
            with zf.open(member_name) as f:
                with io.TextIOWrapper(f, encoding='utf-8', errors='replace') as tf:
                    reader = csv.DictReader(tf)
                    for row in reader:
                        try:
                            if row['action_type'] != 'respond': continue
                            
                            item = row['item_id']
                            if not item.startswith('q'): continue
                            
                            correct_ans = q_map.get(item)
                            if not correct_ans: continue
                            
                            is_correct = 1 if row.get('user_answer') == correct_ans else 0
                            
                            if item not in item_stats:
                                item_stats[item] = [0, 0]
                            
                            item_stats[item][0] += is_correct
                            item_stats[item][1] += 1
                        except: continue

    print(f"正在保存难度表至 {OUT_FILE} ...")
    with open(OUT_FILE, 'w') as f:
        f.write("item_id,avg_correctness\n")
        count = 0
        for item, stats in item_stats.items():
            total = stats[1]
            if total < 5: continue # 忽略极其冷门的题目
            
            avg_acc = stats[0] / total
            f.write(f"{item},{avg_acc:.4f}\n")
            count += 1
            
    print(f"完成！共计算了 {count} 道题目的难度系数。")

if __name__ == '__main__':
    main()