import pandas as pd
from pathlib import Path

# 路径配置
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "results" / "matched_difficulty.csv"

def main():
    if not INPUT_FILE.exists():
        print("文件不存在")
        return

    df = pd.read_csv(INPUT_FILE)
    
    # 筛选出 Hard 组 (根据你之前设定的阈值，通常 diff <= 0.4)
    hard_df = df[df['diff_t'] <= 0.4].copy()
    
    print(f"Hard 组总行数: {len(hard_df)}")
    
    if len(hard_df) == 0:
        print("没有 Hard 组数据？请检查阈值。")
        return

    # 1. 检查 Outcome 是否完全一样
    same_outcome_count = (hard_df['outcome_t'] == hard_df['outcome_c']).sum()
    print(f"Outcome 完全相同的行数: {same_outcome_count} / {len(hard_df)}")
    
    # 2. 检查是否是“自匹配” (T_idx == C_idx)
    # 注意：如果你的 CSV 里没有保存原始索引列 t_idx, c_idx，这一步可能没法做，
    # 但我们可以对比 ps_score 和 diff 是否完全一致
    
    # 检查 PS Score 是否完全一致 (浮点数可能有微小差异，用 isclose)
    import numpy as np
    ps_same = np.isclose(hard_df['ps_t'], hard_df['ps_c']).sum()
    print(f"PS Score 完全一致的行数: {ps_same} / {len(hard_df)}")
    
    # 3. 打印前 10 行详细数据看看
    print("\nHard 组前 10 行样本预览:")
    print(hard_df[['t_idx', 'c_idx', 'outcome_t', 'outcome_c', 'diff_t', 'diff_c']].head(10))

    # 4. 致命检查：是否所有 Hard 数据的 Outcome 都是 0 (或者都是 1)？
    print("\nHard 组 Outcome 分布:")
    print(hard_df[['outcome_t', 'outcome_c']].value_counts())

if __name__ == "__main__":
    main()