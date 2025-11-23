import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

RESULT_FILE = PROJECT_ROOT / "data" / "results" / "matched_kt3_hint.csv"

# (可选) 检查文件是否存在
if not RESULT_FILE.exists():
    print(f"警告: 找不到文件 {RESULT_FILE}")

def analyze():
    print(f"正在读取 {RESULT_FILE} ...")
    try:
        df = pd.read_csv(RESULT_FILE)
    except FileNotFoundError:
        print("找不到文件，请先运行 matching_kt3.py 生成结果。")
        return

    if len(df) == 0:
        print("结果文件为空。")
        return

    print(f"成功加载 {len(df)} 对匹配样本")
    print("-" * 40)

    # === 1. 总体效应 (ATE) ===
    mean_treated = df['outcome_t'].mean()
    mean_control = df['outcome_c'].mean()
    
    diff = mean_treated - mean_control
    lift = (diff / mean_control * 100) if mean_control > 0 else 0.0

    print("【总体效果分析 (ATC: 没看提示的人如果看了会怎样)】")
    print(f"  - Treated (看提示) 平均正确率: {mean_treated:.4f} ({mean_treated*100:.2f}%)")
    print(f"  - Control (未看提示) 平均正确率: {mean_control:.4f} ({mean_control*100:.2f}%)")
    print(f"  - 绝对差异 (Impact): {diff:+.4f}")
    print(f"  - 相对提升 (Lift):   {lift:+.2f}%")
    
    if diff > 0:
        print(f"结论: 提示具有 正向 效果。")
    else:
        print(f"结论: 提示效果不显著或为负。")

    print("-" * 40)

    # === 2. 质量控制 ===
    avg_dist = df['dist'].mean()
    print("【匹配质量检查】")
    print(f"  - 平均匹配距离 (L2 distance): {avg_dist:.6e}")
    if avg_dist < 1e-4:
        print("匹配极其精准")
    else:
        print("匹配误差较大")

    print("-" * 40)

    # === 3. 分组异质性分析 ===
    if df['ps_t'].nunique() > 10:
        print("【分组分析：谁最受益？】")
        try:
            # 这里的 PS 是倾向性得分。
            # Low = 预测不爱看提示的人 (通常是学霸或自信者)
            # High = 预测爱看提示的人 (通常是学困生或依赖者)
            df['ps_group'] = pd.qcut(df['ps_t'], 3, labels=['Low (不爱看)', 'Mid (一般)', 'High (爱看)'])
            
            grouped = df.groupby('ps_group', observed=False).agg(
                Treated_Acc=('outcome_t', 'mean'),
                Control_Acc=('outcome_c', 'mean'),
                Count=('outcome_t', 'count')
            )
            
            grouped['Lift_Points'] = (grouped['Treated_Acc'] - grouped['Control_Acc']) * 100
            
            print(grouped.round(4))
        except ValueError:
            print("无法分箱。")
    else:
        print("PS 分数过于集中，跳过分组分析。")

if __name__ == '__main__':
    analyze()