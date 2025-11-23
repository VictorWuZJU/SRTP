import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "results" / "matched_kt3_diff.csv"
OUT_DIR = PROJECT_ROOT / "figures" / "plots_diff"

# (推荐) 自动创建输出文件夹，防止因为文件夹不存在而报错
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    
    print("正在读取数据...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except:
        print("没找到文件，请先运行 matching_kt3_diff.py")
        return

    # === 定义难度分组 ===
    # diff_t 是题目的平均正确率。
    # 值越高 = 越容易 (Easy)
    # 值越低 = 越难 (Hard)
    # 我们按 Treated 组做题目的难度来分 (因为我们关心的是看了提示的那道题难不难)
    
    # 逻辑分箱 (可以根据数据分布调整阈值，这里用经典的 0.4 / 0.7)
    def categorize_diff(x):
        if x > 0.75: return 'Easy (High Acc)'
        elif x < 0.45: return 'Hard (Low Acc)'
        else: return 'Medium'

    df['Difficulty_Group'] = df['diff_t'].apply(categorize_diff)
    
    # 排序：Hard -> Medium -> Easy
    order = ['Hard (Low Acc)', 'Medium', 'Easy (High Acc)']

    # === 绘图: 正确率对比 ===
    print("绘制难度分组正确率图...")
    df_melted = pd.melt(df, id_vars=['Difficulty_Group'], value_vars=['outcome_t', 'outcome_c'], 
                        var_name='Condition', value_name='Accuracy')
    df_melted['Condition'] = df_melted['Condition'].map({'outcome_t': 'With Hint', 'outcome_c': 'No Hint'})

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    sns.barplot(data=df_melted, x='Difficulty_Group', y='Accuracy', hue='Condition', 
                order=order, palette={'With Hint': '#ff6b6b', 'No Hint': '#4ecdc4'}, capsize=.1)
    
    plt.title('Hint Effect by Question Difficulty')
    plt.ylim(0, 1.0)


    win_path = str(OUT_DIR).replace('/', '\\') 
 

    # === 计算 Lift 并打印 ===
    print("\n详细数据:")
    grouped = df.groupby('Difficulty_Group', observed=False).agg(
        Treated_Acc=('outcome_t', 'mean'),
        Control_Acc=('outcome_c', 'mean'),
        Count=('outcome_t', 'count')
    )
    grouped['Lift'] = (grouped['Treated_Acc'] - grouped['Control_Acc']) * 100
    # 按自定义顺序打印
    print(grouped.reindex(order))

if __name__ == '__main__':
    main()