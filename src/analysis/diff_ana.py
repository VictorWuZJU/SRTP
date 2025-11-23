import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# ================== 配置区域 ==================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "results" / "matched_difficulty.csv"
FIG_DIR = PROJECT_ROOT / "figures" / "analysis"

# 确保输出目录存在
FIG_DIR.mkdir(parents=True, exist_ok=True)
# ============================================

def check_balance(df):
    """1. 质量检验：检查匹配后的协变量平衡性"""
    print("\n=== 1. 匹配质量检验 (Balance Check) ===")
    
    diff_gap = (df['diff_t'] - df['diff_c']).mean()
    ps_gap = (df['ps_t'] - df['ps_c']).mean()
    
    print(f"样本对数: {len(df)}")
    print(f"题目难度平均差异 (Diff Gap): {diff_gap:.6f} (应接近 0)")
    print(f"PS分数平均差异 (PS Gap)  : {ps_gap:.6f} (应接近 0)")
    
    if abs(diff_gap) < 0.01 and abs(ps_gap) < 0.01:
        print("检验通过：Treated 和 Control 组高度平衡！")
    else:
        print("警告：组间仍存在细微差异，请留意。")

def analyze_overall_effect(df):
    """2. 总体效应分析 (Overall ATT)"""
    print("\n=== 2. 总体提示效应 (Overall Effect) ===")
    
    acc_t = df['outcome_t'].mean()
    acc_c = df['outcome_c'].mean()
    att = acc_t - acc_c
    
    # 配对 t 检验 (因为是 1:1 匹配数据，样本非独立，必须用 paired t-test)
    t_stat, p_val = stats.ttest_rel(df['outcome_t'], df['outcome_c'])
    
    print(f"Treated (With Hint) 正确率: {acc_t:.4f}")
    print(f"Control (No Hint)   正确率: {acc_c:.4f}")
    print(f"提升 (Lift/ATT)        : {att*100:.2f}%")
    print(f"P-value (Paired t-test)  : {p_val:.4e}")
    
    if p_val < 0.05:
        print("结果显著：提示确实有效！")
    else:
        print("结果不显著：提示可能没啥用。")

def analyze_heterogeneity(df):
    """3. 异质性分析：不同难度下的效果差异"""
    print("\n=== 3. 难度异质性分析 (Heterogeneity by Difficulty) ===")
    
    # 定义分组函数
    def categorize_diff(x):
        # 这里的阈值可以根据 diff_t 的分布直方图微调
        # 假设 0.7 以上是简单，0.4 以下是难
        if x >= 0.7: return 'Easy'
        elif x <= 0.4: return 'Hard'
        else: return 'Medium'
    
    df['Difficulty_Group'] = df['diff_t'].apply(categorize_diff)
    
    # 统计分组数据
    groups = []
    for name, group in df.groupby('Difficulty_Group'):
        mean_t = group['outcome_t'].mean()
        mean_c = group['outcome_c'].mean()
        lift = mean_t - mean_c
        # 分组内的配对 t 检验
        _, p_val = stats.ttest_rel(group['outcome_t'], group['outcome_c'])
        
        groups.append({
            'Group': name,
            'Count': len(group),
            'With Hint': mean_t,
            'No Hint': mean_c,
            'Lift': lift,
            'P-value': p_val
        })
    
    res_df = pd.DataFrame(groups).set_index('Group')
    # 自定义排序
    order = ['Hard', 'Medium', 'Easy']
    res_df = res_df.reindex(order)
    
    print(res_df)
    return res_df

def plot_results(df, res_summary):
    """画图"""
    print(f"\n正在绘图至 {FIG_DIR}...")
    
    # 准备画图数据 (Melt format)
    # 我们只取需要的列
    plot_df = df[['diff_t', 'outcome_t', 'outcome_c']].copy()
    plot_df['Difficulty_Group'] = plot_df['diff_t'].apply(
        lambda x: 'Easy' if x >= 0.7 else ('Hard' if x <= 0.4 else 'Medium')
    )
    
    df_melted = plot_df.melt(
        id_vars=['Difficulty_Group'], 
        value_vars=['outcome_t', 'outcome_c'],
        var_name='Condition', 
        value_name='Accuracy'
    )
    
    df_melted['Condition'] = df_melted['Condition'].map({
        'outcome_t': 'With Hint', 
        'outcome_c': 'No Hint'
    })
    
    # 设置绘图风格
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.figure(figsize=(10, 6))
    
    # 绘制柱状图
    order = ['Hard', 'Medium', 'Easy']
    # palette: 蓝色和橙色对比度高
    ax = sns.barplot(
        data=df_melted, 
        x='Difficulty_Group', 
        y='Accuracy', 
        hue='Condition',
        order=order,
        palette=['#e74c3c', '#2ecc71'], # 红绿对比
        capsize=.1,
        errwidth=1.5
    )
    
    # 在图上标注 Lift
    for i, group_name in enumerate(order):
        lift = res_summary.loc[group_name, 'Lift']
        p_val = res_summary.loc[group_name, 'P-value']
        
        # 简单的显著性标记
        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
        
        # 获取柱子高度
        # 这里为了简单，直接标在图形上方中间
        y_pos = max(res_summary.loc[group_name, 'With Hint'], res_summary.loc[group_name, 'No Hint']) + 0.05
        if y_pos > 1.0: y_pos = 0.95
        
        plt.text(i, y_pos, f"+{lift*100:.1f}%\n({sig})", 
                 ha='center', va='bottom', color='black', fontweight='bold')

    plt.title('Impact of Hints by Question Difficulty (1:1 Matched)', pad=20)
    plt.ylim(0, 1.15) # 留出上方空间写字
    plt.ylabel('Average Accuracy')
    plt.xlabel('Difficulty Level')
    plt.legend(title='Condition', loc='upper left')
    
    save_path = FIG_DIR / "hint_impact_analysis.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图片已保存: {save_path}")

def main():
    if not INPUT_FILE.exists():
        print(f"文件不存在: {INPUT_FILE}")
        return

    # 读取数据
    df = pd.read_csv(INPUT_FILE)
    
    # 执行三步分析
    check_balance(df)
    analyze_overall_effect(df)
    summary = analyze_heterogeneity(df)
    
    # 绘图
    plot_results(df, summary)

if __name__ == "__main__":
    main()