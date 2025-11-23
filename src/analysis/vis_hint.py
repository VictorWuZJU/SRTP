import matplotlib
# 必须在导入 pyplot 之前设置后端，适配 WSL 无 GUI 环境
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "results" / "matched_kt3_hint.csv"
OUT_DIR = PROJECT_ROOT / "figures"

OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
        print(f"创建目录: {OUT_DIR}")

    print(f"正在读取数据: {INPUT_FILE} ...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except Exception as e:
        print(f"读取失败: {e}")
        return

    # 设置绘图风格 (学术风)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    # ==========================================
    # 图表 1: 倾向性得分分布图 (Balance Check)
    # 作用: 证明匹配质量。Treated 和 Control 的分布应该高度重合。
    # ==========================================
    print("正在绘制图表 1: PS分布对比 (Balance Check)...")
    plt.figure(figsize=(10, 6))
    
    # 绘制密度图
    sns.kdeplot(df['ps_t'], color='red', fill=True, label='Treated (Hint Used)', alpha=0.4)
    sns.kdeplot(df['ps_c'], color='blue', fill=True, label='Control (No Hint)', alpha=0.4)
    
    plt.title('Propensity Score Distribution After Matching', fontsize=14)
    plt.xlabel('Propensity Score (Probability of Using Hint)')
    plt.ylabel('Density')
    plt.legend()
    
    save_path = os.path.join(OUT_DIR, "1_ps_distribution.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: {save_path}")

    # ==========================================
    # 数据预处理: 准备分组分析数据
    # ==========================================
    if df['ps_t'].nunique() > 10:
        # 分箱：Low (学霸), Mid (潜能), High (学困)
        df['Group'] = pd.qcut(df['ps_t'], 3, labels=['Low (High Ability)', 'Mid (Potential)', 'High (Low Ability)'])
        
        # 转换为长格式 (Long Format) 方便 Seaborn 绘图
        # 我们需要把 outcome_t 和 outcome_c 合并到一列
        df_melted = pd.melt(
            df, 
            id_vars=['Group'], 
            value_vars=['outcome_t', 'outcome_c'],
            var_name='Condition', 
            value_name='Accuracy'
        )
        
        # 重命名标签让图例更好看
        df_melted['Condition'] = df_melted['Condition'].map({
            'outcome_t': 'With Hint', 
            'outcome_c': 'No Hint'
        })

        # ==========================================
        # 图表 2: 分组正确率对比 (The Discovery)
        # 作用: 展示不同人群在看与不看提示下的表现差异
        # ==========================================
        print("正在绘制图表 2: 分组正确率对比...")
        plt.figure(figsize=(10, 6))
        
        # Barplot 会自动计算置信区间 (那个黑色的竖线 error bar)
        bar_plot = sns.barplot(
            data=df_melted, 
            x='Group', 
            y='Accuracy', 
            hue='Condition',
            palette={'With Hint': '#ff6b6b', 'No Hint': '#4ecdc4'}, # 红绿对比
            capsize=.1
        )
        
        plt.title('Impact of Hints by Student Group (Heterogeneous Effects)', fontsize=14)
        plt.ylim(0.3, 0.7) # 设置Y轴范围让差异更明显
        plt.ylabel('Answer Correctness')
        plt.xlabel('Student Group (Based on Propensity Score)')
        plt.legend(title='Condition', loc='upper left')
        
        save_path = os.path.join(OUT_DIR, "2_group_accuracy.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已保存: {save_path}")

        # ==========================================
        # 图表 3: 提升幅度 (Lift) 直方图
        # 作用: 直观展示 "谁受益，谁受损"
        # ==========================================
        print("正在绘制图表 3: 净提升幅度 (Net Lift)...")
        
        # 手动计算均值差异
        group_stats = df.groupby('Group', observed=False).agg({
            'outcome_t': 'mean', 
            'outcome_c': 'mean'
        })
        group_stats['Lift'] = (group_stats['outcome_t'] - group_stats['outcome_c']) * 100
        
        plt.figure(figsize=(8, 5))
        
        # 定义颜色：正值用绿色，负值用红色
        colors = ['#ff6b6b' if x < 0 else '#1dd1a1' for x in group_stats['Lift']]
        
        ax = sns.barplot(
            x=group_stats.index, 
            y=group_stats['Lift'],
            palette=colors
        )
        
        # 在柱子上方/下方标数值
        for i, v in enumerate(group_stats['Lift']):
            offset = 0.5 if v >= 0 else -1.0
            ax.text(i, v + offset, f"{v:+.2f}%", ha='center', fontweight='bold')

        plt.axhline(0, color='black', linewidth=1) # 添加 0 轴线
        plt.title('Net Improvement by Hint (Lift %)', fontsize=14)
        plt.ylabel('Accuracy Lift (Percentage Points)')
        plt.xlabel('Student Group')
        
        save_path = os.path.join(OUT_DIR, "3_net_lift.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已保存: {save_path}")

    else:
        print("PS 分数太少，跳过分组绘图。")

    # ------------------------------
    print("-" * 30)
    print(f"分析完成！图表已保存至: {OUT_DIR}")
    
    # 生成方便 Windows 访问的路径 (假设是标准的 Ubuntu WSL 发行版)
    # 将 Linux 路径 /home/username/... 转换为 Windows 路径 \\wsl$\Ubuntu\home\username\...
    try:
        # 获取绝对路径字符串
        abs_path = str(OUT_DIR.resolve())
        # 1. 先在外面把反斜杠替换好
        win_path_suffix = abs_path.replace('/', '\\')
        # 2. 再拼接到 f-string 里
        win_path = f"\\\\wsl$\\Ubuntu{win_path_suffix}"
        print(f"Windows 访问路径 (复制到资源管理器): {win_path}")
    except Exception:
        pass
if __name__ == "__main__":
    main()
