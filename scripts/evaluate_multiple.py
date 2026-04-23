import json
import argparse
import os
from collections import defaultdict

# --- 尝试导入 matplotlib --- 
PLOT_AVAILABLE = False
try:
    import matplotlib.pyplot as plt
    import numpy as np
    # 尝试配置中文字体 (与 evaluate.py 保持一致)
    plt.rcParams['font.sans-serif'] = ['Heiti TC', 'SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    PLOT_AVAILABLE = True
    print("Matplotlib 已找到，多模型对比绘图功能可用。")
except ImportError:
    print("警告: 未找到 matplotlib 库，多模型对比绘图功能将被禁用。如需绘图，请运行 'pip install matplotlib numpy' 安装。")
# ------------------------

# --- 从 evaluate.py 导入核心评估函数 --- 
try:
    # 假设 evaluate.py 与此脚本在同一目录
    from evaluate import evaluate as evaluate_single_model
except ImportError:
    print("错误: 无法从 evaluate.py 导入 evaluate 函数。请确保 evaluate.py 在当前目录或 Python 路径中。")
    exit(1)
# -------------------------------------

def plot_comparison(results, metric_key, report_key, title, ylabel, output_path):
    """绘制多个模型在指定指标上的对比柱状图 (优化美观度)"""
    if not PLOT_AVAILABLE or not results:
        return

    model_names = [res['模型名称'] for res in results]
    values = [res['评估摘要'][report_key][metric_key] for res in results]
    
    # --- 简化柱上标签 --- 
    if '%' in ylabel:
        # 准确率只显示百分比，一位小数
        labels_on_bars = [f"{v:.1f}%" for v in values]
    elif report_key == '加权百分制得分':
        # 加权得分只显示百分制得分，一位小数
        labels_on_bars = [f"{v:.1f}" for v in values]
    else:
        # 其他情况显示两位小数
        labels_on_bars = [f"{v:.2f}" for v in values]
    # ---------------------

    num_models = len(model_names)
    # --- 使用更适合出版物的颜色方案 --- 
    # colors = plt.cm.viridis(np.linspace(0, 1, num_models))
    colors = plt.cm.tab10(np.linspace(0, 1, min(10, num_models))) # tab10 颜色方案
    if num_models > 10:
        colors = list(colors) * (num_models // 10) + list(colors)[:num_models % 10]
    # --------------------------------

    # --- 调整图形尺寸和字体 --- 
    fig, ax = plt.subplots(figsize=(max(6, num_models * 0.7), 6)) # 调整尺寸计算和高度
    plt.rcParams.update({'font.size': 10}) # 基础字体大小
    # --------------------------

    bars = ax.bar(model_names, values, color=colors, width=0.6)

    ax.set_ylabel(ylabel, fontsize=11)
    # --- 修改标题格式 --- 
    benchmark_title = "(TCM-SKIN-Benchmark)"
    # main_title = title # 旧方式，直接使用传入的 title
    # 根据 report_key 动态生成更规范的标题
    if report_key == '整体准确率':
        main_title = "各模型整体准确率对比"
    elif report_key == '加权百分制得分':
        main_title = "各模型加权得分对比"
    else:
        main_title = title # 保留传入的标题作为备用
        
    ax.set_title(f"{main_title}\n{benchmark_title}", fontsize=12, pad=15)
    # ---------------------

    # 根据指标设置 Y 轴范围
    max_val = max(values) if values else 10
    upper_limit = 105 if '%' in ylabel or '100' in ylabel else max_val * 1.15
    ax.set_ylim(0, upper_limit)

    plt.xticks(rotation=45, ha='right', fontsize=10) # 调整旋转角度和字体

    # --- 添加水平网格线 --- 
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True) # 让网格线在柱子后面
    # ---------------------

    # 在柱状图上显示具体数值 (调整位置和字体)
    for bar, label in zip(bars, labels_on_bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + upper_limit*0.01, # 根据 Y 轴上限调整偏移
                label,
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout(pad=1.5) # 调整整体边距
    try:
        plt.savefig(output_path, dpi=300) # 提高 DPI 以适应出版物
        print(f"对比图表已保存: {output_path} (DPI=300)")
    except Exception as e:
        print(f"错误: 保存对比图表失败 {output_path}: {e}")
    plt.close(fig)

# --- Helper function for drawing grouped bar content ---
def _draw_grouped_bar_content(ax, results, category_key, metric_key, xlabel, ylabel, title_part, sort_categories=None):
    """Helper function to draw grouped bar content onto a given Axes object."""
    model_names = [res['模型名称'] for res in results]
    num_models = len(model_names)

    # Dynamic category gathering (same as before)
    all_categories = set()
    for res in results:
        if category_key in res['评估摘要']:
            all_categories.update(res['评估摘要'][category_key].keys())
        else:
            print(f"警告: 模型 {res['模型名称']} 的摘要中缺少 '{category_key}' 数据。")
    if not all_categories:
        print(f"错误: 所有模型的摘要中都缺少 '{category_key}' 数据，无法绘制图表部分: {title_part}")
        return False # Indicate failure

    if sort_categories:
        categories = sorted(list(all_categories), key=lambda x: sort_categories.index(x) if x in sort_categories else 99)
    else:
        categories = sorted(list(all_categories))
    num_categories = len(categories)

    # Data extraction (same as before)
    data = defaultdict(lambda: [0.0] * num_categories)
    category_map = {cat: i for i, cat in enumerate(categories)}
    for res in results:
        model_name = res['模型名称']
        summary_category_data = res['评估摘要'].get(category_key, {})
        for cat_name, stats in summary_category_data.items():
            if cat_name in category_map:
                idx = category_map[cat_name]
                try: data[model_name][idx] = stats.get(metric_key, 0.0)
                except (TypeError, AttributeError): data[model_name][idx] = 0.0

    # Plotting settings
    x = np.arange(num_categories)
    width = 0.8 / num_models
    total_width = width * num_models
    colors = plt.cm.tab10(np.linspace(0, 1, min(10, num_models)))
    if num_models > 10: colors = list(colors) * (num_models // 10) + list(colors)[:num_models % 10]

    # Draw bars
    for i, model_name in enumerate(model_names):
        offset = width * i - total_width / 2 + width / 2
        rects = ax.bar(x + offset, data[model_name], width, label=model_name, color=colors[i])
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., height + 0.5,
                    f'{height:.1f}{"%" if "%" in ylabel else ""}',
                    ha='center', va='bottom', fontsize=8, rotation=90)

    # Format axes
    ax.set_ylabel(ylabel, fontsize=10) # Slightly smaller font for subplots
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
    ax.legend(loc='best', ncol=min(2, num_models), fontsize=8) # Adjusted legend
    ax.set_ylim(0, 105 if '%' in ylabel else max(max(v) for v in data.values()) * 1.15 if data else 10)
    if '%' in ylabel: ax.set_ylim(0, 105)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_title(title_part, fontsize=11) # Title for the subplot
    return True # Indicate success

# --- Modified function to plot single grouped bar chart ---
def plot_comparison_grouped_bar(results, category_key, metric_key, title_part, xlabel, ylabel, output_path, sort_categories=None):
    """绘制多个模型在指定类别上的指标对比 (分组条形图) - Uses helper"""
    if not PLOT_AVAILABLE or not results: return

    # Estimate required width for figure sizing
    num_models = len(results)
    all_categories_est = set().union(*(res['评估摘要'].get(category_key, {}).keys() for res in results))
    num_categories_est = len(all_categories_est)
    est_width = max(8, num_categories_est * num_models * 0.3)

    fig, ax = plt.subplots(figsize=(est_width, 6))
    plt.rcParams.update({'font.size': 10})

    # Call the helper to draw content
    success = _draw_grouped_bar_content(ax, results, category_key, metric_key, xlabel, ylabel, title_part, sort_categories)

    if success:
        # Set overall figure title
        benchmark_title = "(TCM-SKIN-Benchmark)"
        direction_note = " (单选/多选)" if category_key in ['按问题方向准确率', '按问题方向加权得分'] else ""
        metric_desc = "准确率" if metric_key == "准确率(%)" else "加权得分率" if metric_key == "得分(%)" else metric_key
        main_title = f"各模型按{xlabel}{metric_desc}对比{direction_note}" # Use title_part for consistency if needed
        fig.suptitle(f"{main_title}\n{benchmark_title}", fontsize=12, y=1.02) # Use suptitle

        plt.tight_layout(pad=1.5, rect=[0, 0, 1, 0.95]) # Adjust layout for suptitle
        try:
            plt.savefig(output_path, dpi=300)
            print(f"对比图表已保存: {output_path} (DPI=300)")
        except Exception as e:
            print(f"错误: 保存对比图表失败 {output_path}: {e}")
    else:
         print(f"跳过保存图表: {output_path} (数据不足或错误)")
    plt.close(fig)

# --- New function to plot combined grouped bar chart --- 
def plot_combined_comparison(results, category_prefix, xlabel, output_path, sort_categories=None, layout='horizontal'):
    """绘制准确率和加权得分率的左右或上下拼接对比图"""
    if not PLOT_AVAILABLE or not results: return

    category_key_acc = f"{category_prefix}准确率"
    category_key_weighted = f"{category_prefix}加权得分"
    metric_key_acc = "准确率(%)"
    metric_key_weighted = "得分(%)"
    ylabel_acc = "准确率 (%)"
    ylabel_weighted = "加权得分率 (%)"
    title_part_acc = f"按{xlabel}准确率"
    title_part_weighted = f"按{xlabel}加权得分率"

    # Estimate required size based on number of categories and models
    num_models = len(results)
    all_categories_acc = set().union(*(res['评估摘要'].get(category_key_acc, {}).keys() for res in results))
    num_categories = len(all_categories_acc)
    base_width_per_cat = num_models * 0.3
    base_height = 6

    # --- Adjust layout and figure size --- 
    if layout == 'vertical':
        nrows, ncols = 2, 1
        share_axis = 'x' # Share X axis for vertical layout
        # Adjust figsize: width based on categories, height doubled
        est_width = max(8, num_categories * base_width_per_cat * 1.2) 
        est_height = base_height * 1.8 # Increase height for vertical stack
        figsize = (est_width, est_height)
    else: # Default to horizontal
        nrows, ncols = 1, 2
        share_axis = 'y' # Share Y axis for horizontal layout
        # Adjust figsize: width doubled, height standard
        est_width = max(12, num_categories * base_width_per_cat * 2) 
        est_height = base_height
        figsize = (est_width, est_height)
    # --------------------------------------

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=(share_axis=='x'), sharey=(share_axis=='y'))
    ax1 = axes[0] if nrows > 1 or ncols > 1 else axes # Handle case of single plot if needed, though unlikely here
    ax2 = axes[1] if nrows > 1 or ncols > 1 else None 
    plt.rcParams.update({'font.size': 10})

    # Draw top/left plot (Accuracy)
    success1 = _draw_grouped_bar_content(ax1, results, category_key_acc, metric_key_acc, xlabel, ylabel_acc, title_part_acc, sort_categories)
    if layout == 'vertical':
         ax1.set_xlabel('') # Remove x label from top plot
         ax1.legend().set_visible(False) # Optionally hide legend on top plot

    # Draw bottom/right plot (Weighted Score Rate)
    if ax2:
        success2 = _draw_grouped_bar_content(ax2, results, category_key_weighted, metric_key_weighted, xlabel, ylabel_weighted, title_part_weighted, sort_categories)
        if layout == 'horizontal':
             # ax2.set_ylabel('') # Optionally remove y label from right plot
             pass
        # For vertical, keep the xlabel on ax2 (bottom)
    else: # Should not happen with 1x2 or 2x1 layout
        success2 = False 

    if success1 and success2:
        # Set overall figure title
        benchmark_title = "(TCM-SKIN-Benchmark)"
        direction_note = " (单选/多选)" if category_prefix == '按问题方向' else ""
        main_title = f"各模型按{xlabel}准确率与加权得分率对比{direction_note}"
        fig.suptitle(f"{main_title}\n{benchmark_title}", fontsize=13, y=0.99 if layout=='vertical' else 1.03) # Adjust title position

        # Adjust legend for vertical layout (place it centrally below plots)
        if layout == 'vertical' and ax2:
             handles, labels = ax2.get_legend_handles_labels()
             fig.legend(handles, labels, loc='lower center', ncol=min(num_models, 4), bbox_to_anchor=(0.5, -0.05), fontsize=9)
             ax2.legend().set_visible(False) # Hide individual legend on bottom plot
             plt.subplots_adjust(bottom=0.2) # Make space for the legend
        elif layout == 'horizontal' and ax1 and ax2: # Combine legends for horizontal
            handles1, labels1 = ax1.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            # Simple approach: just use one legend if models are the same
            ax1.legend(loc='best', ncol=min(2, num_models), fontsize=8)
            ax2.legend().set_visible(False)
             

        plt.tight_layout(pad=1.5, rect=[0, 0.05 if layout == 'vertical' else 0, 1, 0.93 if layout=='vertical' else 0.95]) # Adjust layout rect
        try:
            plt.savefig(output_path, dpi=300)
            print(f"组合对比图表已保存: {output_path} (DPI=300)")
        except Exception as e:
            print(f"错误: 保存组合对比图表失败 {output_path}: {e}")
    else:
        print(f"跳过保存组合图表: {output_path} (数据不足或错误)")
    plt.close(fig)

def plot_comparison_combined_metrics(results, output_path):
    """绘制对比每个模型整体准确率和加权得分的组合条形图"""
    if not PLOT_AVAILABLE or not results:
        return
        
    model_names = [res['模型名称'] for res in results]
    num_models = len(model_names)

    try:
        accuracy_values = [res['评估摘要']['整体准确率']['准确率(%)'] for res in results]
        weighted_values = [res['评估摘要']['加权百分制得分']['得分(满分100)'] for res in results]
    except KeyError as e:
        print(f"错误: 缺少必要的摘要数据 ({e})，无法绘制组合指标图。")
        return

    x = np.arange(num_models)
    width = 0.35 # 每个指标的柱子宽度

    fig, ax = plt.subplots(figsize=(max(6, num_models * 0.8), 6))
    plt.rcParams.update({'font.size': 10})

    colors = plt.cm.tab10(np.linspace(0, 1, 10)) # 使用tab10前两个颜色

    rects1 = ax.bar(x - width/2, accuracy_values, width, label='整体准确率 (%)', color=colors[0])
    rects2 = ax.bar(x + width/2, weighted_values, width, label='加权得分 (/100)', color=colors[1])

    # 添加标签
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            label_text = f"{height:.1f}{'%' if rects == rects1 else ''}" # 准确率加%
            ax.text(rect.get_x() + rect.get_width() / 2., height + 0.5,
                    label_text,
                    ha='center', va='bottom', fontsize=9)

    # --- 格式化图表 ---
    ax.set_ylabel('分数 / 百分比', fontsize=11)
    benchmark_title = "(TCM-SKIN-Benchmark)"
    # title = "模型整体准确率 vs 加权得分对比" # 旧标题
    main_title = "各模型整体准确率与加权得分对比"
    ax.set_title(f"{main_title}\n{benchmark_title}", fontsize=12, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=10)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0, 105) # Y轴范围0-105

    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout(pad=1.5)
    try:
        plt.savefig(output_path, dpi=300)
        print(f"对比图表已保存: {output_path} (DPI=300)")
    except Exception as e:
        print(f"错误: 保存对比图表失败 {output_path}: {e}")
    plt.close(fig)

def run_comparison(ground_truth_file, prediction_files, output_dir):
    """执行多个模型的评估并生成对比报告和图表"""
    if not os.path.exists(ground_truth_file):
        print(f"错误: 标准答案文件未找到 {ground_truth_file}")
        return

    results = []
    print("\n--- 开始评估多个模型 ---")
    for pred_file in prediction_files:
        model_name = os.path.splitext(os.path.basename(pred_file))[0]
        print(f"评估模型: {model_name} (文件: {pred_file})...")
        if not os.path.exists(pred_file):
            print(f"警告: 预测文件未找到 {pred_file}，跳过模型 {model_name}。")
            continue

        # 调用 evaluate.py 中的函数，只获取摘要，不保存单个文件
        summary = evaluate_single_model(pred_file, ground_truth_file,
                                        save_report=False,
                                        save_plots=False,
                                        return_summary=True)

        if summary and '整体准确率' in summary and '加权百分制得分' in summary:
            results.append({
                "模型名称": model_name,
                "预测文件名": os.path.basename(pred_file),
                "评估摘要": summary
            })
            print(f"模型 {model_name} 评估完成。整体准确率: {summary['整体准确率']['准确率(%)']:.2f}%，加权得分: {summary['加权百分制得分']['得分(满分100)']:.2f}")
        else:
            print(f"警告: 模型 {model_name} 评估失败或返回的摘要数据不完整，跳过。")

    if not results:
        print("\n错误: 没有成功评估的模型，无法生成对比报告和图表。")
        return

    print("\n--- 所有模型评估完成，生成对比结果 ---")

    # --- 保存对比报告 --- 
    os.makedirs(output_dir, exist_ok=True)
    report_output_path = os.path.join(output_dir, "comparison_report.json")
    try:
        # 按加权得分降序排序
        results_sorted = sorted(results, key=lambda x: x['评估摘要']['加权百分制得分']['得分(满分100)'], reverse=True)
        comparison_data = {
            "标准答案文件": os.path.basename(ground_truth_file),
            "对比结果": results_sorted
        }
        with open(report_output_path, 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, ensure_ascii=False, indent=2)
        print(f"\n对比报告已保存至: {report_output_path}")
    except Exception as e:
        print(f"\n错误: 保存对比报告失败: {e}")

    # --- 绘制对比图表 --- 
    if PLOT_AVAILABLE:
        # --- Overall comparisons ---
        plot_comparison( # Overall Accuracy
            results=results_sorted, metric_key='准确率(%)', report_key='整体准确率',
            title='各模型整体准确率对比', ylabel='准确率 (%)',
            output_path=os.path.join(output_dir, "comparison_accuracy.png")
        )
        plot_comparison( # Overall Weighted Score
            results=results_sorted, metric_key='得分(满分100)', report_key='加权百分制得分',
            title='各模型加权得分对比', ylabel='得分 (满分100)',
            output_path=os.path.join(output_dir, "comparison_weighted_score.png")
        )
        plot_comparison_combined_metrics( # Overall Acc vs Weighted Score Combined Bar
            results=results_sorted,
            output_path=os.path.join(output_dir, "comparison_overall_vs_weighted.png")
        )

        # --- Comparisons by Type ---
        sort_type = ['单选题', '多选题', '连续选择题']
        plot_comparison_grouped_bar( # Accuracy by Type
            results=results_sorted, category_key='按题型准确率', metric_key='准确率(%)',
            title_part='按题型准确率', xlabel='题型', ylabel='准确率 (%)',
            output_path=os.path.join(output_dir, "comparison_by_type.png"), sort_categories=sort_type
        )
        plot_comparison_grouped_bar( # Weighted Score by Type
            results=results_sorted, category_key='按题型加权得分', metric_key='得分(%)',
            title_part='按题型加权得分率', xlabel='题型', ylabel='加权得分率 (%)',
            output_path=os.path.join(output_dir, "comparison_weighted_by_type.png"), sort_categories=sort_type
        )
        plot_combined_comparison( # Combined by Type (Acc vs Weighted)
             results=results_sorted, category_prefix='按题型', xlabel='题型',
             output_path=os.path.join(output_dir, "comparison_combined_by_type.png"), sort_categories=sort_type
        )

        # --- Comparisons by Difficulty ---
        sort_difficulty = ['低', '中', '高', '未知']
        plot_comparison_grouped_bar( # Accuracy by Difficulty
            results=results_sorted, category_key='按难度准确率', metric_key='准确率(%)',
            title_part='按难度准确率', xlabel='难度级别', ylabel='准确率 (%)',
            output_path=os.path.join(output_dir, "comparison_by_difficulty.png"), sort_categories=sort_difficulty
        )
        plot_comparison_grouped_bar( # Weighted Score by Difficulty
            results=results_sorted, category_key='按难度加权得分', metric_key='得分(%)',
            title_part='按难度加权得分率', xlabel='难度级别', ylabel='加权得分率 (%)',
            output_path=os.path.join(output_dir, "comparison_weighted_by_difficulty.png"), sort_categories=sort_difficulty
        )
        plot_combined_comparison( # Combined by Difficulty (Acc vs Weighted)
             results=results_sorted, category_prefix='按难度', xlabel='难度级别',
             output_path=os.path.join(output_dir, "comparison_combined_by_difficulty.png"), sort_categories=sort_difficulty
        )

        # --- Comparisons by Direction (Single/Multi Choice Only) ---
        sort_direction = None # Use alphabetical sort
        plot_comparison_grouped_bar( # Accuracy by Direction
            results=results_sorted, category_key='按问题方向准确率', metric_key='准确率(%)',
            title_part='按问题方向准确率', xlabel='问题方向', ylabel='准确率 (%)',
            output_path=os.path.join(output_dir, "comparison_by_direction.png"), sort_categories=sort_direction
        )
        plot_comparison_grouped_bar( # Weighted Score by Direction
            results=results_sorted, category_key='按问题方向加权得分', metric_key='得分(%)',
            title_part='按问题方向加权得分率', xlabel='问题方向', ylabel='加权得分率 (%)',
            output_path=os.path.join(output_dir, "comparison_weighted_by_direction.png"), sort_categories=sort_direction
        )
        plot_combined_comparison( # Combined by Direction (Acc vs Weighted) - VERTICAL LAYOUT
             results=results_sorted, category_prefix='按问题方向', xlabel='问题方向',
             output_path=os.path.join(output_dir, "comparison_combined_by_direction.png"), sort_categories=sort_direction,
             layout='vertical' # <--- Specify vertical layout here
        )

    else:
        print("Matplotlib 不可用，跳过生成对比图表。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="横向评估多个模型的 QA 性能并生成对比报告/图表")
    parser.add_argument("ground_truth_file", help="包含标准答案和题目信息的 JSON 文件路径 (例如: data/full_test_set.json)")
    parser.add_argument("prediction_files", nargs='+', help="一个或多个模型预测结果 JSON 文件路径")
    parser.add_argument("--output_dir", default="model_results", help="保存对比报告和图表的目录路径 (默认为 'model_results')")

    args = parser.parse_args()

    run_comparison(args.ground_truth_file, args.prediction_files, args.output_dir) 