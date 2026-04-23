import json
import argparse
from collections import defaultdict
import os
import sys

# --- 尝试导入 matplotlib --- 
PLOT_AVAILABLE = False
try:
    import matplotlib.pyplot as plt
    import numpy as np
    # 尝试配置中文字体
    plt.rcParams['font.sans-serif'] = ['Heiti TC', 'SimHei', 'Arial Unicode MS', 'Microsoft YaHei'] # 优先使用 macOS 和 Windows 字体
    plt.rcParams['axes.unicode_minus'] = False # 正确显示负号
    PLOT_AVAILABLE = True
    print("Matplotlib 已找到，绘图功能可用。")
except ImportError:
    print("警告: 未找到 matplotlib 库，绘图功能将被禁用。如需绘图，请运行 'pip install matplotlib numpy' 安装。")
# ------------------------

# --- 评分权重配置 ---
BASE_POINTS = {
    "单选题": 1,
    "多选题": 2,
    "连续选择题": 3
}

DIFFICULTY_MULTIPLIER = {
    "低": 1.0,
    "中": 1.5,
    "高": 2.0,
    None: 1.0 # 如果缺少难度信息，默认为1.0
}
# ---------------------

def compare_answers(prediction, ground_truth, question_type):
    """比较预测答案和标准答案是否一致"""
    if question_type == "单选题":
        return isinstance(prediction, str) and prediction == ground_truth
    elif question_type == "多选题":
        if not isinstance(prediction, list) or not isinstance(ground_truth, list):
            return False
        if not prediction and not ground_truth:
             return True
        if not prediction or not ground_truth:
             return False
        try:
            # 确保列表内元素可比较 (转为字符串以防万一)
            pred_sorted = sorted([str(p) for p in prediction])
            gt_sorted = sorted([str(g) for g in ground_truth])
            return pred_sorted == gt_sorted
        except TypeError:
            print(f"警告: 多选题答案列表包含不可排序的元素。Prediction: {prediction}, Ground Truth: {ground_truth}")
            return False
    elif question_type == "连续选择题":
        if not isinstance(prediction, list) or not isinstance(ground_truth, list) or len(prediction) != len(ground_truth):
            return False
        for i, (pred_sub, gt_sub) in enumerate(zip(prediction, ground_truth)):
            sub_type = "多选题" if isinstance(gt_sub, list) else "单选题"
            if not compare_answers(pred_sub, gt_sub, sub_type):
                return False
        return True
    else:
        print(f"警告: 未知的题型 '{question_type}'. 无法比较。")
        return False

def calculate_question_score(question_type, difficulty):
    """根据题型和难度计算单题最高分"""
    base = BASE_POINTS.get(question_type, 0)
    multiplier = DIFFICULTY_MULTIPLIER.get(difficulty, 1.0)
    return base * multiplier

def get_report_dir_path(prediction_path, output_dir):
    """根据预测文件名生成报告保存目录"""
    base_name = os.path.splitext(os.path.basename(prediction_path))[0]
    report_dir = os.path.join(output_dir, base_name)
    return report_dir

def plot_accuracy_by_type(report_data, output_path, model_name):
    """绘制按题型分类的准确率柱状图"""
    if not PLOT_AVAILABLE:
        return

    data = report_data['评估摘要']['按题型准确率']
    types = list(data.keys())
    accuracies = [stats['准确率(%)'] for stats in data.values()]
    labels = [f"{stats['正确数']}/{stats['总数']}" for stats in data.values()]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(types, accuracies, color=['#4e79a7', '#f28e2c', '#e15759'])

    ax.set_ylabel('准确率 (%)')
    ax.set_title(f"{model_name}: 按题型准确率\n(TCM-SKIN-Benchmark)")
    ax.set_ylim(0, 105)

    # 在柱状图上显示具体数值
    for bar, label in zip(bars, labels):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
                f'{height:.1f}%\n({label})',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    try:
        plt.savefig(output_path, dpi=150)
        print(f"图表已保存: {output_path}")
    except Exception as e:
        print(f"错误: 保存图表失败 {output_path}: {e}")
    plt.close(fig)

def plot_accuracy_by_type_difficulty(report_data, output_path, model_name):
    """绘制按题型和难度分类的准确率 (分组柱状图)，只显示数据中存在的难度级别"""
    if not PLOT_AVAILABLE:
        return

    try:
        data = report_data["详细结果"]["按题型和难度准确率"]
    except KeyError:
        print("警告: 报告中未找到 '按题型和难度准确率' 数据，无法绘制图表。")
        return

    question_types = sorted(data.keys())
    if not question_types:
         print("警告: 在 '按题型和难度准确率' 数据中未找到题型信息，无法绘制图表。")
         return

    # --- 确定实际存在的难度级别 (从报告数据中提取， 'None' 已映射为 '未知') ---
    actual_difficulties_in_data = set()
    for qt_data in data.values():
        actual_difficulties_in_data.update(qt_data.keys())

    if not actual_difficulties_in_data:
         print("警告: 在 '按题型和难度准确率' 数据中未找到难度信息，无法绘制图表。")
         return

    # 按 '低', '中', '高', '未知' 排序实际存在的难度
    # (如果 '未知' 不在 actual_difficulties_in_data 中，它就不会出现在排序结果里)
    difficulties_to_plot = sorted(list(actual_difficulties_in_data), key=lambda x: ('低', '中', '高', '未知').index(x) if x in ('低', '中', '高', '未知') else 99)


    # --- 后续代码使用 difficulties_to_plot ---
    accuracies = defaultdict(lambda: [0.0] * len(difficulties_to_plot))
    labels = defaultdict(lambda: [''] * len(difficulties_to_plot))
    totals = defaultdict(lambda: [0] * len(difficulties_to_plot))

    difficulty_map = {d: i for i, d in enumerate(difficulties_to_plot)} # 使用实际要绘制的难度

    for qt in question_types:
        for diff_key, stats in data.get(qt, {}).items(): # diff_key is already mapped
            if diff_key in difficulty_map: # Check if this difficulty should be plotted
                idx = difficulty_map[diff_key]
                accuracies[qt][idx] = stats['准确率(%)']
                labels[qt][idx] = f"({stats['正确数']}/{stats['总数']})"
                totals[qt][idx] = stats['总数']

    x = np.arange(len(difficulties_to_plot)) # 使用实际数量
    width = 0.25 # 柱子宽度
    num_types = len(question_types)

    fig, ax = plt.subplots(figsize=(max(8, len(difficulties_to_plot) * 2 * num_types * 0.5), 6)) # 根据难度和题型数量调整宽度

    colors = ['#4e79a7', '#f28e2c', '#e15759'] # Blue, Orange, Red

    for i, qt in enumerate(question_types):
         # 计算偏移量使分组居中
        offset = width * (i - (num_types - 1) / 2)
        # 仅获取需要绘制的难度的数据
        current_accuracies = [accuracies[qt][difficulty_map[diff]] for diff in difficulties_to_plot]
        rects = ax.bar(x + offset, current_accuracies, width, label=qt, color=colors[i % len(colors)])

        for j, rect in enumerate(rects):
             diff_key = difficulties_to_plot[j] # 获取这个柱子对应的难度
             height = rect.get_height()
             # 使用映射的索引获取正确的 total 和 label
             if totals[qt][difficulty_map[diff_key]] > 0:
                 label_text = labels[qt][difficulty_map[diff_key]]
                 ax.text(rect.get_x() + rect.get_width() / 2., height + 1,
                         f'{height:.1f}%\n{label_text}',
                         ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('准确率 (%)')
    ax.set_title(f"{model_name}: 按题型与难度准确率\n(TCM-SKIN-Benchmark)")
    # 设置X轴刻度位置和标签
    ax.set_xticks(x) #刻度居中在每个难度组
    ax.set_xticklabels(difficulties_to_plot) # 设置刻度标签为实际绘制的难度

    ax.legend(loc='upper right', ncol=num_types)
    ax.set_ylim(0, 110)
    plt.tight_layout()
    try:
        plt.savefig(output_path, dpi=150)
        print(f"图表已保存: {output_path}")
    except Exception as e:
        print(f"错误: 保存图表失败 {output_path}: {e}")
    plt.close(fig)

# 首先，定义新的绘图函数 plot_accuracy_by_direction
def plot_accuracy_by_direction(report_data, output_path, model_name):
    """绘制按问题方向分类的准确率柱状图"""
    if not PLOT_AVAILABLE:
        return

    try:
        data = report_data['评估摘要']['按问题方向准确率']
    except KeyError:
        print("警告: 报告中未找到 '按问题方向准确率' 数据，无法绘制图表。")
        return

    directions = list(data.keys())
    if not directions:
        print("警告: '按问题方向准确率' 数据为空，无法绘制图表。")
        return

    accuracies = [stats['准确率(%)'] for stats in data.values()]
    labels = [f"{stats['正确数']}/{stats['总数']}" for stats in data.values()]
    colors = plt.cm.viridis(np.linspace(0, 1, len(directions))) # 使用 colormap

    fig, ax = plt.subplots(figsize=(max(8, len(directions) * 1.5), 5)) # 动态调整宽度
    bars = ax.bar(directions, accuracies, color=colors)

    ax.set_ylabel('准确率 (%)')
    ax.set_title(f"{model_name}: 按问题方向准确率 (单选/多选)\n(TCM-SKIN-Benchmark)")
    ax.set_ylim(0, 105)
    plt.xticks(rotation=30, ha='right') # 旋转标签防止重叠

    # 在柱状图上显示具体数值
    for bar, label in zip(bars, labels):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
                f'{height:.1f}%\n({label})',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    try:
        plt.savefig(output_path, dpi=150)
        print(f"图表已保存: {output_path}")
    except Exception as e:
        print(f"错误: 保存图表失败 {output_path}: {e}")
    plt.close(fig)

# 定义新的绘图函数 plot_accuracy_by_direction_type
def plot_accuracy_by_direction_type(report_data, output_path, model_name):
    """绘制按问题方向和题型分类的准确率 (分组柱状图)"""
    if not PLOT_AVAILABLE:
        return

    try:
        data = report_data['详细结果']['按问题方向和题型准确率']
    except KeyError:
        print("警告: 报告中未找到 '按问题方向和题型准确率' 数据，无法绘制图表。")
        return

    directions = sorted(data.keys())
    if not directions:
        print("警告: '按问题方向和题型准确率' 数据为空，无法绘制图表。")
        return

    # 从数据中动态获取所有题型
    all_types = set()
    for dir_data in data.values():
        all_types.update(dir_data.keys())
    question_types = sorted(list(all_types))
    if not question_types:
        print("警告: '按问题方向和题型准确率' 数据中未找到题型信息，无法绘制图表。")
        return

    accuracies = defaultdict(lambda: [0.0] * len(directions))
    labels = defaultdict(lambda: [''] * len(directions))
    totals = defaultdict(lambda: [0] * len(directions))

    direction_map = {d: i for i, d in enumerate(directions)}

    for qt in question_types:
        for dir_name, type_data in data.items():
            stats = type_data.get(qt)
            if stats and dir_name in direction_map:
                idx = direction_map[dir_name]
                accuracies[qt][idx] = stats['准确率(%)']
                labels[qt][idx] = f"({stats['正确数']}/{stats['总数']})"
                totals[qt][idx] = stats['总数']

    x = np.arange(len(directions))
    # 根据题型数量调整宽度和偏移量
    num_types = len(question_types)
    width = 0.8 / num_types # 总宽度设为0.8，均分给各题型
    total_width = width * num_types

    fig, ax = plt.subplots(figsize=(max(12, len(directions) * 1.8), 6)) # 动态调整宽度

    # 使用 tab10 colormap
    colors = plt.cm.tab10(np.linspace(0, 1, min(10, num_types)))
    if num_types > 10: # 如果类型过多，循环使用颜色
        colors = list(colors) * (num_types // 10) + list(colors)[:num_types % 10]


    for i, qt in enumerate(question_types):
        # 计算偏移量使分组居中
        offset = width * i - total_width / 2 + width / 2
        rects = ax.bar(x + offset, accuracies[qt], width, label=qt, color=colors[i % len(colors)])
        for j, rect in enumerate(rects):
            height = rect.get_height()
            if totals[qt][j] > 0: # Only label if there are questions
                ax.text(rect.get_x() + rect.get_width() / 2., height + 0.5, # 微调标签位置
                        f'{height:.1f}%\n{labels[qt][j]}',
                        ha='center', va='bottom', fontsize=7) # 减小字体

    ax.set_ylabel('准确率 (%)')
    ax.set_title(f"{model_name}: 按问题方向与题型准确率\n(TCM-SKIN-Benchmark)")
    ax.set_xticks(x, directions) # 设置刻度位置和标签
    plt.xticks(rotation=30, ha='right') # 旋转标签
    ax.legend(loc='upper right', ncol=min(3, num_types)) # 最多显示3列图例
    ax.set_ylim(0, 110)
    plt.tight_layout()
    try:
        plt.savefig(output_path, dpi=150)
        print(f"图表已保存: {output_path}")
    except Exception as e:
        print(f"错误: 保存图表失败 {output_path}: {e}")
    plt.close(fig)

# 定义新的绘图函数 plot_performance_summary
def plot_performance_summary(report_data, output_path, model_name):
    """绘制包含整体准确率和加权得分的性能概览图"""
    if not PLOT_AVAILABLE:
        return

    try:
        overall_acc_data = report_data["评估摘要"]["整体准确率"]
        weighted_score_data = report_data["评估摘要"]["加权百分制得分"]

        overall_accuracy = overall_acc_data["准确率(%)"]
        weighted_score = weighted_score_data["得分(满分100)"]
        overall_label = f"{overall_acc_data["正确数"]}/{overall_acc_data["总数"]}"
        # 修正加权得分标签的格式化
        weighted_label = f"{weighted_score_data["获得加权分"]:.1f}/{weighted_score_data["最高可能加权分"]:.1f}"

    except KeyError as e:
        print(f"警告: 报告摘要中缺少必要数据 ({e})，无法绘制性能概览图。")
        return
    except TypeError as e:
         print(f"警告: 报告摘要中数据类型错误 ({e})，无法绘制性能概览图。")
         return


    metrics = ["整体准确率", "加权百分制得分"]
    values = [overall_accuracy, weighted_score]
    labels_on_bars = [f"{overall_accuracy:.1f}%\n({overall_label})", f"{weighted_score:.1f}\n(加权分: {weighted_label})"]
    colors = ["#1f77b4", "#ff7f0e"] # Blue, Orange

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(metrics, values, color=colors, width=0.6) # 调整柱子宽度

    ax.set_ylabel("分数 / 百分比 (%)")
    ax.set_title(f"{model_name}: 整体准确率与加权得分\n(TCM-SKIN-Benchmark)")
    ax.set_ylim(0, 105)

    # 在柱状图上显示具体数值和标签
    for bar, label in zip(bars, labels_on_bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
                label,
                ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    try:
        plt.savefig(output_path, dpi=150)
        print(f"图表已保存: {output_path}")
    except Exception as e: # 捕获更通用的异常
        print(f"错误: 保存性能概览图失败 {output_path}: {e}")
    plt.close(fig)

def evaluate(prediction_path, ground_truth_path, save_report=True, save_plots=False, output_dir=".", return_summary=False):
    """加载数据并执行评估，包括准确率、百分制加权评分和生成详细报告/图表 (增加问题方向维度)。

    Args:
        prediction_path (str): 预测文件路径。
        ground_truth_path (str): 标准答案文件路径。
        save_report (bool, optional): 是否保存单个模型的详细JSON报告。 默认为 True。
        save_plots (bool, optional): 是否保存单个模型的可视化图表。 默认为 False。
        output_dir (str, optional): 单个模型报告/图表的根输出目录。 默认为 "."。
        return_summary (bool, optional): 是否返回评估摘要字典而不是仅执行操作。 默认为 False。

    Returns:
        dict or None: 如果 return_summary 为 True，则返回包含评估摘要的字典；否则返回 None。
    """
    if not os.path.exists(ground_truth_path):
        print(f"错误: 标准答案文件未找到 {ground_truth_path}")
        return
    if not os.path.exists(prediction_path):
        print(f"错误: 预测结果文件未找到 {prediction_path}")
        return

    try:
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            ground_truth_data = json.load(f)
    except json.JSONDecodeError:
        print(f"错误: 标准答案文件 JSON 解码失败 {ground_truth_path}")
        return

    try:
        with open(prediction_path, 'r', encoding='utf-8') as f:
            prediction_data = json.load(f)
    except json.JSONDecodeError:
        print(f"错误: 预测结果文件 JSON 解码失败 {prediction_path}")
        return

    # --- 数据处理与检查 ---
    gt_map = {item['id']: item for item in ground_truth_data}
    pred_map = {item['id']: item for item in prediction_data}
    model_name = os.path.splitext(os.path.basename(prediction_path))[0]
    missing_ids_in_pred = set(gt_map.keys()) - set(pred_map.keys())
    extra_ids_in_pred = set(pred_map.keys()) - set(gt_map.keys())

    # --- 重新确保所有计数器被初始化 --- 
    correct_counts_by_type = defaultdict(int)
    total_counts_by_type = defaultdict(int)
    earned_score_by_type = defaultdict(float) 
    max_score_by_type = defaultdict(float)    

    correct_counts_by_difficulty = defaultdict(int)
    total_counts_by_difficulty = defaultdict(int)   # 确保此行存在
    earned_score_by_difficulty = defaultdict(float) 
    max_score_by_difficulty = defaultdict(float)    

    correct_counts_by_direction = defaultdict(int)
    total_counts_by_direction = defaultdict(int)
    earned_score_by_direction = defaultdict(float) 
    max_score_by_direction = defaultdict(float)    

    detailed_correct_counts_type_diff = defaultdict(lambda: defaultdict(int))
    detailed_total_counts_type_diff = defaultdict(lambda: defaultdict(int))
    detailed_correct_counts_dir_type = defaultdict(lambda: defaultdict(int))
    detailed_total_counts_dir_type = defaultdict(lambda: defaultdict(int))
    detailed_correct_counts_dir_diff = defaultdict(lambda: defaultdict(int))
    detailed_total_counts_dir_diff = defaultdict(lambda: defaultdict(int))
    detailed_earned_score_dir_type = defaultdict(lambda: defaultdict(float))
    detailed_max_score_dir_type = defaultdict(lambda: defaultdict(float))
    detailed_earned_score_dir_diff = defaultdict(lambda: defaultdict(float))
    detailed_max_score_dir_diff = defaultdict(lambda: defaultdict(float))
    # ---------------------------------------

    total_earned_score = 0.0
    total_max_possible_score = 0.0

    # --- 遍历标准答案进行评估 ---
    evaluated_ids = set()
    all_directions = set()

    for gt_id, gt_item in gt_map.items():
        evaluated_ids.add(gt_id)
        q_type = gt_item.get('question_type')
        difficulty = gt_item.get('difficulty')
        ground_truth = gt_item.get('answer')
        direction_raw = gt_item.get('question_direction', '未知')

        # --- 处理 direction --- (Logic from previous edit)
        if isinstance(direction_raw, list):
            direction = "混合方向"
        elif direction_raw is None:
            direction = "未知"
        else:
            direction = str(direction_raw)
        all_directions.add(direction)
        # ----------------------

        if q_type not in BASE_POINTS:
            print(f"警告: ID {gt_id} 的题型 '{q_type}' 不在评分配置中，跳过评估。")
            continue

        # --- 更新总数和最高分计数器 ---
        max_score_for_question = calculate_question_score(q_type, difficulty)
        total_max_possible_score += max_score_for_question

        total_counts_by_type[q_type] += 1
        max_score_by_type[q_type] += max_score_for_question

        diff_key_for_count = difficulty # Use original None for dict key
        total_counts_by_difficulty[diff_key_for_count] += 1
        max_score_by_difficulty[diff_key_for_count] += max_score_for_question

        total_counts_by_direction[direction] += 1
        max_score_by_direction[direction] += max_score_for_question

        detailed_total_counts_type_diff[q_type][diff_key_for_count] += 1
        detailed_total_counts_dir_type[direction][q_type] += 1
        detailed_total_counts_dir_diff[direction][diff_key_for_count] += 1
        detailed_max_score_dir_type[direction][q_type] += max_score_for_question
        detailed_max_score_dir_diff[direction][diff_key_for_count] += max_score_for_question

        # --- 比较答案 ---
        is_correct = False
        if gt_id not in pred_map:
            pass # Missing, count as incorrect
        else:
            pred_item = pred_map[gt_id]
            prediction = pred_item.get('prediction')
            if prediction is None:
                print(f"警告: ID {gt_id} 的预测结果中缺少 'prediction' 字段，视为错误。")
            else:
                try:
                    is_correct = compare_answers(prediction, ground_truth, q_type)
                except Exception as e:
                    print(f"错误: 比较 ID {gt_id} 时发生异常: {e}. 视为错误。")

        # --- 更新正确数和获得分数计数器 ---
        if is_correct:
            correct_counts_by_type[q_type] += 1
            earned_score_by_type[q_type] += max_score_for_question

            correct_counts_by_difficulty[diff_key_for_count] += 1
            earned_score_by_difficulty[diff_key_for_count] += max_score_for_question

            correct_counts_by_direction[direction] += 1
            earned_score_by_direction[direction] += max_score_for_question

            total_earned_score += max_score_for_question
            
            detailed_correct_counts_type_diff[q_type][diff_key_for_count] += 1
            detailed_correct_counts_dir_type[direction][q_type] += 1
            detailed_correct_counts_dir_diff[direction][diff_key_for_count] += 1
            detailed_earned_score_dir_type[direction][q_type] += max_score_for_question
            detailed_earned_score_dir_diff[direction][diff_key_for_count] += max_score_for_question

    overall_total = len(evaluated_ids)
    overall_correct = sum(correct_counts_by_type.values())

    # --- 计算最终指标 (增加类别加权得分率) ---
    overall_accuracy = overall_correct / overall_total * 100 if overall_total > 0 else 0
    final_weighted_score = (total_earned_score / total_max_possible_score * 100) if total_max_possible_score > 0 else 0

    # 1. 按题型准确率 (Summary)
    accuracy_by_type_summary = {}
    for q_type, total in total_counts_by_type.items():
        correct = correct_counts_by_type[q_type]
        accuracy = correct / total * 100 if total > 0 else 0
        accuracy_by_type_summary[q_type] = {"准确率(%)": round(accuracy, 2), "正确数": correct, "总数": total}

    # 1b. 按题型加权得分率 (Summary)
    weighted_score_by_type_summary = {}
    for q_type, max_s in max_score_by_type.items():
        earned_s = earned_score_by_type[q_type]
        score_perc = (earned_s / max_s * 100) if max_s > 0 else 0
        weighted_score_by_type_summary[q_type] = {
            "得分(%)": round(score_perc, 2),
            "获得加权分": round(earned_s, 1),
            "最高可能加权分": round(max_s, 1)
        }

    # 2. 按难度准确率 (Summary)
    accuracy_by_difficulty_summary = {}
    for diff, total in total_counts_by_difficulty.items():
        correct = correct_counts_by_difficulty[diff]
        accuracy = correct / total * 100 if total > 0 else 0
        diff_key = diff if diff is not None else "未知"
        accuracy_by_difficulty_summary[diff_key] = {"准确率(%)": round(accuracy, 2), "正确数": correct, "总数": total}
        
    # 2b. 按难度加权得分率 (Summary)
    weighted_score_by_difficulty_summary = {}
    for diff, max_s in max_score_by_difficulty.items():
        earned_s = earned_score_by_difficulty[diff]
        score_perc = (earned_s / max_s * 100) if max_s > 0 else 0
        diff_key = diff if diff is not None else "未知"
        weighted_score_by_difficulty_summary[diff_key] = {
            "得分(%)": round(score_perc, 2),
            "获得加权分": round(earned_s, 1),
            "最高可能加权分": round(max_s, 1)
        }

    # --- 3. 按问题方向准确率 (Summary) - 仅包含单选和多选 ---
    accuracy_by_direction_summary = {}
    temp_correct_by_dir = defaultdict(int)
    temp_total_by_dir = defaultdict(int)
    target_qtypes = {"单选题", "多选题"}

    for direction, type_counts in detailed_total_counts_dir_type.items():
        if direction == "混合方向": # 跳过混合方向
            continue
        for q_type, total in type_counts.items():
            if q_type in target_qtypes:
                temp_total_by_dir[direction] += total
                # 从 detailed_correct_counts_dir_type 获取对应的正确数
                temp_correct_by_dir[direction] += detailed_correct_counts_dir_type.get(direction, {}).get(q_type, 0)
                
    for direction, total in temp_total_by_dir.items():
        correct = temp_correct_by_dir[direction]
        accuracy = correct / total * 100 if total > 0 else 0
        accuracy_by_direction_summary[direction] = {"准确率(%)": round(accuracy, 2), "正确数": correct, "总数": total}
    # --------------------------------------------------

    # --- 3b. 按问题方向加权得分率 (Summary) - 仅包含单选和多选 ---
    weighted_score_by_direction_summary = {}
    temp_earned_score_by_dir = defaultdict(float)
    temp_max_score_by_dir = defaultdict(float)

    for direction, type_scores in detailed_max_score_dir_type.items():
        if direction == "混合方向": # 跳过混合方向
            continue
        for q_type, max_s in type_scores.items():
            if q_type in target_qtypes:
                temp_max_score_by_dir[direction] += max_s
                # 从 detailed_earned_score_dir_type 获取对应的得分
                temp_earned_score_by_dir[direction] += detailed_earned_score_dir_type.get(direction, {}).get(q_type, 0.0)

    for direction, max_s in temp_max_score_by_dir.items():
        earned_s = temp_earned_score_by_dir[direction]
        score_perc = (earned_s / max_s * 100) if max_s > 0 else 0
        weighted_score_by_direction_summary[direction] = {
            "得分(%)": round(score_perc, 2),
            "获得加权分": round(earned_s, 1),
            "最高可能加权分": round(max_s, 1)
        }
    # ------------------------------------------------------

    # 4. 按题型和难度准确率 (Detailed)
    detailed_accuracy_type_diff = defaultdict(lambda: defaultdict(dict))
    for q_type, diff_counts in detailed_total_counts_type_diff.items():
        for diff, total in diff_counts.items():
            correct = detailed_correct_counts_type_diff[q_type][diff]
            accuracy = correct / total * 100 if total > 0 else 0
            diff_key = diff if diff is not None else "未知"
            detailed_accuracy_type_diff[q_type][diff_key] = {"准确率(%)": round(accuracy, 2), "正确数": correct, "总数": total}

    # 5. 按问题方向和题型准确率 (Detailed) - 新增
    detailed_accuracy_dir_type = defaultdict(lambda: defaultdict(dict))
    for direction, type_counts in detailed_total_counts_dir_type.items():
        for q_type, total in type_counts.items():
            correct = detailed_correct_counts_dir_type[direction][q_type]
            accuracy = correct / total * 100 if total > 0 else 0
            detailed_accuracy_dir_type[direction][q_type] = {"准确率(%)": round(accuracy, 2), "正确数": correct, "总数": total}

    # 6. 按问题方向和难度准确率 (Detailed) - 新增
    detailed_accuracy_dir_diff = defaultdict(lambda: defaultdict(dict))
    for direction, diff_counts in detailed_total_counts_dir_diff.items():
        for diff, total in diff_counts.items():
            correct = detailed_correct_counts_dir_diff[direction][diff]
            accuracy = correct / total * 100 if total > 0 else 0
            diff_key = diff if diff is not None else "未知"
            detailed_accuracy_dir_diff[direction][diff_key] = {"准确率(%)": round(accuracy, 2), "正确数": correct, "总数": total}


    # --- 准备报告数据 (增加类别加权得分) --- 
    report_data = {
        "元数据": {
            "预测文件名": os.path.basename(prediction_path),
            "标准答案文件名": os.path.basename(ground_truth_path),
            "评估配置": {
                "基础分值": BASE_POINTS,
                "难度乘数": DIFFICULTY_MULTIPLIER
            },
            "所有问题方向": sorted([d for d in all_directions if d != "混合方向" or any(q_type not in target_qtypes for q_type in detailed_total_counts_dir_type.get(d, {}))])
        },
        "评估摘要": {
            "整体准确率": {
                "准确率(%)": round(overall_accuracy, 2),
                "正确数": overall_correct,
                "总数": overall_total,
            },
            "加权百分制得分": {
                 "得分(满分100)": round(final_weighted_score, 2),
                 "获得加权分": round(total_earned_score, 1),
                 "最高可能加权分": round(total_max_possible_score, 1),
            },
            "按题型准确率": accuracy_by_type_summary,
            "按题型加权得分": weighted_score_by_type_summary,
            "按难度准确率": accuracy_by_difficulty_summary,
            "按难度加权得分": weighted_score_by_difficulty_summary,
            "按问题方向准确率": accuracy_by_direction_summary,
            "按问题方向加权得分": weighted_score_by_direction_summary
        },
        "详细结果": {
            # 按题型维度
            "按题型和难度准确率": {qt: dict(diff_acc) for qt, diff_acc in detailed_accuracy_type_diff.items()},
            "按题型和难度正确数": {qt: dict(diff_counts) for qt, diff_counts in detailed_correct_counts_type_diff.items()},
            "按题型和难度总数": {qt: { (diff if diff is not None else "未知"): count for diff, count in diff_counts.items()}
                              for qt, diff_counts in detailed_total_counts_type_diff.items()},
            # 按问题方向维度 - 新增
            "按问题方向和题型准确率": {direction: dict(type_acc) for direction, type_acc in detailed_accuracy_dir_type.items()},
            "按问题方向和题型正确数": {direction: dict(type_counts) for direction, type_counts in detailed_correct_counts_dir_type.items()},
            "按问题方向和题型总数": {direction: dict(type_counts) for direction, type_counts in detailed_total_counts_dir_type.items()},
            "按问题方向和难度准确率": {direction: dict(diff_acc) for direction, diff_acc in detailed_accuracy_dir_diff.items()},
            "按问题方向和难度正确数": {direction: dict(diff_counts) for direction, diff_counts in detailed_correct_counts_dir_diff.items()},
            "按问题方向和难度总数": {direction: { (diff if diff is not None else "未知"): count for diff, count in diff_counts.items()}
                              for direction, diff_counts in detailed_total_counts_dir_diff.items()},
        },
        "警告信息": {
            "预测中缺失的ID": sorted(list(missing_ids_in_pred)),
            "预测中多余的ID": sorted(list(extra_ids_in_pred))
        }
    }

    # --- 输出结果与保存 (增加类别加权得分信息) --- 
    print("\n--- 评估结果摘要 ---")
    print(f"评估文件: {os.path.basename(ground_truth_path)}")
    print(f"总题目数 (参与评估): {overall_total}")
    print(f"整体准确率: {report_data['评估摘要']['整体准确率']['准确率(%)']:.2f}% ({report_data['评估摘要']['整体准确率']['正确数']}/{report_data['评估摘要']['整体准确率']['总数']})")
    print(f"百分制加权得分: {report_data['评估摘要']['加权百分制得分']['得分(满分100)']:.2f} / 100.00")

    print("按题型准确率:")
    for q_type, stats in report_data['评估摘要']['按题型准确率'].items():
        print(f"  - {q_type}: {stats['准确率(%)']:.2f}% ({stats['正确数']}/{stats['总数']})" )

    print("按题型加权得分:")
    for q_type, stats in report_data['评估摘要']['按题型加权得分'].items():
        print(f"  - {q_type}: {stats['得分(%)']:.2f}% (加权分: {stats['获得加权分']:.1f}/{stats['最高可能加权分']:.1f})" )

    print("按难度准确率:")
    sorted_diffs = sorted(report_data['评估摘要']['按难度准确率'].keys(), key=lambda x: ('低', '中', '高', '未知').index(x) if x in ('低', '中', '高', '未知') else 99)
    for diff in sorted_diffs:
        stats = report_data['评估摘要']['按难度准确率'][diff]
        print(f"  - {diff}: {stats['准确率(%)']:.2f}% ({stats['正确数']}/{stats['总数']})" )

    print("按难度加权得分:")
    sorted_diffs_score = sorted(report_data['评估摘要']['按难度加权得分'].keys(), key=lambda x: ('低', '中', '高', '未知').index(x) if x in ('低', '中', '高', '未知') else 99)
    for diff in sorted_diffs_score:
        stats = report_data['评估摘要']['按难度加权得分'][diff]
        print(f"  - {diff}: {stats['得分(%)']:.2f}% (加权分: {stats['获得加权分']:.1f}/{stats['最高可能加权分']:.1f})" )

    print("按问题方向准确率:")
    sorted_dirs = sorted(report_data['评估摘要']['按问题方向准确率'].keys())
    for direction in sorted_dirs:
        stats = report_data['评估摘要']['按问题方向准确率'][direction]
        print(f"  - {direction}: {stats['准确率(%)']:.2f}% ({stats['正确数']}/{stats['总数']})" )

    print("按问题方向加权得分:")
    sorted_dirs_score = sorted(report_data['评估摘要']['按问题方向加权得分'].keys())
    for direction in sorted_dirs_score:
        stats = report_data['评估摘要']['按问题方向加权得分'][direction]
        print(f"  - {direction}: {stats['得分(%)']:.2f}% (加权分: {stats['获得加权分']:.1f}/{stats['最高可能加权分']:.1f})" )

    # 确定报告和图表保存目录
    report_dir = get_report_dir_path(prediction_path, output_dir)

    # 保存详细报告
    if save_report:
        os.makedirs(report_dir, exist_ok=True) # 创建目录
        report_filename = f"{os.path.splitext(os.path.basename(prediction_path))[0]}_report.json"
        report_output_path = os.path.join(report_dir, report_filename)
        try:
            with open(report_output_path, 'w', encoding='utf-8') as f:
                # 使用自定义 encoder 处理 defaultdict
                class DefaultdictEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if isinstance(obj, defaultdict):
                            return dict(obj)
                        # Let the base class default method raise the TypeError
                        return json.JSONEncoder.default(self, obj)
                json.dump(report_data, f, ensure_ascii=False, indent=2, cls=DefaultdictEncoder)
            print(f"\n详细评估报告已保存至: {report_output_path}")
        except Exception as e:
            print(f"\n错误: 保存评估报告失败: {e}")
            # 打印 report_data 以便调试
            # import pprint
            # pprint.pprint(report_data)
    else:
         print("\n详细评估报告未保存 (根据选项)。")

    # 保存图表 (增加 direction 相关图表)
    if save_plots:
        if PLOT_AVAILABLE:
            os.makedirs(report_dir, exist_ok=True) # 确保目录存在

            # 性能概览图 (新增)
            plot_summary_path = os.path.join(report_dir, "performance_summary.png")
            plot_performance_summary(report_data, plot_summary_path, model_name)

            # 原有图表
            plot_acc_type_path = os.path.join(report_dir, "accuracy_by_type.png")
            plot_accuracy_by_type(report_data, plot_acc_type_path, model_name)

            plot_acc_type_diff_path = os.path.join(report_dir, "accuracy_by_type_difficulty.png")
            plot_accuracy_by_type_difficulty(report_data, plot_acc_type_diff_path, model_name)

            # 按方向图表
            plot_acc_dir_path = os.path.join(report_dir, "accuracy_by_direction.png")
            plot_accuracy_by_direction(report_data, plot_acc_dir_path, model_name)

            plot_acc_dir_type_path = os.path.join(report_dir, "accuracy_by_direction_type.png")
            plot_accuracy_by_direction_type(report_data, plot_acc_dir_type_path, model_name)

            # 可以根据需要添加 plot_accuracy_by_direction_difficulty 的调用
            # plot_acc_dir_diff_path = os.path.join(report_dir, "accuracy_by_direction_difficulty.png")
            # plot_accuracy_by_direction_difficulty(report_data, plot_acc_dir_diff_path, model_name) # 需要先实现这个函数

        else:
            print("警告: --save_plots 选项已指定，但 matplotlib 不可用，无法生成图表。")
    else:
        print("图表未生成 (根据选项)。")

    if return_summary:
        return report_data.get("评估摘要", {}) # 返回摘要部分，如果不存在则返回空字典

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估中医皮肤科 QA 基准测试结果 (包括准确率、加权分数、报告和图表，增加按问题方向统计)") # 更新描述
    parser.add_argument("prediction_file", help="模型预测结果 JSON 文件路径")
    parser.add_argument("ground_truth_file", help="包含标准答案和题目信息的 JSON 文件路径 (例如: data/full_test_set.json)")
    parser.add_argument("--output_dir", default="model_results", help="保存评估报告和图表的根目录路径 (默认为 'model_results')")
    parser.add_argument("--no_report", action="store_true", help="禁止保存详细的 JSON 评估报告")
    parser.add_argument("--save_plots", action="store_true", help="生成并保存评估结果的可视化图表 (需要 matplotlib 和 numpy)")

    args = parser.parse_args()

    evaluate(args.prediction_file, args.ground_truth_file,
             save_report=not args.no_report,
             save_plots=args.save_plots,
             output_dir=args.output_dir) 