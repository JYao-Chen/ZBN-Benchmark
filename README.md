# ZBN-Benchmark 中医皮肤科 QA 基准测试

这是一个用于评估大型语言模型在中医皮肤科赵炳南流派知识问答方面能力的基准测试仓库。

## 目录结构

```
ZBN-Benchmark/
├── data/                  # 标准化后的测试集和答案文件
│   ├── questions.json       # 仅包含问题和选项
│   ├── answers.json         # 仅包含问题ID和答案
│   └── full_test_set.json   # 包含完整的问题、选项、答案、类型、难度等信息
├── scripts/               # 测评和推理脚本
│   ├── evaluate.py              # 单个模型评估脚本
│   ├── evaluate_multiple.py     # 多个模型对比评估脚本
│   └── creat_openai_eval_async.py # 使用 OpenAI API 进行异步推理的示例脚本
├── model_results/         # 各模型在本基准测试上的原始输出结果
│   ├── Qwen2.5-72B/
│   ├── QwQ-32B/
│   ├── ... (其他模型结果)
├── paper_figures/         # 论文中使用的相关数据分析配图
│   ├── combined_direction_distributions.png
│   ├── ... (其他图表)
└── README.md              # 本说明文件
```

## 数据说明

*   `data/` 目录包含了用于评估的问答数据。
*   `questions.json` 适合直接用于模型推理输入。
*   `answers.json` 包含标准答案，用于评估脚本。
*   `full_test_set.json` 包含所有元数据，方便详细分析。

## 脚本说明

*   `scripts/evaluate.py`: 用于评估单个模型预测结果的准确率和加权得分。需要提供模型预测文件和 `data/full_test_set.json` 文件。
*   `scripts/evaluate_multiple.py`: 用于横向对比多个模型评估结果，并生成对比报告和图表。需要提供多个模型的预测文件和 `data/full_test_set.json` 文件。
*   `scripts/creat_openai_eval_async.py`: 提供了一个使用 OpenAI 异步 API 对 `data/questions.json` 进行推理的示例，可以根据需要修改以适配不同的模型或 API。
    **注意:** 此脚本需要通过命令行参数 `--api_key` 和 `--base_url` 提供有效的 OpenAI API 凭据。其他参数如模型名称、温度、并发数等也可通过命令行调整。

## 模型结果

*   `model_results/` 目录存放了不同模型在此基准上运行 `scripts/creat_openai_eval_async.py` 后生成的原始预测输出文件 (通常是 JSON 格式)。

## 论文配图

*   `paper_figures/` 目录包含与本基准测试相关的论文中使用的数据分析图表。

## 使用方法

1.  **准备模型预测**: 使用 `data/questions.json` 或 `data/full_test_set.json` 中的问题，让你的模型生成预测结果。请确保预测结果的格式与 `scripts/evaluate.py` 或 `scripts/evaluate_multiple.py` 所期望的格式一致 (通常是一个包含 `id` 和 `prediction` 字段的 JSON 列表)。将预测结果保存为 JSON 文件。
    *   如果使用 `scripts/creat_openai_eval_async.py` 生成预测，示例如下 (请在 `TCM-SKIN-Benchmark` 目录下运行，并替换占位符):
        ```bash
        python scripts/creat_openai_eval_async.py --input_file data/questions.json --output_file model_results/<你的模型名称>/predictions.json --model_name <OpenAI模型ID> --api_key <你的OpenAI_API_Key> --base_url <你的API_Base_URL> --temperature 0.1 --concurrency 10
        ```
        *(注意：上面的命令是一整行，为了可读性分成了多行显示。实际运行时可以写在一行，或者使用 `\` 换行符，但要确保 `\` 后面没有多余空格并紧跟换行)*
2.  **运行评估脚本**:
    *   评估单个模型: `python scripts/evaluate.py <你的模型预测文件.json> data/full_test_set.json --save_plots`
    *   评估多个模型: `python scripts/evaluate_multiple.py data/full_test_set.json <模型A预测.json> <模型B预测.json> ... --output_dir comparison_results`
3.  **查看结果**: 评估脚本会输出准确率、加权得分等指标，并可选择生成报告和图表。

## 引用

如果您的研究使用了本基准测试，请考虑引用我们的工作。

## 贡献

欢迎提出问题、报告错误或贡献代码。

## 许可证
