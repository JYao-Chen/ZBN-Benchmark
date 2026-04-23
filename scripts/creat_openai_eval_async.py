# -*- coding: utf-8 -*-
# @Time : 2025/4/18 上午10:27 
# @Author : Yao
# @File : creat_openai_eval_async.py

import json
import os
import time
import ast # 用于安全地解析字符串形式的列表/字典
import asyncio # Added asyncio
from tqdm.asyncio import tqdm # Use tqdm's async version
from openai import AsyncOpenAI, APITimeoutError, APIConnectionError, RateLimitError, APIStatusError # Use AsyncOpenAI
import concurrent.futures # Added concurrent.futures
import argparse # <<< Added argparse

# --- OpenAI Async Client 初始化 (已移除) ---
# 全局客户端已被移除，将在 run_openai_eval_async 函数内部根据参数初始化
# aclient = AsyncOpenAI(...)


def format_options(options):
    """将选项列表格式化为 A. xxx B. xxx 的形式"""
    # This is CPU-bound, stays synchronous
    return "\n".join([f"{chr(ord('A') + i)}. {option}" for i, option in enumerate(options)])

def format_prompt(question_data):
    """根据问题数据格式化Prompt，强调领域和输出格式"""
    # This is CPU-bound, stays synchronous
    q_type = question_data.get("question_type", "未知题型")
    question_text = question_data.get("question", "")
    options = question_data.get("options", [])

    prompt = f"请根据以下中医皮肤科赵炳南流派相关问题，给出最合适的答案。\n\n问题类型：{q_type}\n\n"

    if q_type == "单选题":
        prompt += f"问题描述：\n{question_text}\n\n"
        prompt += f"选项：\n{format_options(options)}\n\n"
        prompt += "请严格按照以下格式返回答案：\n"
        prompt += "请**仅**返回用**双引号**包裹的**唯一大写字母**，不要包含任何其他文字或标点符号。"
        prompt += "\n输出样例如下：\n"
        prompt += "\"A\""
    elif q_type == "多选题":
        prompt += f"问题描述：\n{question_text}\n\n"
        prompt += f"选项：\n{format_options(options)}\n\n"
        prompt += "请严格按照以下格式返回答案：\n"
        prompt += "请**仅**返回用**双引号**包裹的、**包含所有正确选项大写字母的列表字符串**，确保列表内的字母按字母顺序排列，不要包含任何其他文字。"
        prompt += "\n输出样例如下：\n"
        prompt += "\"[\"A\", \"C\"]\""
    elif q_type == "连续选择题":
        sub_questions = []
        if isinstance(question_text, list): sub_questions = question_text
        prompt += "子问题列表：\n"
        if isinstance(sub_questions, list) and isinstance(options, list) and len(sub_questions) == len(options):
            for i, sub_q in enumerate(sub_questions):
                sub_options = options[i]
                q_type_indicator = ""
                if sub_q.strip().startswith("（单选）"): q_type_indicator = " (单选)"
                elif sub_q.strip().startswith("（多选）"): q_type_indicator = " (多选)"
                prompt += f"\n子问题 {i+1}{q_type_indicator}:\n{sub_q}\n选项：\n{format_options(sub_options)}\n"
        else:
            prompt += f"问题描述：\n{question_text}\n\n选项结构复杂或非标准格式，请根据具体子问题回答。\n"
        prompt += "\n请严格按照以下格式返回所有子问题的答案：\n"
        prompt += "请**仅**返回用**双引号**包裹的、包含每个子问题答案的**列表字符串**，列表元素的格式必须遵循其子问题类型（单选为大写字母字符串，多选为按字母顺序排序的大写字母字符串列表），不要包含任何其他文字。"
        prompt += "\n输出样例如下：\n"
        prompt += "\"[\"B\", [\"A\", \"D\"], \"C\"]\""
    else:
        prompt += f"问题描述：\n{question_text}\n\n"
        if options: prompt += f"选项：\n{format_options(options)}\n\n"
        prompt += "请根据问题内容和类型给出答案，并**仅**返回用**双引号**包裹的答案本身（例如单选返回 \"A\"，多选返回 \"[\"A\", \"C\"]\"），不要包含额外文字。"
    return prompt

def parse_prediction(raw_prediction, question_type):
    """尝试解析模型返回的、期望被双引号包裹的原始预测字符串为目标格式"""
    # This is CPU-bound, stays synchronous
    prediction = raw_prediction.strip()

    # 1. 检查是否以双引号开头和结尾
    if not (prediction.startswith('"') and prediction.endswith('"')):
        # print(f"警告: 预测 '{raw_prediction}' 未按预期被双引号包裹。")
        return None # 格式不符，无法解析

    # 2. 提取引号内的内容
    content_inside_quotes = prediction[1:-1].strip()

    # 3. 根据题型解析内部内容
    if question_type == "单选题":
        # 期望内部是单个大写字母
        if len(content_inside_quotes) == 1 and 'A' <= content_inside_quotes <= 'Z':
            return content_inside_quotes
        # else:
            # print(f"警告: 单选题引号内内容 '{content_inside_quotes}' 无效。")
        return None
    
    elif question_type == "多选题" or question_type == "连续选择题":
        # 期望内部是列表字符串，如 '["A", "C"]' 或 '["B", ["A", "D"], "C"]'
        try:
            parsed_content = ast.literal_eval(content_inside_quotes)
            
            # 对多选题进行额外验证和排序
            if question_type == "多选题":
                if isinstance(parsed_content, list) and all(isinstance(item, str) and len(item) == 1 and 'A' <= item.upper() <= 'Z' for item in parsed_content):
                    return sorted([item.upper() for item in parsed_content])
                else:
                    # print(f"警告: 多选题引号内解析出的列表内容无效 '{parsed_content}'")
                    return None
            
            # 对连续选择题，仅验证最外层是列表 (更详细验证可选)
            elif question_type == "连续选择题":
                if isinstance(parsed_content, list):
                    return parsed_content
                else:
                    # print(f"警告: 连续选择题引号内解析出的内容不是列表 '{parsed_content}'")
                    return None
                    
        except (ValueError, SyntaxError, TypeError):
            # print(f"警告: 无法将引号内内容 '{content_inside_quotes}' 解析为 {question_type} 的预期格式。")
            return None
            
    # 对于未知题型，如果被引号包裹，我们尝试返回内部内容，但不做进一步解析
    # 或者，如果未知题型也应遵循特定格式，在此添加逻辑
    else: 
        # print(f"警告: 未知题型 '{question_type}'。返回引号内原始内容。")
        return content_inside_quotes # 或者返回 None，取决于如何处理未知题型

async def process_question_async(question_item, aclient: AsyncOpenAI, semaphore: asyncio.Semaphore, max_retries: int, model_name: str, temperature: float, max_tokens: int):
    """异步处理单个问题，包括API调用、解析和重试"""
    original_id = question_item.get("id", f"unknown_{question_item}") # Simplified ID for error case
    question_type = question_item.get("question_type", "未知题型")
    prompt_content = format_prompt(question_item)

    success = False
    final_result = None
    last_error = f"Error: Max retries ({max_retries}) reached"

    for attempt in range(max_retries):
        prediction_result = None
        raw_prediction_content = "Error: No response"
        try:
            # Limit concurrent API calls using semaphore
            async with semaphore:
                completion = await aclient.chat.completions.create(
                    model=model_name, # 使用传入的 model_name
                    messages=[{"role": "user", "content": prompt_content}],
                    max_tokens=max_tokens, # <<--- 使用传入的 max_tokens
                    temperature=temperature, # 使用传入的 temperature
                )
            raw_prediction_content = completion.choices[0].message.content
            prediction_result = parse_prediction(raw_prediction_content, question_type)

            if prediction_result is not None:
                final_result = prediction_result
                success = True
                break # Success, exit retry loop
            else:
                last_error = f"Error: Parsing failed (Attempt {attempt + 1}). Raw: '{raw_prediction_content[:50]}...'"
                # print(f"\n问题 {original_id}: {last_error}") # Optional: uncomment for detailed parsing errors

        except APITimeoutError:
            last_error = f"Error: Request Timeout (Attempt {attempt + 1})"
        except APIConnectionError as e:
            last_error = f"Error: Connection Failed (Attempt {attempt + 1}): {e}"
        except RateLimitError:
            last_error = f"Error: Rate Limit Exceeded (Attempt {attempt + 1})"
            wait_time = 5 + attempt * 5 # Exponential backoff example
            # print(f"\n问题 {original_id}: {last_error}. Waiting {wait_time}s...")
            await asyncio.sleep(wait_time) # Async sleep
        except APIStatusError as e:
            last_error = f"Error: API Status {e.status_code} (Attempt {attempt + 1}): {e.response}"
            if not (500 <= e.status_code < 600): # Don't retry client errors (4xx)
                 break
        except Exception as e:
            last_error = f"Error: Unknown (Attempt {attempt + 1}): {e}"
            break # Don't retry unknown errors

        # Wait before retrying if not successful
        if not success and attempt < max_retries - 1:
            await asyncio.sleep(1 + attempt) # Small backoff before retry

    if not success:
        print(f"\n问题 {original_id} 最终失败: {last_error}")
        final_result = last_error # Record the last known error

    return {"id": original_id, "prediction": final_result}

# -- 重构 sync_writer 为单次写入函数 --
def write_line_to_file(filepath: str, result: dict):
    """同步函数：将单个结果字典写入 JSONL 文件。"""
    try:
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
        return True # Indicate success
    except Exception as e:
        print(f"\n[File Writer Error] 写入文件 {filepath} 失败: {e} for result {result}")
        return False # Indicate failure

# -- 新增：批量写入函数 -- 
def write_batch_to_file(filepath: str, batch: list):
    """同步函数：将一批结果字典写入 JSONL 文件。"""
    try:
        with open(filepath, "a", encoding="utf-8") as f:
            for result in batch:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
        return len(batch) # 返回成功写入的数量
    except Exception as e:
        print(f"\n[File Writer Error] 批量写入文件 {filepath} 失败: {e}")
        return 0 # 返回 0 表示失败

async def run_openai_eval_async(input_file, output_prediction_file, api_key: str, base_url: str, model_name="gpt-3.5-turbo", temperature=0.0, max_tokens=20, max_retries=3, concurrency_limit=10, write_batch_size=50, batch_delay_seconds=1.0):
    """异步从输入文件读取问题，调用OpenAI API，使用单独线程批量保存结果，支持断点续传和重试"""

    # --- 在函数内部根据参数初始化客户端 ---
    try:
        aclient = AsyncOpenAI(
            api_key=api_key,        # <<< Use argument
            base_url=base_url,      # <<< Use argument
            timeout=60.0,           # 设置请求超时时间 (秒)
        )
        # 可选：添加一个简单的测试调用来验证凭据 (如果需要)
        # await aclient.models.list() 
        print(f"OpenAI 客户端已使用 API Key (尾部): ...{api_key[-4:] if api_key else 'None'} 和 Base URL: {base_url} 初始化。")
    except Exception as e:
        print(f"错误: 初始化 OpenAI 客户端失败: {e}")
        print("请检查您的 API Key 和 Base URL 是否正确以及网络连接。")
        return
    # ---------------------------------------

    try:
        with open(input_file, 'r', encoding='utf-8') as f_in:
            questions = json.load(f_in)
    except FileNotFoundError:
        print(f"错误：输入文件未找到 {input_file}")
        return
    except json.JSONDecodeError:
        print(f"错误：无法解析输入文件 {input_file}")
        return

    output_dir = os.path.dirname(output_prediction_file)
    if output_dir: os.makedirs(output_dir, exist_ok=True)

    predictions = [] # Holds successfully processed predictions loaded from previous runs
    processed_ids = set()
    intermediate_jsonl_path = output_prediction_file + ".jsonl"

    # --- 断点续传逻辑 (优先 .json, 其次 .jsonl) ---
    json_loaded_successfully = False
    if os.path.exists(output_prediction_file):
        try:
            with open(output_prediction_file, 'r', encoding='utf-8') as f_prev:
                loaded_data = json.load(f_prev)
                if isinstance(loaded_data, list):
                    # Process loaded JSON data
                    valid_predictions_loaded = []
                    for item in loaded_data:
                        if 'id' in item:
                            prediction_value = item.get('prediction')
                            is_successful = prediction_value is not None and not (
                                isinstance(prediction_value, str) and prediction_value.startswith("Error:")
                            )
                            if is_successful:
                                processed_ids.add(item['id'])
                                valid_predictions_loaded.append(item)
                    predictions = valid_predictions_loaded # Load successful predictions
                    print(f"成功从最终 JSON 文件加载了 {len(processed_ids)} 个先前成功处理的预测结果。")
                    json_loaded_successfully = True
                else:
                    print(f"警告: 现有预测文件 {output_prediction_file} 格式无效 (不是列表)。将尝试从 .jsonl 加载。")
        except (json.JSONDecodeError, Exception) as e:
             print(f"警告: 加载或解析现有 JSON 文件 {output_prediction_file} 时发生错误: {e}。将尝试从 .jsonl 加载。")
    
    if not json_loaded_successfully:
        print(f"最终 JSON 文件未找到或加载失败，尝试从中间 .jsonl 文件 ({intermediate_jsonl_path}) 加载状态...")
        if os.path.exists(intermediate_jsonl_path):
            try:
                # Use a dictionary to store the latest valid prediction for each ID
                latest_valid_predictions = {}
                processed_ids_jsonl = set()
                with open(intermediate_jsonl_path, "r", encoding="utf-8") as f_jsonl:
                    for line_num, line in enumerate(f_jsonl): # Add line number for debugging
                        try:
                            item = json.loads(line)
                            item_id = item.get('id')
                            if item_id:
                                prediction_value = item.get('prediction')
                                is_successful = prediction_value is not None and not (
                                    isinstance(prediction_value, str) and prediction_value.startswith("Error:")
                                )
                                if is_successful:
                                    # Store the latest successful prediction found for this ID
                                    latest_valid_predictions[item_id] = item
                                    processed_ids_jsonl.add(item['id'])
                                else:
                                    # If we encounter an error for an ID previously marked successful, keep the successful one
                                    if item_id not in processed_ids_jsonl:
                                         # If it wasn't successful before either, remove any potential prior error entry
                                         latest_valid_predictions.pop(item_id, None) 
                        except json.JSONDecodeError:
                            print(f"警告：无法解析 .jsonl 文件中的行 {line_num + 1}: {line.strip()}")
                # Use the data loaded from jsonl
                processed_ids = processed_ids_jsonl
                # Load the *latest* valid predictions found in the jsonl
                predictions = list(latest_valid_predictions.values()) 
                print(f"成功从 .jsonl 文件加载了 {len(processed_ids)} 个先前成功处理的预测 ID。")
            except Exception as e:
                print(f"警告: 从 .jsonl 文件加载状态时发生错误: {e}。将从头开始。")
                processed_ids = set()
                predictions = []
        else:
            print("未找到 .jsonl 文件，将从头开始处理所有问题。")
            processed_ids = set()
            predictions = []
    # --------------------------------------------------

    questions_to_process = [q for q in questions if q.get("id") not in processed_ids]
    total_questions_to_process = len(questions_to_process)
    if total_questions_to_process == 0:
        print("所有问题都已处理完毕。")
        # Optional: Check if final json exists and is complete, otherwise generate it from jsonl
        if not os.path.exists(output_prediction_file):
             print(f"最终 JSON 文件 {output_prediction_file} 未找到，尝试从 .jsonl 生成...")
             try:
                 final_predictions_list = []
                 intermediate_jsonl_path = output_prediction_file + ".jsonl"
                 if os.path.exists(intermediate_jsonl_path):
                     with open(intermediate_jsonl_path, "r", encoding="utf-8") as f_jsonl:
                         for line in f_jsonl:
                             try:
                                 final_predictions_list.append(json.loads(line))
                             except json.JSONDecodeError:
                                 print(f"警告：无法解析 .jsonl 文件中的行: {line.strip()}")
                     with open(output_prediction_file, "w", encoding="utf-8") as f_final:
                         json.dump(final_predictions_list, f_final, ensure_ascii=False, indent=2)
                     print(f"已成功从 {intermediate_jsonl_path} 生成最终 JSON 文件。")
                 else:
                     print(f"错误：中间 .jsonl 文件 {intermediate_jsonl_path} 也未找到。无法生成最终 JSON。")
             except Exception as e:
                 print(f"错误：从 .jsonl 生成最终 JSON 时出错: {e}")
        return

    start_time = time.time()
    print(f"开始处理 {total_questions_to_process} 个新问题 (模型: {model_name}, 温度: {temperature}, MaxTokens: {max_tokens}, 最大重试: {max_retries}, 并发数: {concurrency_limit})...")

    # Define path for intermediate JSON Lines file
    intermediate_jsonl_path = output_prediction_file + ".jsonl"

    # Initialize the .jsonl file by ensuring it exists (or clearing if needed?)
    # Let's just ensure the directory exists; append mode will create the file.
    # with open(intermediate_jsonl_path, "w", encoding="utf-8") as f: pass # Clear file if restarting?

    semaphore = asyncio.Semaphore(concurrency_limit)
    loop = asyncio.get_running_loop() # Get the current event loop
    # Create a ThreadPoolExecutor to run blocking I/O
    # Using None uses the default executor, which is usually a ThreadPoolExecutor
    executor = concurrent.futures.ThreadPoolExecutor() 

    # Keep track of results in memory *only for this run* if needed for immediate stats
    results_this_run = [] 
    # Keep track of writer futures to ensure they complete
    writer_futures = []
    # Batch for writing
    # write_batch = [] # We'll create batches for writing after API batch completes

    # --- Helper function to chunk data --- 
    def chunk_list(data, size):
        for i in range(0, len(data), size):
            yield data[i:i + size]

    # --- Process in batches with delay --- 
    api_batch_size = concurrency_limit # Process in chunks matching concurrency
    with tqdm(total=total_questions_to_process, desc="Processing Questions") as pbar:
        for i, batch_questions in enumerate(chunk_list(questions_to_process, api_batch_size)):
            pbar.set_description(f"Submitting Batch {i+1}/{ (total_questions_to_process + api_batch_size - 1) // api_batch_size } ({len(batch_questions)} questions)")
            tasks_in_batch = [
                asyncio.create_task(process_question_async(q_item, aclient, semaphore, max_retries, model_name, temperature, max_tokens))
                for q_item in batch_questions
            ]
            
            # Wait for the current batch of API calls to complete
            batch_results = await asyncio.gather(*tasks_in_batch)
            
            # Process results from the completed batch and schedule writes
            pbar.set_description(f"Processing Batch {i+1} results...")
            batch_to_write_to_file = []
            for result in batch_results:
                 if result:
                     results_this_run.append(result) # Track overall results for this run
                     batch_to_write_to_file.append(result)
                     pbar.update(1) # Update progress bar for each processed result
            
            # Schedule writing for the completed batch
            if batch_to_write_to_file:
                 writer_future = loop.run_in_executor(
                     executor,
                     write_batch_to_file, # Use the batch write function
                     intermediate_jsonl_path,
                     batch_to_write_to_file
                 )
                 writer_futures.append(writer_future)
                 
            # --- Introduce delay AFTER processing a batch (if not the last batch) --- 
            total_batches = (total_questions_to_process + api_batch_size - 1) // api_batch_size
            if i < total_batches - 1:
                pbar.set_description(f"Batch {i+1} complete. Pausing {batch_delay_seconds}s...")
                await asyncio.sleep(batch_delay_seconds) # Pause for the specified duration
            else:
                pbar.set_description("Final batch complete.")
    # ------------------------------------

    # --- Wait for all background writing tasks to complete --- 
    print(f"\n所有 API 任务完成。等待 {len(writer_futures)} 个文件写入操作完成...")
    if writer_futures:
         # Wait for all scheduled writes to finish
         done, pending = await asyncio.wait(writer_futures, return_when=asyncio.ALL_COMPLETED)
         print(f"文件写入完成。成功: {sum(1 for f in done if f.result() is True)}, 失败: {sum(1 for f in done if f.result() is False)}")
         if pending: print(f"警告：仍有 {len(pending)} 个写入任务待处理？这不应该发生。")
    else:
         print("没有需要等待的文件写入操作。")
         
    # Shutdown the executor gracefully
    executor.shutdown(wait=True)
    print("线程池已关闭。")
    # ---------------------------------------------------------

    end_time = time.time()
    elapsed_time = end_time - start_time

    # --- Final Step: Convert JSONL to JSON List --- 
    print(f"\n所有任务处理完成。正在将结果从 {intermediate_jsonl_path} 写入最终 JSON 文件 {output_prediction_file}...")
    # Use a dictionary to store the latest entry found for each ID in the jsonl
    final_predictions_map = {}
    final_processed_ids_from_jsonl = set()
    try:
        if os.path.exists(intermediate_jsonl_path):
            with open(intermediate_jsonl_path, "r", encoding="utf-8") as f_jsonl:
                for line_num, line in enumerate(f_jsonl): # Add line number
                    try:
                        item = json.loads(line)
                        item_id = item.get("id")
                        if item_id:
                            # Always store the latest encountered entry for an ID
                            final_predictions_map[item_id] = item
                    except json.JSONDecodeError:
                        print(f"警告：最终转换时无法解析 .jsonl 文件中的行 {line_num + 1}: {line.strip()}")
            
            # Now, build the final list and stats from the unique latest entries
            final_predictions_list = list(final_predictions_map.values())
            for item_id, item in final_predictions_map.items():
                 prediction_value = item.get("prediction")
                 is_successful = prediction_value is not None and not (
                    isinstance(prediction_value, str) and prediction_value.startswith("Error:")
                 )
                 if is_successful:
                     final_processed_ids_from_jsonl.add(item_id)
            
            # Write the final JSON list file (now guaranteed unique by ID)
            with open(output_prediction_file, "w", encoding="utf-8") as f_final:
                json.dump(final_predictions_list, f_final, ensure_ascii=False, indent=2)
            print(f"最终 JSON 文件已成功保存到：{output_prediction_file}")
            # Optionally remove the intermediate file
            # os.remove(intermediate_jsonl_path)
            # print(f"已删除中间 .jsonl 文件: {intermediate_jsonl_path}")

        else: 
             print(f"错误：无法找到中间 .jsonl 文件 {intermediate_jsonl_path} 进行最终转换。")
             # Fallback: try to save whatever results were collected in memory this run? 
             # This might be incomplete if the script was interrupted before finishing all tasks.
             # with open(output_prediction_file, "w", encoding="utf-8") as f_final:
             #     json.dump(results_this_run, f_final, ensure_ascii=False, indent=2)
             # print(f"警告: 仅保存了本次运行内存中的 {len(results_this_run)} 个结果。")
             final_predictions_list = results_this_run # Use in-memory results for stats
             final_processed_ids_from_jsonl = {p['id'] for p in results_this_run if 'id' in p and p.get('prediction') is not None and not (isinstance(p.get('prediction'), str) and p.get('prediction').startswith('Error:'))}


    except Exception as e:
        print(f"错误：在最终转换 JSONL 到 JSON 时发生错误: {e}")
        final_predictions_list = results_this_run # Use in-memory results for stats if conversion fails
        final_processed_ids_from_jsonl = {p['id'] for p in results_this_run if 'id' in p and p.get('prediction') is not None and not (isinstance(p.get('prediction'), str) and p.get('prediction').startswith('Error:'))}

    # --- Recalculate stats based on the final list derived from jsonl --- 
    total_in_final_output = len(final_predictions_list)
    successful_in_final_output = len(final_processed_ids_from_jsonl)
    failed_in_final_output = total_in_final_output - successful_in_final_output

    print(f"\n处理总结。总问题数: {len(questions)}, 最终输出条目数: {total_in_final_output}, 成功: {successful_in_final_output}, 失败: {failed_in_final_output}。总耗时: {elapsed_time:.2f} 秒")
    # print(f"最终预测结果已保存到：{output_prediction_file}") # Message already printed above


if __name__ == "__main__":
    # --- 使用 argparse 解析命令行参数 ---
    parser = argparse.ArgumentParser(description="通过 OpenAI API 异步评估 QA 数据集")
    parser.add_argument("--input_file", default='../data/questions.json', help="输入问题 JSON 文件路径")
    parser.add_argument("--output_file", required=True, help="输出预测结果 JSON 文件路径")
    parser.add_argument("--model_name", default="gpt-3.5-turbo", help="要使用的 OpenAI 模型名称")
    parser.add_argument("--api_key", required=True, help="OpenAI API Key")
    parser.add_argument("--base_url", default="https://api.openai.com/v1", help="OpenAI API Base URL")
    parser.add_argument("--temperature", type=float, default=0.6, help="模型采样温度")
    parser.add_argument("--max_tokens", type=int, default=30, help="模型生成最大 token 数")
    parser.add_argument("--max_retries", type=int, default=3, help="API 调用失败最大重试次数")
    parser.add_argument("--concurrency", type=int, default=10, help="并发 API 请求数")
    parser.add_argument("--write_batch_size", type=int, default=50, help="每批次写入文件的结果数量")
    parser.add_argument("--batch_delay", type=float, default=1.0, help="API 请求批次之间的延迟秒数")

    args = parser.parse_args()

    # --- 修正相对路径计算 ---
    # script_dir = os.path.dirname(__file__)
    # # 如果 input_file 是相对路径，则相对于脚本目录解析
    # if not os.path.isabs(args.input_file):
    #     input_json_path = os.path.join(script_dir, args.input_file)
    # else:
    #     input_json_path = args.input_file

    # --- 新的路径处理：直接使用用户提供的路径 (相对于CWD或绝对路径) ---
    input_json_path = args.input_file
    # -------------------------------------------------------------------

    # 输出路径直接使用用户提供的
    output_pred_path = args.output_file
    # -------------------------

    print(f"输入文件: {input_json_path}")
    print(f"输出文件: {output_pred_path}")
    print(f"模型: {args.model_name}, 温度: {args.temperature}, Max Tokens: {args.max_tokens}")
    print(f"API Base URL: {args.base_url}")
    print(f"并发数: {args.concurrency}, 重试次数: {args.max_retries}, 批次延迟: {args.batch_delay}s")

    # Run the main async function with arguments from CLI
    asyncio.run(run_openai_eval_async(
        input_json_path,
        output_pred_path,
        api_key=args.api_key,            # <<< Pass argument
        base_url=args.base_url,          # <<< Pass argument
        model_name=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        concurrency_limit=args.concurrency,
        write_batch_size=args.write_batch_size,
        batch_delay_seconds=args.batch_delay
    ))
