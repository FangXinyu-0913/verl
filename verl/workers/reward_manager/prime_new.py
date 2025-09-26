# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Any, Callable, Optional

import psutil
import torch
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


async def single_compute_score_with_retry(evaluation_func, completion, reference, task, task_extra_info, executor, 
                                       initial_timeout=30.0, max_timeout=120.0, retry_count=2):
    """
    渐进式超时重试机制：先用短超时，失败后逐步增加超时时间
    """
    loop = asyncio.get_running_loop()
    
    timeouts = [initial_timeout, max_timeout // 2, max_timeout][:retry_count + 1]
    
    for attempt, timeout in enumerate(timeouts):
        try:
            start_time = time.time()
            future = loop.run_in_executor(executor, partial(evaluation_func, task, completion, reference, task_extra_info))
            result = await asyncio.wait_for(future, timeout=timeout)
            
            # 记录成功的执行时间
            execution_time = time.time() - start_time
            if execution_time > timeout * 0.8:  # 如果接近超时，记录警告
                print(f"[Warning] Task nearly timed out: {execution_time:.2f}s (limit: {timeout}s)")
            
            return result
            
        except asyncio.TimeoutError:
            if attempt < len(timeouts) - 1:
                print(f"[Retry] Task timeout after {timeout}s, retrying with longer timeout...")
                continue
            else:
                print(f"[Timeout] Task failed after all retries (max timeout: {max_timeout}s)")
                return -1
                
        except Exception as e:
            print(f"[Error] Task failed on attempt {attempt + 1}: {e}, completion: {completion[:80]}")
            if attempt < len(timeouts) - 1:
                continue
            else:
                return None
    
    return None


async def single_compute_score(evaluation_func, completion, reference, task, task_extra_info, executor, timeout=60.0):
    """保持原有接口兼容性的包装函数"""
    return await single_compute_score_with_retry(
        evaluation_func, completion, reference, task, task_extra_info, executor, 
        initial_timeout=timeout, max_timeout=min(timeout * 2, 120.0), retry_count=1
    )


def get_optimal_process_count(num_tasks, max_processes=64):
    """
    根据任务数和系统资源动态调整进程数
    """
    cpu_count = psutil.cpu_count(logical=True)
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    
    # 基于CPU核心数的建议
    cpu_based = min(cpu_count * 2, max_processes)
    
    # 基于内存的建议(假设每个进程需要约500MB)  
    memory_based = max(1, int(available_memory_gb * 2))
    
    # 基于任务数的建议
    task_based = min(num_tasks, max_processes)
    
    # 取最小值确保系统稳定
    optimal = min(cpu_based, memory_based, task_based)
    
    return max(1, optimal)


async def parallel_compute_score_async(
    evaluation_func, completions, references, tasks, extra_info=None, num_processes=None, 
    initial_timeout=30.0, max_timeout=120.0, batch_size=None
):
    """
    改进的并行reward计算函数，包含：
    - 自适应进程数调整
    - 渐进式超时机制  
    - 批量处理优化
    - 完善的进程清理
    """
    if extra_info is None:
        extra_info = [None] * len(tasks)
    
    # 动态调整进程数
    if num_processes is None:
        num_processes = get_optimal_process_count(len(tasks))
        print(f"[Info] Auto-adjusted process count to: {num_processes}")
    
    # 分批处理大量任务
    if batch_size is None:
        batch_size = max(100, len(tasks) // 4)  # 默认分成4批或最少100个一批
    
    total_scores = []
    total_failed = 0
    overall_start_time = time.time()
    
    # 分批处理
    for batch_idx in range(0, len(tasks), batch_size):
        batch_end = min(batch_idx + batch_size, len(tasks))
        batch_completions = completions[batch_idx:batch_end]
        batch_references = references[batch_idx:batch_end] 
        batch_tasks = tasks[batch_idx:batch_end]
        batch_extra_info = extra_info[batch_idx:batch_end]
        
        print(f"[Batch {batch_idx//batch_size + 1}] Processing tasks {batch_idx}-{batch_end-1} ({len(batch_tasks)} tasks)")
        batch_start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            try:
                # Create tasks for current batch using improved timeout mechanism
                tasks_async = [
                    single_compute_score_with_retry(
                        evaluation_func, c, r, t, ei, executor, 
                        initial_timeout=initial_timeout, 
                        max_timeout=max_timeout,
                        retry_count=1
                    )
                    for c, r, t, ei in zip(batch_completions, batch_references, batch_tasks, batch_extra_info, strict=True)
                ]
                results = await asyncio.gather(*tasks_async, return_exceptions=True)
                
            except Exception as e:
                print(f"[Exception] Batch {batch_idx//batch_size + 1} failed: {e}")
                # 为失败的整个批次设置默认分数
                results = [0.0] * len(batch_tasks)
                
            finally:
                # 确保进程正确清理
                terminated_count = 0
                try:
                    for pid, proc in executor._processes.items():
                        try:
                            p = psutil.Process(pid)
                            if p.is_running():
                                p.terminate()
                                try:
                                    p.wait(timeout=3)
                                except psutil.TimeoutExpired:
                                    p.kill()
                                    p.wait(timeout=2)
                                terminated_count += 1
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                        except Exception as e:
                            print(f"[Warning] Failed to terminate process {pid}: {e}")
                    
                    if terminated_count > 0:
                        print(f"[Cleanup] {terminated_count} subprocess(es) terminated for batch.")
                        
                except Exception as e:
                    print(f"[Warning] Process cleanup failed: {e}")

        # Process batch results
        batch_scores = []
        batch_failed = 0
        
        for result, completion, reference, task in zip(results, batch_completions, batch_references, batch_tasks, strict=True):
            if isinstance(result, Exception) or result is None or result == -1:
                batch_scores.append(0.0)
                batch_failed += 1
            elif isinstance(result, int | float | bool):
                batch_scores.append(float(result))
            else:
                batch_scores.append(float(result[0]))
        
        total_scores.extend(batch_scores)
        total_failed += batch_failed
        
        batch_time = time.time() - batch_start_time
        print(f"[Batch {batch_idx//batch_size + 1}] Completed in {batch_time:.2f}s, failed: {batch_failed}/{len(batch_tasks)}")

    overall_time = time.time() - overall_start_time
    success_rate = (len(total_scores) - total_failed) / len(total_scores) * 100 if total_scores else 0
    
    print(f"[Summary] Total: {len(total_scores)} tasks, Failed: {total_failed}, Success rate: {success_rate:.1f}%, Total time: {overall_time:.2f}s")
    
    return total_scores


def run_reward_scoring(evaluation_func, completions, references, tasks, extra_info=None, 
                      num_processes=None, initial_timeout=30.0, max_timeout=120.0, batch_size=None):
    """
    改进的reward评分函数，支持：
    - 自适应进程数（默认自动调整）
    - 可配置的超时参数
    - 可配置的批处理大小
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            parallel_compute_score_async(
                evaluation_func, completions, references, tasks, extra_info, 
                num_processes, initial_timeout, max_timeout, batch_size
            )
        )
    finally:
        loop.close()


@register("prime")
class PrimeRewardManager(AbstractRewardManager):
    """
    改进的PRIME Reward Manager，包含：
    - 自适应超时和进程管理  
    - 批处理优化
    - 详细的性能监控
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_examine: int,
        compute_score: Optional[Callable] = None,
        reward_fn_key: str = "data_source",
        # 新增配置参数
        num_processes: Optional[int] = None,  # None表示自动调整
        initial_timeout: float = 30.0,        # 初始超时时间(秒)  
        max_timeout: float = 120.0,           # 最大超时时间(秒)
        batch_size: Optional[int] = None,     # 批处理大小，None表示自动调整
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        
        # 新增的性能配置
        self.num_processes = num_processes
        self.initial_timeout = initial_timeout
        self.max_timeout = max_timeout
        self.batch_size = batch_size
        
        # 性能统计
        self.total_computations = 0
        self.total_compute_time = 0.0

    def verify(self, data):
        """
        改进的验证方法，使用新的超时和批处理机制
        """
        start_time = time.time()
        
        # batched scoring
        prompt_ids = data.batch["prompts"]
        response_ids = data.batch["responses"]
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        ground_truth = [data_item.non_tensor_batch["reward_model"]["ground_truth"] for data_item in data]
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        extra_info = data.non_tensor_batch.get("extra_info", None)

        assert len(sequences_str) == len(ground_truth) == len(data_sources)
        
        print(f"[Verify] Starting reward computation for {len(sequences_str)} sequences...")
        
        try:
            scores = run_reward_scoring(
                self.compute_score,
                completions=sequences_str,
                references=ground_truth,
                tasks=data_sources,
                extra_info=extra_info,
                num_processes=self.num_processes,      # 使用配置的进程数
                initial_timeout=self.initial_timeout,  # 使用配置的初始超时
                max_timeout=self.max_timeout,          # 使用配置的最大超时
                batch_size=self.batch_size,            # 使用配置的批大小
            )
        except asyncio.TimeoutError:
            print("[Timeout] Global reward scoring timed out. Setting all as 0.")
            scores = [0.0 for _ in range(len(sequences_str))]
        except Exception as e:
            print(f"[Error] Unexpected error during scoring. Setting all as 0. {e}")
            scores = [0.0 for _ in range(len(sequences_str))]
        
        # 更新性能统计
        computation_time = time.time() - start_time
        self.total_computations += len(sequences_str)
        self.total_compute_time += computation_time
        
        avg_time_per_task = computation_time / len(sequences_str) if sequences_str else 0
        print(f"[Verify] Completed in {computation_time:.2f}s, avg {avg_time_per_task:.3f}s/task")
        
        # 定期打印累计统计
        if self.total_computations % 1000 == 0:  # 每1000个任务打印一次
            overall_avg = self.total_compute_time / self.total_computations
            print(f"[Stats] Total computations: {self.total_computations}, "
                  f"Overall avg time: {overall_avg:.3f}s/task")
        
        data.batch["acc"] = torch.tensor(scores, dtype=torch.float32, device=prompt_ids.device)
        return scores

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        already_print_data_sources = {}

        # batched scoring
        prompt_ids = data.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]

        response_ids = data.batch["responses"]
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=-1)
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        data_sources = data.non_tensor_batch["data_source"]

        scores = self.verify(data)

        for i in range(len(data)):
            data_source = data_sources[i]
            reward_tensor[i, valid_response_length[i].item() - 1] = scores[i]

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        if return_dict:
            return {"reward_tensor": reward_tensor}
        else:
            return reward_tensor
