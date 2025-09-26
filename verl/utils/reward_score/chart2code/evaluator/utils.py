
import subprocess
import time
import os
import signal
import psutil

def execute_python_with_timeout_and_memory(code_file, timeout=100, memory_limit_mb=1000):
    """
    执行指定的 Python 脚本文件，并在超时或内存超限时终止。

    参数:
        code_file (str): 要执行的 Python 脚本文件路径。
        timeout (int): 允许的最大执行时间（秒）。
        memory_limit_mb (int): 允许的最大内存使用量（MB）。

    返回:
        bool: 如果脚本在限制内正常完成，返回 True；否则返回 False。
    """
    command = ["python3", code_file]

    try:
        # 启动子进程，创建新会话以便于终止整个进程组
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True
        )
        proc = psutil.Process(process.pid)
        start_time = time.time()

        while True:
            time.sleep(0.1)
            elapsed_time = time.time() - start_time

            # 检查超时
            if elapsed_time > timeout:
                print(f"进程超过了设定的超时时间 {timeout} 秒，正在终止。")
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                return False

            # 检查内存使用
            memory_usage = proc.memory_info().rss / (1024 * 1024)  # 转换为 MB
            if memory_usage > memory_limit_mb:
                print(f"进程内存使用超过了限制 {memory_limit_mb} MB（当前: {memory_usage:.2f} MB），正在终止。")
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                return False

            # 检查进程是否已结束
            if process.poll() is not None:
                break

        # 获取输出
        # stdout, stderr = process.communicate()
        # print("进程输出:")
        # print(stdout.decode())
        # if stderr:
        #     print("进程错误输出:")
        #     print(stderr.decode())

        return True

    except Exception as e:
        print(f"发生异常: {e}")
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        return False

    finally:
        # 确保进程被终止
        if process.poll() is None:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            print("进程被强制终止。")


def execute_python_with_timeout_old(code_file, timeout=100, memory_limit_mb=1000):
    try:
        subprocess.run(["python3", code_file], timeout=timeout, check=True)
        return True
    except subprocess.TimeoutExpired:
        print(f"进程超时（{timeout}秒）终止 {code_file}")
        return False
    except subprocess.CalledProcessError:
        print("进程执行出错")
        return False

import subprocess
import time
import psutil # 引入 psutil
from typing import Optional

import os, sys, time, subprocess, psutil, signal
from typing import Optional, Callable

def execute_python_with_timeout(
    code_file: str, 
    timeout: int = 300,                         # 兼容旧参数：仍作为硬超时
    memory_limit_mb: int = 1000,
    check_interval_sec: float = 1.0,            # 存活轮询
    memory_check_interval_sec: Optional[float] = 10.0,  # None/0 -> 关闭内存检查
    soft_timeout: Optional[int] = None,         # 若不传，默认 soft=timeout*0.8，hard=timeout
    max_soft_extensions: int = 2,               # 软延长最多次数
    cpu_progress_window_sec: float = 5.0,       # 近5秒是否有CPU进展
    cpu_progress_min_increase: float = 0.1,     # 至少增加0.1秒CPU时间才算“在推进”
    on_event: Optional[Callable[[str], None]] = None
) -> bool:
    """
    运行一个 Python 文件，并监控运行时间与内存。
    软/硬超时 + 分频内存检查 + 渐进终止。返回 True/False，不抛异常。
    """
    log = on_event or (lambda msg: None)

    if not os.path.isfile(code_file):
        log(f"[runner] 文件不存在: {code_file}")
        return False

    # 平台相关的进程组/新会话设置
    popen_kwargs = {}
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    else:
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    # 软/硬超时初始化
    hard_timeout = timeout
    if soft_timeout is None:
        soft_timeout = int(hard_timeout * 0.8) if hard_timeout else None
    now = time.time()
    soft_deadline = now + soft_timeout if soft_timeout else None
    hard_deadline = now + hard_timeout if hard_timeout else None
    soft_extensions_used = 0

    # 启动子进程（不捕获输出，避免管道阻塞）
    try:
        proc = subprocess.Popen(
            [sys.executable, code_file],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            close_fds=True,
            **popen_kwargs
        )
        p = psutil.Process(proc.pid)
    except Exception as e:
        log(f"[runner] 启动失败: {e}")
        return False

    # CPU 进度检测：记录最近一次观测
    last_cpu_check_t = time.time()
    last_cpu_times = 0.0
    try:
        ct = p.cpu_times()
        last_cpu_times = (ct.user + getattr(ct, "system", 0.0))
    except psutil.Error:
        pass

    # 内存检查节流
    next_mem_check_t = time.time()

    def terminate_tree(grace=2.0):
        """先温和终止，再强杀，尽量清理干净"""
        try:
            children = p.children(recursive=True)
        except psutil.Error:
            children = []

        try:
            if os.name == "posix":
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            else:
                for ch in children:
                    try: ch.terminate()
                    except Exception: pass
                if p.is_running():
                    try: p.terminate()
                    except Exception: pass
        except Exception:
            pass

        # 等宽限
        psutil.wait_procs(children, timeout=grace)
        try:
            proc.wait(timeout=0.1)
        except Exception:
            pass

        # 仍存活则强杀
        if proc.poll() is None:
            try:
                if os.name == "posix":
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                else:
                    for ch in p.children(recursive=True):
                        try: ch.kill()
                        except Exception: pass
                    if p.is_running():
                        try: p.kill()
                        except Exception: pass
            except Exception:
                pass

        # 回收
        try:
            proc.wait(timeout=1.0)
        except Exception:
            pass

    try:
        while proc.poll() is None:
            now = time.time()

            # 1) 硬超时判定
            if hard_deadline and now >= hard_deadline:
                log(f"[runner] 硬超时({hard_timeout}s)，终止: {code_file}")
                terminate_tree()
                return False

            # 2) 内存检查（分频/可关）
            if memory_check_interval_sec and now >= next_mem_check_t:
                next_mem_check_t = now + memory_check_interval_sec
                try:
                    rss = p.memory_info().rss
                    for ch in p.children(recursive=True):
                        rss += ch.memory_info().rss
                    if rss / (1024 * 1024) > memory_limit_mb:
                        log(f"[runner] 内存超限(>{memory_limit_mb}MB)，终止: {code_file}")
                        terminate_tree()
                        return False
                except psutil.NoSuchProcess:
                    break
                except psutil.Error:
                    pass

            # 3) 软超时：如果接近/达到软超时，检查是否“还有进展”
            if soft_deadline and now >= soft_deadline and soft_extensions_used < max_soft_extensions:
                # 看最近 cpu_progress_window_sec 内 CPU 时间是否增长
                progress_ok = False
                try:
                    ct = p.cpu_times()
                    cpu_now = (ct.user + getattr(ct, "system", 0.0))
                    # 只在窗口边界再算一次
                    if (now - last_cpu_check_t) >= cpu_progress_window_sec:
                        cpu_inc = cpu_now - last_cpu_times
                        progress_ok = (cpu_inc >= cpu_progress_min_increase)
                        last_cpu_times = cpu_now
                        last_cpu_check_t = now
                except psutil.Error:
                    pass

                if progress_ok:
                    soft_extensions_used += 1
                    # 把软截止顺延一段（这里顺延同 soft_timeout 的时长，必要时你可以改成固定值）
                    soft_deadline = now + soft_timeout
                    log(f"[runner] 软超时触发但仍在推进：顺延({soft_extensions_used}/{max_soft_extensions})")
                else:
                    # 没进展，按硬终止流程
                    log(f"[runner] 软超时且无明显进展，终止: {code_file}")
                    terminate_tree()
                    return False

            # 4) 等待下一轮
            #   取三者中的最小：存活轮询间隔、软/硬截止剩余时间（避免睡过头）
            sleep_candidates = [check_interval_sec]
            if soft_deadline:
                sleep_candidates.append(max(0.05, soft_deadline - now))
            if hard_deadline:
                sleep_candidates.append(max(0.05, hard_deadline - now))
            time.sleep(max(0.05, min(sleep_candidates)))

        # 进程已退出
        return proc.returncode == 0

    except Exception as e:
        log(f"[runner] 监控异常: {e}")
        terminate_tree()
        return False

# def execute_python_with_timeout(
#     code_file: str, 
#     timeout: int = 300, 
#     memory_limit_mb: int = 1000,
#     check_interval_sec: int = 1  # 每隔5秒检查一次，您可以按需调整
# ) -> bool:
#     """
#     执行一个 Python 文件，并同时监控其运行时间和内存使用。

#     Args:
#         code_file: 要执行的 Python 文件路径。
#         timeout: 总超时时间（秒）。
#         memory_limit_mb: 内存使用上限（MB）。
#         check_interval_sec: 监控检查的时间间隔（秒）。

#     Returns:
#         True 如果成功执行，False 如果超时、超内存或执行出错。
#     """
#     try:
#         # 1. 使用 Popen 非阻塞地启动子进程
#         # preexec_fn=os.setsid 在Linux/macOS下可以确保我们能杀死整个进程组，防止子进程产生孙子进程逃逸
#         import os
#         proc = subprocess.Popen(["python3", code_file], preexec_fn=os.setsid)
#         p_psutil = psutil.Process(proc.pid)
#     except (FileNotFoundError, psutil.NoSuchProcess):
#         print(f"进程启动失败或立即退出: {code_file}")
#         return False
        
#     end_time = time.time() + timeout

#     try:
#         # 2. 监控循环
#         while proc.poll() is None:  # proc.poll() is None 表示进程仍在运行
#             # 检查总运行时间是否超时
#             if time.time() > end_time:
#                 print(f"进程超时 ({timeout}秒)，将被终止 Old: {code_file}")
#                 # 杀死整个进程组
#                 os.killpg(os.getpgid(p_psutil.pid), 9) 
#                 return False

#             # 检查内存使用
#             try:
#                 # p_psutil.memory_info().rss 返回的是字节数
#                 # 我们需要遍历所有子进程来获取总内存
#                 mem_children = sum(child.memory_info().rss for child in p_psutil.children(recursive=True))
#                 total_mem_bytes = p_psutil.memory_info().rss + mem_children
#                 total_mem_mb = total_mem_bytes / (1024 * 1024)
                
#                 if total_mem_mb > memory_limit_mb:
#                     print(f"内存超限 (使用: {total_mem_mb:.2f}MB > 限制: {memory_limit_mb}MB)，将被终止: {code_file}")
#                     os.killpg(os.getpgid(p_psutil.pid), 9)
#                     return False
#             except psutil.NoSuchProcess:
#                 # 进程在检查间隙自己结束了，是正常情况
#                 break
            
#             # 等待指定间隔
#             time.sleep(check_interval_sec)

#         # 3. 循环结束，判断进程退出状态
#         if proc.returncode == 0:
#             # print(f"进程成功执行: {code_file}")
#             return True
#         else:
#             print(f"进程执行出错，退出码: {proc.returncode} 文件: {code_file}")
#             return False
            
#     except Exception as e:
#         # 捕获其他未知错误
#         print(f"监控过程中发生未知错误: {e}")
#         # 确保进程被杀死
#         if p_psutil.is_running():
#             os.killpg(os.getpgid(p_psutil.pid), 9)
#         return False