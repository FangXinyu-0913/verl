
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


def execute_python_with_timeout(code_file, timeout=50, memory_limit_mb=1000):
    try:
        subprocess.run(["python3", code_file], timeout=timeout, check=True)
        return True
    except subprocess.TimeoutExpired:
        print(f"进程超时（{timeout}秒）终止")
        return False
    except subprocess.CalledProcessError:
        print("进程执行出错")
        return False