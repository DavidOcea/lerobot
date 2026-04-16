import logging
import os
import time
from functools import wraps

# 全局开关：通过环境变量 MONITOR_ENABLED=1 启用监控日志
# 默认情况下不输出（DEBUG级别，且logging默认级别为INFO）
MONITOR_ENABLED = os.environ.get("MONITOR_ENABLED", "0") == "1"

def monitor_performance(func):
    """
    一个监控函数性能的装饰器，会打印调用频率、间隔和时长。

    输出级别：logging.DEBUG
    正常运行时不输出，只有设置日志级别为DEBUG时才输出。

    快速启用方式：
    - 设置环境变量 MONITOR_ENABLED=1（强制启用）
    - 或设置 logging level 为 DEBUG

    性能优化：当监控关闭时，直接返回原函数，不添加任何开销。
    """
    # 性能优化：当监控关闭时，直接返回原函数，避免装饰器开销
    if not MONITOR_ENABLED:
        return func

    @wraps(func)
    def wrapper(*args, **kwargs):
        # --- 1. 初始化状态 (只在第一次调用时执行) ---
        if not hasattr(wrapper, 'call_count'):
            wrapper.call_count = 0
            wrapper.total_duration = 0.0
            wrapper.last_call_time = time.perf_counter()

        # --- 2. 计算本次调用的指标 ---
        current_time = time.perf_counter()
        interval = current_time - wrapper.last_call_time
        frequency = 1.0 / interval if interval > 0 else float('inf')

        # --- 3. 执行原函数并计时 ---
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        duration = end_time - start_time

        # --- 4. 更新累计状态 ---
        wrapper.call_count += 1
        wrapper.total_duration += duration
        wrapper.last_call_time = current_time

        # --- 5. 计算平均值 ---
        avg_duration = wrapper.total_duration / wrapper.call_count

        # --- 6. 输出报告（DEBUG级别）---
        logging.debug(
            f"--- Function '{func.__qualname__}' Monitor --- "
            f"Call #{wrapper.call_count} "
            f"Duration: {duration:.6f}s "
            f"Interval: {interval:.6f}s "
            f"Frequency: {frequency:.2f}Hz "
            f"Average Duration: {avg_duration:.6f}s"
        )

        return result

    return wrapper