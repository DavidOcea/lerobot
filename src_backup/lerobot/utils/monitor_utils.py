import time
from functools import wraps

def monitor_performance(func):
    """
    一个监控函数性能的装饰器，会打印调用频率、间隔和时长。
    """
    # 使用@wraps(func)可以保留原函数的元信息（如__name__, __doc__）
    @wraps(func)
    def wrapper(*args, **kwargs):
        # --- 1. 初始化状态 (只在第一次调用时执行) ---
        # 我们将状态信息直接附加到wrapper函数对象上，避免使用全局变量
        if not hasattr(wrapper, 'call_count'):
            wrapper.call_count = 0
            wrapper.total_duration = 0.0
            # 使用perf_counter()，因为它提供高精度且不受系统时间调整的影响
            wrapper.last_call_time = time.perf_counter()

        # --- 2. 计算本次调用的指标 ---
        current_time = time.perf_counter()
        
        # 调用间隔 = 当前时间 - 上次调用的时间
        interval = current_time - wrapper.last_call_time
        
        # 频率 = 1 / 间隔 (如果间隔为0则为无穷大)
        frequency = 1.0 / interval if interval > 0 else float('inf')

        # --- 3. 执行原函数并计时 ---
        start_time = time.perf_counter()
        result = func(*args, **kwargs) # 执行真正的函数
        end_time = time.perf_counter()
        duration = end_time - start_time

        # --- 4. 更新累计状态 ---
        wrapper.call_count += 1
        wrapper.total_duration += duration
        wrapper.last_call_time = current_time # 更新上次调用时间

        # --- 5. 计算平均值 ---
        avg_duration = wrapper.total_duration / wrapper.call_count

        # --- 6. 打印报告 ---
        print(f"--- Function '{func.__name__}' Monitor ---")
        print(f"  Call #{wrapper.call_count}")
        print(f"  Duration (本次时长): {duration:.6f} s")
        print(f"  Interval (调用间隔): {interval:.6f} s")
        print(f"  Frequency (瞬时频率): {frequency:.2f} Hz")
        print(f"  Average Duration (平均时长): {avg_duration:.6f} s")
        print("-" * (len(func.__name__) + 24))

        return result
    
    return wrapper