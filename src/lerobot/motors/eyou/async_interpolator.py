import queue
import threading
import time
from typing import List, Dict, Any, Optional, Tuple

# 假设 HardwareInterface 已定义
from .hardware_interface import HardwareInterface

class AsyncInterpolator(HardwareInterface):
    """
    一个实现了 HardwareInterface 的装饰器类。
    它接收一个基础的 HardwareInterface 对象，并为其添加异步插值功能。
    如果配置中的 interpolation_n <= 1，则禁用插值功能，作为一个简单的直通包装器。
    """

    def __init__(self, hardware: HardwareInterface, config: Dict[str, Any]):
        """
        构造函数。
        :param hardware: 一个基础的、实现了 HardwareInterface 的硬件对象。
        :param config: 包含插值器控制参数的配置字典。
        """
        if not isinstance(hardware, HardwareInterface):
            raise TypeError("hardware object must implement HardwareInterface")
        
        self._base_hardware: HardwareInterface = hardware
        self._config = config
        
        interpolation_n = int(self._config.get("interpolation_n", 2))
        self._interpolation_enabled = interpolation_n > 1

        if self._interpolation_enabled:
            print(f"AsyncInterpolator: Interpolation is ENABLED (interpolation_n = {interpolation_n}).")
            self._command_queue = queue.Queue(maxsize=1)
            self._writer_thread: Optional[threading.Thread] = None
            self._stop_event = threading.Event()
            
            control_frequency = float(self._config.get("control_frequency", 30.0))
            self._writer_frequency = control_frequency * interpolation_n
            self._interp_duration = 1.0 / control_frequency
            
            self._interp_start_pos: List[float] = []
            self._interp_end_pos: List[float] = []
            self._interp_start_time: float = 0.0
        else:
            print("AsyncInterpolator: Interpolation is DISABLED (interpolation_n <= 1). Operating in pass-through mode.")
            self._command_queue = None
            self._writer_thread = None
            self._stop_event = None

    # --- HardwareInterface 方法实现 (大部分保持不变) ---

    def init(self, config: Dict[str, Any]) -> bool:
        return self._base_hardware.init(config)

    def activate(self) -> bool:
        print("AsyncInterpolator: Activating...")
        if not self._base_hardware.activate():
            print("AsyncInterpolator Error: Base hardware activation failed.")
            return False
        
        if self._interpolation_enabled:
            #  bootstrapping: 只需要在启动时读取一次硬件位置
            initial_pos = self._base_hardware.read()
            if not initial_pos or any(p is None for p in initial_pos):
                print("AsyncInterpolator Error: Failed to read valid initial positions.")
                self._base_hardware.deactivate()
                return False
                
            self._interp_start_pos = list(initial_pos)
            self._interp_end_pos = list(initial_pos)
            self._interp_start_time = time.monotonic()

            self._stop_event.clear()
            self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
            self._writer_thread.start()
        
        print("AsyncInterpolator: Activated successfully.")
        return True

    def deactivate(self):
        print("AsyncInterpolator: Deactivating...")
        if self._interpolation_enabled and self._writer_thread and self._writer_thread.is_alive():
            self._stop_event.set()
            if self._command_queue:
                try: self._command_queue.put_nowait([]) 
                except queue.Full: pass
            self._writer_thread.join(timeout=1.0)
        
        self._base_hardware.deactivate()
        print("AsyncInterpolator: Deactivated.")

    def read(self) -> List[Tuple[Optional[float], Optional[float]]]:
        # 移除了不必要的 print 语句以减少控制台输出
        return self._base_hardware.read()

    def write(self, commands_positions: List[float]):
        if self._interpolation_enabled:
            try:
                while not self._command_queue.empty():
                    self._command_queue.get_nowait()
                self._command_queue.put_nowait(commands_positions)
            except (queue.Full, AttributeError):
                pass
        else:
            self._base_hardware.write(commands_positions)
            
    def get_joint_count(self) -> int:
        return self._base_hardware.get_joint_count()

    # --- ✨ 内部写入线程 (核心修改区) ✨ ---
    def _writer_loop(self):
        """
        高频写入循环。
        这个循环不从硬件读取数据，以避免I/O延迟。
        它基于时间进行线性插值，并在接收到新命令时平滑地过渡到新的目标。
        """
        print("Writer thread started.")
        period = 1.0 / self._writer_frequency
        
        while not self._stop_event.is_set():
            loop_start_time = time.perf_counter()

            # 1. 首先计算当前时刻应该在的位置
            now = time.monotonic()
            time_since_start = now - self._interp_start_time
            alpha = min(1.0, max(0.0, time_since_start / self._interp_duration))
            
            # 使用列表推导式进行线性插值 (lerp)
            interpolated_positions = [
                s + alpha * (e - s) 
                for s, e in zip(self._interp_start_pos, self._interp_end_pos)
            ]

            # 2. 检查是否有新的目标命令
            try:
                new_target = self._command_queue.get_nowait()
                # 如果有新目标，则开始一个新的插值段
                # 新的起点是当前插值计算出的位置，以确保平滑过渡
                self._interp_start_pos = interpolated_positions.copy()
                self._interp_end_pos = new_target
                self._interp_start_time = time.monotonic() # 重置插值计时器
            except queue.Empty:
                # 没有新命令，继续当前的插值
                pass
            
            # 3. 将计算出的位置写入底层硬件
            self._base_hardware.write(interpolated_positions)

            # 4. 精确控制循环频率
            loop_end_time = time.perf_counter()
            sleep_time = period - (loop_end_time - loop_start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

        print("Writer thread stopped.")