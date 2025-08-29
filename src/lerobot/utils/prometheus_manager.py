# prometheus_manager.py
import threading
from prometheus_client import start_http_server, Gauge

class PrometheusManager:
    _instance = None
    _lock = threading.Lock()
    _server_started = False
    _gauges = {}

    def __new__(cls, *args, **kwargs):
        # 使用双重检查锁定来确保线程安全的单例模式
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(PrometheusManager, cls).__new__(cls)
        return cls._instance

    def _initialize_gauges(self):
        """
        私有方法，用于初始化所有需要的指标。
        只有在第一次被请求时才会创建。
        """
        if 'joint_position' not in self._gauges:
            # 关键：增加了 'robot_name' 标签来区分 Leader 和 Follower
            self._gauges['joint_position'] = Gauge(
                'robot_joint_position_radians',
                'Current position of a robot joint in radians',
                ['robot_name', 'joint_name','joint_id']  # 标签列表
            )
        # 你可以在这里添加更多的全局指标，例如 'joint_velocity', 'motor_temperature' 等
        # if 'joint_velocity' not in self._gauges:
        #     self._gauges['joint_velocity'] = Gauge(...)

    def get_gauge(self, name: str) -> Gauge:
        """
        获取一个已命名的 Gauge 指标。如果不存在，则先初始化。
        """
        with self._lock:
            if not self._gauges:
                self._initialize_gauges()
            if name not in self._gauges:
                raise ValueError(f"Gauge '{name}' is not defined in PrometheusManager.")
            return self._gauges[name]

    def start_server(self, port: int):
        """
        启动 Prometheus HTTP 服务器。
        内部会检查以确保服务器只被启动一次。
        """
        if port is None:
            return
            
        with self._lock:
            if not self._server_started:
                try:
                    thread = threading.Thread(target=lambda: start_http_server(port), daemon=True)
                    thread.start()
                    self._server_started = True
                    print(f"Prometheus metrics server started globally on port {port}")
                except Exception as e:
                    # 如果端口已被占用，这里会抛出异常
                    print(f"Could not start Prometheus server on port {port}. It might already be running. Error: {e}")
                    # 即使启动失败，也标记为已尝试，避免重复尝试
                    self._server_started = True 


# 创建一个全局可用的单例实例
prometheus_manager = PrometheusManager()