"""
Seer (仙工) AGV TCP Controller.

This module implements direct TCP communication with Seer AGV systems
without requiring ROS2. Based on the protocol specification from
tcp_bridge_node.py in ros2_ws.

Protocol Format (16-byte header + JSON payload):
┌─────────────────────────────────────────────────────────┐
│ sync (1B) │ version (1B) │ seq (2B LE) │ data_len (4B BE)│
│ api_type (2B BE) │ reserved (6B)                         │
├─────────────────────────────────────────────────────────┤
│ JSON payload (data_len bytes)                           │
└─────────────────────────────────────────────────────────┘

Port allocation (基于官方API文档 + 扫描确认):
- 19204: 状态查询 (API 1000-1800)
- 19205: 控制 (API 2000-2026: 停止开环运动/重定位/开环运动/加载地图等)
- 19206: 导航 (API 3001-3115: 路径导航/暂停/继续/取消等)
- 19207: 配置 (API 4005-4803)

Reference: /root/workspace/dc_dir/ros2_ws/src/sm_test_tcp_bridge/sm_test_tcp_bridge/tcp_bridge_node.py
"""

import json
import logging
import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AGVPosition:
    """AGV位置信息 (SLAM坐标系)."""
    x: float  # 米
    y: float  # 米
    theta: float  # 弧度，航向角


@dataclass
class AGVStatus:
    """AGV综合状态信息."""
    battery: int  # 电量百分比 (0-100)
    status_code: int  # 状态码: 0=空闲, 1=执行任务, 2=充电, 3=异常, 4=暂停
    current_station: str  # 当前站点ID
    position: AGVPosition
    is_moving: bool
    error_code: int  # 错误码，0表示正常
    error_message: str  # 错误描述
    vx: float = 0.0   # 线速度 m/s (正=前进)
    vy: float = 0.0   # 线速度 m/s (正=左移)
    vtheta: float = 0.0  # 角速度 rad/s


class SeerAGVController:
    """仙工AGV TCP控制器.

    通过TCP协议直接与AGV通信，无需ROS2依赖。
    支持状态查询、导航控制、任务管理等功能。

    Usage:
        controller = SeerAGVController(host="192.168.1.100")
        if controller.connect():
            # 查询状态
            status = controller.get_status()
            print(f"电量: {status.battery}%")

            # 导航到站点
            controller.move_to_station("station_B")

            # 等待到达
            controller.wait_for_arrival("station_B", timeout=60)

            controller.disconnect()
    """

    # ========== API类型码定义 (基于 2026-04-23 全面扫描结果) ==========

    # 状态查询类 API (端口19204)
    API_STATUS_QUERY = 0x03E8      # 1000 - 综合状态查询 (系统版本/地图名/vehicle_id, 38字段)
    API_ODOMETER_QUERY = 0x03EA    # 1002 - 里程/电压查询 (odo, controller_voltage, time) — 注意: 无battery_level!
    API_TASK_STATUS_QUERY = 0x03EC # 1004 - 位置+站点查询 ✅ (x, y, angle, current_station, loc_state)
    API_OBSTACLE_QUERY = 0x03EE    # 1006 - 障碍物检测 ✅ (blocked, nearest_obstacles, block_x/y)
    API_BRAKE_QUERY = 0x03F0       # 1008 - 制动状态 (brake)
    API_PATH_QUERY = 0x03F2        # 1010 - 路径数据 (path)
    API_EMC_QUERY = 0x03F4         # 1012 - EMC急停状态 ✅ (emergency, soft_emc, driver_emc, electric)
    API_IMU_QUERY = 0x03F6         # 1014 - IMU数据 (acc_x/y/z, pitch/roll/yaw, qw/qx/qy/qz)
    API_ULTRASONIC_QUERY = 0x03F8  # 1016 - 超声波传感器 (ultrasonic_nodes)
    API_ENCODER_QUERY = 0x03FA     # 1018 - 编码器 (encoder, motor_encoder)
    API_NAV_STATUS_QUERY = 0x03FC  # 1020 - 导航详情 ✅ (task_status, running_status, target_id/dist)
    API_LOADMAP_QUERY = 0x03FE     # 1022 - 地图加载状态 (loadmap_status)
    API_TRACKING_QUERY = 0x0400    # 1024 - 跟踪状态 (target_x/y, tracking_status)
    API_TASKLIST_QUERY = 0x0402    # 1026 - 任务列表状态 (tasklist_status)
    API_MOTOR_QUERY = 0x0410       # 1040 - 电机详情 ✅ (motor_info: speed, position, current, emc)
    API_ERRORS_QUERY = 0x041A      # 1050 - 系统日志 (errors, warnings, notices, fatals)
    API_CLIENT_QUERY = 0x0424      # 1060 - 连接客户端 (ip, port, locked)
    API_AGGREGATE_QUERY = 0x044C   # 1100 - 超级聚合查询 ✅✅ 包含vx, vy, w速度字段!
    API_BATTERY_QUERY = 0x044E     # 1102 - 电量详情 ✅ (battery_level, charging, voltage, controller_temp)
    API_TASK_PKG_QUERY = 0x0456    # 1110 - 任务进度包 (task_status_package: percentage, distance)
    API_STATION_QUERY = 0x0515    # 1301 - 查询地图站点信息 (返回所有站点id/x/y/r)

    # 控制类 API (端口19205) — 基于官方文档确认
    API_EMERGENCY_STOP = 0x07D0   # 2000 - 停止开环运动 (robot_control_stop_req)
    API_CLEAR_STOP = 0x07D1       # 2001 - (扫描: ret_code=0, 未在官方文档)
    API_RELOCATE = 0x07D2         # 2002 - 重定位 (robot_control_reloc_req, 需要: x, y, angle)
    API_CONFIRM_LOC = 0x07D3      # 2003 - 确认定位正确 (robot_control_comfirmloc_req)
    API_CANCEL_RELOCATE = 0x07D4  # 2004 - 取消重定位 (robot_control_cancelreloc_req)
    API_OPEN_LOOP_MOVE = 0x07DA   # 2010 - 开环运动 (robot_control_motion_req, 会取消导航任务!)
    API_LOAD_MAP = 0x07E6         # 2022 - 切换载入的地图 (robot_control_loadmap_req, 需要: map_name)

    # 导航类 API (端口19206) — 基于官方文档确认
    # 之前扫描错误: 用0x07D0-0x07FF范围扫19206导致全部"error api type"
    # 正确API范围是3000+(0x0BB9+)
    API_PAUSE_NAV = 0x0BB9        # 3001 - 暂停当前导航 (robot_task_pause_req)
    API_RESUME_NAV = 0x0BBA       # 3002 - 继续当前导航 (robot_task_resume_req)
    API_CANCEL_NAV = 0x0BBB       # 3003 - 取消当前导航 (robot_task_cancel_req)
    API_NAVIGATE_STATION = 0x0BEB # 3051 - 路径导航 (robot_task_gotarget_req, 需要: id, source_id)
    API_GET_PATH = 0x0BED         # 3053 - 获取路径导航的路径 (不执行导航, 只返回路径规划)
    API_TRANSLATE = 0x0BEF        # 3055 - 平动 (robot_task_translate_req, 需要: dist, vx/vy)
    API_TURN = 0x0BF0             # 3056 - 转动 (robot_task_turn_req, 需要: angle, vw)
    API_CIRCULAR = 0x0BF2         # 3058 - 圆弧运动 (robot_task_circular_req, 需要: rot_radius/rot_degree/rot_speed)
    API_PATH_ENABLE = 0x0BF3      # 3059 - 启用和禁用线路 (robot_task_path_req)
    API_NAVIGATE_PATH = 0x0BFA    # 3066 - 指定路径导航 (robot_task_gotargetlist_req, 多站点序列)

    # 任务管理类 API (端口19206)
    API_EXECUTE_TASKLIST = 0x0C22 # 3106 - 执行预存任务链 (robot_tasklist_name_req, 需要: name)
    API_TASKLIST_STATUS = 0x0C1D  # 3101 - 查询机器人任务链 (robot_tasklist_status_req)
    API_TASKLIST_LIST = 0x0C2B    # 3115 - 查询所有任务链 (robot_tasklist_list_req, 无参数)

    # 配置类 API (端口19207) — 基于官方文档确认
    API_LOCK_CONTROL = 0x0FA5     # 4005 - 抢占控制权 (robot_config_lock_req, 需要: nick_name)
    API_UNLOCK_CONTROL = 0x0FA6   # 4006 - 释放控制权 (robot_config_unlock_req, 无参数)
    API_CLEAR_ALL_ERRORS = 0x0FA9 # 4009 - 清除所有报错 (robot_config_clearallerrors_req, 无参数)

    # ========== 端口分配 (基于官方文档) ==========
    PORT_STATUS = 19204     # 状态查询
    PORT_CONTROL = 19205    # 控制 (停止运动/重定位/开环运动)
    PORT_NAVIGATION = 19206 # 导航 (路径导航/暂停/继续/取消)
    PORT_CONFIG = 19207     # 配置 (抢占控制权/释放控制权/清除报错)

    # ========== 状态码映射 ==========
    STATUS_IDLE = 0         # 空闲
    STATUS_EXECUTING = 1    # 执行任务中
    STATUS_CHARGING = 2     # 充电中
    STATUS_ERROR = 3        # 异常
    STATUS_PAUSED = 4       # 暂停

    def __init__(
        self,
        host: str,
        port: int = 19204,
        connection_timeout: float = 5.0,
        read_timeout: float = 2.0,
        auto_reconnect: bool = True,
    ):
        """初始化AGV控制器.

        Args:
            host: AGV IP地址
            port: 默认端口 (用于状态查询)
            connection_timeout: TCP连接超时时间
            read_timeout: 读取响应超时时间
            auto_reconnect: 是否自动重连
        """
        self.host = host
        self.default_port = port
        self.connection_timeout = connection_timeout
        self.read_timeout = read_timeout
        self.auto_reconnect = auto_reconnect

        # Socket连接池 (多端口)
        self._sockets: dict[int, socket.socket] = {}
        self._socket_locks: dict[int, threading.Lock] = {}  # Per-port thread safety

        # 序列号 (自增)
        self._seq_num = 0

        # 状态缓存
        self._last_status: Optional[AGVStatus] = None
        self._last_position: Optional[AGVPosition] = None
        self._last_update_time: float = 0.0
        self._status_cache_ttl: float = 0.5  # 状态缓存有效期

        # 导航状态
        self._current_navigation_target: Optional[str] = None
        self._navigation_start_time: float = 0.0

        # 站点地图缓存 (可选)
        self._station_map: dict[str, AGVPosition] = {}

    def connect(self) -> bool:
        """建立TCP连接.

        连接所有必要的端口，包括状态查询、导航控制等。

        Returns:
            True if connection successful, False otherwise.
        """
        try:
            # 需要连接的端口列表 (19204状态 + 19205控制 + 19206导航 + 19207配置)
            ports_to_connect = [
                self.PORT_STATUS,
                self.PORT_CONTROL,
                self.PORT_NAVIGATION,
                self.PORT_CONFIG,
            ]

            connected_ports = []
            for port in ports_to_connect:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(self.connection_timeout)
                    sock.connect((self.host, port))
                    sock.settimeout(self.read_timeout)
                    self._sockets[port] = sock
                    self._socket_locks[port] = threading.Lock()
                    connected_ports.append(port)
                    logger.info(f"Connected to AGV at {self.host}:{port}")
                except Exception as e:
                    logger.warning(f"Failed to connect to port {port}: {e}")
                    # 继续尝试其他端口

            if len(connected_ports) >= 2:
                # 至少需要状态和导航端口
                logger.info(f"AGV connection established on ports: {connected_ports}")

                # 抢占控制权，确保能下发导航指令
                if self.PORT_CONFIG in self._sockets:
                    self.lock_control()

                return True
            else:
                logger.error("Failed to establish minimum required connections")
                self.disconnect()
                return False

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.disconnect()
            return False

    def disconnect(self):
        """断开所有TCP连接, 释放控制权."""
        # 释放控制权
        if self.PORT_CONFIG in self._sockets:
            try:
                self.unlock_control()
            except Exception:
                pass

        for port, sock in self._sockets.items():
            try:
                sock.close()
                logger.debug(f"Closed socket for port {port}")
            except Exception as e:
                logger.warning(f"Error closing socket {port}: {e}")

        self._sockets.clear()
        logger.info("Disconnected from AGV")

    def is_connected(self) -> bool:
        """检查是否已连接."""
        return len(self._sockets) >= 2

    def reconnect(self) -> bool:
        """重新连接."""
        self.disconnect()
        return self.connect()

    # ========== 协议层实现 ==========

    def _get_next_seq(self) -> int:
        """获取下一个序列号."""
        self._seq_num = (self._seq_num + 1) % 65536
        return self._seq_num

    def _build_packet(self, api_type: int, data: dict = None) -> bytes:
        """构建请求包.

        协议格式:
        - sync: 0x5A (1 byte)
        - version: 0x01 (1 byte)
        - seq: uint16 little-endian (2 bytes)
        - data_len: uint32 big-endian (4 bytes)
        - api_type: uint16 big-endian (2 bytes)
        - reserved: 6 bytes (zeros)
        - payload: JSON data

        Args:
            api_type: API类型码
            data: JSON数据字典

        Returns:
            完整的请求包bytes
        """
        seq = self._get_next_seq()

        # JSON payload
        json_str = json.dumps(data or {}, ensure_ascii=False)
        json_bytes = json_str.encode('utf-8')
        data_len = len(json_bytes)

        # Header (16 bytes)
        # 注意：sync/version/seq是little-endian，data_len/api_type是big-endian
        # 这与tcp_bridge_node.py的实现一致
        header = struct.pack(
            '<BBHI',  # little-endian: sync(1B), version(1B), seq(2B), data_len(4B)
            0x5A,     # sync byte
            0x01,     # version
            seq,      # sequence number
            data_len  # data length (注意：这里用LE，但实际协议可能需要BE)
        )

        # 补充: data_len和api_type使用big-endian
        # 重新构建header，参考tcp_bridge_node.py
        header = bytes([
            0x5A,  # sync
            0x01,  # version
        ])
        # seq: uint16 LE
        header += struct.pack('<H', seq)
        # data_len: uint32 BE (注意这里!)
        header += struct.pack('>I', data_len)
        # api_type: uint16 BE
        header += struct.pack('>H', api_type)
        # reserved: 6 bytes
        header += b'\x00' * 6

        return header + json_bytes

    def _parse_response_header(self, header_bytes: bytes) -> tuple[int, int, int, int]:
        """解析响应header.

        Returns:
            (sync, seq, data_len, api_type)
        """
        if len(header_bytes) != 16:
            raise ValueError(f"Header length mismatch: {len(header_bytes)}")

        sync = header_bytes[0]
        version = header_bytes[1]
        seq = struct.unpack('<H', header_bytes[2:4])[0]
        data_len = struct.unpack('>I', header_bytes[4:8])[0]
        api_type = struct.unpack('>H', header_bytes[8:10])[0]

        if sync != 0x5A:
            raise ValueError(f"Invalid sync byte: {sync:#x}")

        return seq, data_len, api_type

    def _send_request(
        self,
        port: int,
        api_type: int,
        data: dict = None,
        retry_count: int = 2,
    ) -> dict:
        """发送请求并接收响应.

        Args:
            port: 目标端口
            api_type: API类型码
            data: JSON数据
            retry_count: 重试次数

        Returns:
            解析后的响应JSON

        Raises:
            ConnectionError: 连接失败
            TimeoutError: 超时
        """
        if port not in self._sockets:
            if self.auto_reconnect and self.reconnect():
                if port not in self._sockets:
                    raise ConnectionError(f"Not connected to port {port}")
            else:
                raise ConnectionError(f"Not connected to port {port}")

        sock = self._sockets[port]
        lock = self._socket_locks.get(port)
        if lock is None:
            lock = threading.Lock()
            self._socket_locks[port] = lock

        # Serialize socket access per port to prevent request/response
        # interleaving when multiple threads share the same AGV connection
        # (e.g. main control loop + monitoring dashboard background poller).
        with lock:
            for attempt in range(retry_count + 1):
                try:
                    packet = self._build_packet(api_type, data)
                    logger.debug(f"Sending packet to port {port}: api={api_type:#x}, len={len(packet)}")

                    sock.sendall(packet)
                    header_bytes = self._recv_exact(sock, 16)
                    seq, data_len, api_type_resp = self._parse_response_header(header_bytes)

                    if data_len > 0:
                        payload_bytes = self._recv_exact(sock, data_len)
                        payload = json.loads(payload_bytes.decode('utf-8'))
                    else:
                        payload = {}

                    logger.debug(f"Response: seq={seq}, api={api_type_resp:#x}, data={payload}")
                    return payload

                except socket.timeout:
                    logger.warning(f"Timeout on port {port}, attempt {attempt + 1}")
                    if attempt < retry_count:
                        continue
                    raise TimeoutError(f"Timeout waiting for response on port {port}")

                except (ConnectionError, socket.error) as e:
                    logger.warning(f"Connection error on port {port}: {e}")
                    if attempt < retry_count and self.auto_reconnect:
                        self.reconnect()
                        sock = self._sockets.get(port)
                        if sock:
                            continue
                    raise ConnectionError(f"Failed to communicate on port {port}: {e}")

            raise ConnectionError(f"Failed after {retry_count + 1} attempts")

    def _recv_exact(self, sock: socket.socket, n: int) -> bytes:
        """精确接收n字节.

        Args:
            sock: Socket对象
            n: 需要接收的字节数

        Returns:
            接收到的bytes

        Raises:
            ConnectionError: 连接关闭或数据不完整
        """
        data = b''
        while len(data) < n:
            chunk = sock.recv(min(n - len(data), 4096))
            if not chunk:
                raise ConnectionError(f"Connection closed, received {len(data)}/{n} bytes")
            data += chunk
        return data

    # ========== 状态查询API ==========

    def get_battery(self) -> int:
        """获取电量百分比.

        使用 API_BATTERY_QUERY (0x044E) 获取电量数据。
        0x044E 返回 battery_level (0.0-1.0比例), charging, controller_voltage 等。
        注意: 0x03EA (API_ODOMETER_QUERY) 不包含 battery_level，只有里程和电压。

        Returns:
            电量百分比 (0-100)
        """
        try:
            response = self._send_request(
                self.PORT_STATUS,
                self.API_BATTERY_QUERY,
                {}
            )
            # battery_level 是 0.0-1.0 的比例值，需要转换为百分比
            battery_level = response.get('battery_level', 0.0)
            return int(battery_level * 100)
        except Exception as e:
            logger.error(f"Failed to get battery: {e}")
            return 0

    def get_position(self) -> AGVPosition:
        """获取当前位置坐标.

        使用 API_TASK_STATUS_QUERY (0x03EC) 获取位置，因为该 API 返回包含 x, y, angle 的数据。

        Returns:
            AGVPosition with x, y, theta
        """
        try:
            response = self._send_request(
                self.PORT_STATUS,
                self.API_TASK_STATUS_QUERY,  # 使用任务状态查询API获取位置
                {}
            )

            # 解析位置数据 - 任务状态查询返回的格式
            # {'x': -0.8844, 'y': -0.5086, 'angle': 1.2464, 'loc_state': 1, ...}
            x = float(response.get('x', 0.0))
            y = float(response.get('y', 0.0))
            theta = float(response.get('angle', 0.0))

            self._last_position = AGVPosition(x=x, y=y, theta=theta)
            return self._last_position

        except Exception as e:
            logger.error(f"Failed to get position: {e}")
            return AGVPosition(x=0.0, y=0.0, theta=0.0)

    def get_current_station(self) -> str:
        """获取当前站点ID.

        从任务状态查询 API (0x03EC) 获取，该 API 返回 current_station 字段。

        Returns:
            站点ID字符串
        """
        try:
            response = self._send_request(
                self.PORT_STATUS,
                self.API_TASK_STATUS_QUERY,  # 使用任务状态查询API获取站点
                {}
            )

            # current_station 直接在响应中
            station_id = response.get('current_station', '')

            return str(station_id)

        except Exception as e:
            logger.error(f"Failed to get station: {e}")
            return ''

    def get_velocity(self) -> tuple[float, float, float]:
        """获取当前速度.

        使用 API_AGGREGATE_QUERY (0x044C) 获取速度字段 vx, vy, w。
        该API是超级聚合查询，包含位置、速度、电量等所有数据。

        Returns:
            (vx, vy, vtheta) 单位: m/s, m/s, rad/s
        """
        try:
            response = self._send_request(
                self.PORT_STATUS,
                self.API_AGGREGATE_QUERY,
                {}
            )

            vx = float(response.get('vx', 0.0))
            vy = float(response.get('vy', 0.0))
            vtheta = float(response.get('w', 0.0))

            return (vx, vy, vtheta)

        except Exception as e:
            logger.error(f"Failed to get velocity: {e}")
            return (0.0, 0.0, 0.0)

    def get_emc_status(self) -> dict:
        """获取急停/EMC状态.

        API_EMC_QUERY (0x03F4) 返回 EMC 状态。

        Returns:
            {'emergency': bool, 'soft_emc': bool, 'driver_emc': bool}
        """
        try:
            response = self._send_request(
                self.PORT_STATUS,
                self.API_EMC_QUERY,
                {}
            )

            return {
                'emergency': response.get('emergency', False),
                'soft_emc': response.get('soft_emc', False),
                'driver_emc': response.get('driver_emc', False),
                'electric': response.get('electric', False),
            }

        except Exception as e:
            logger.error(f"Failed to get EMC status: {e}")
            return {'emergency': False, 'soft_emc': False, 'driver_emc': False}

    def get_task_status(self) -> dict:
        """获取当前任务状态.

        使用 API_TASK_STATUS_QUERY (0x03EC)，返回位置和站点信息。

        Returns:
            任务状态字典
        """
        try:
            response = self._send_request(
                self.PORT_STATUS,
                self.API_TASK_STATUS_QUERY,
                {}
            )
            return response

        except Exception as e:
            logger.error(f"Failed to get task status: {e}")
            return {}

    def get_obstacle_status(self) -> dict:
        """获取障碍物检测状态.

        使用 API_OBSTACLE_QUERY (0x03EE) 获取障碍物信息。

        Returns:
            {'blocked': bool, 'slowed': bool, 'nearest_obstacles': list}
        """
        try:
            response = self._send_request(
                self.PORT_STATUS,
                self.API_OBSTACLE_QUERY,
                {}
            )
            return {
                'blocked': response.get('blocked', False),
                'slowed': response.get('slowed', False),
                'nearest_obstacles': response.get('nearest_obstacles', []),
                'block_x': response.get('block_x', 0.0),
                'block_y': response.get('block_y', 0.0),
            }

        except Exception as e:
            logger.error(f"Failed to get obstacle status: {e}")
            return {'blocked': False, 'slowed': False, 'nearest_obstacles': []}

    def get_navigation_detail(self) -> dict:
        """获取导航详细状态.

        使用 API_NAV_STATUS_QUERY (0x03FC) 获取当前导航任务的详细信息，
        包括 task_status, running_status, target_id, target_dist 等。

        Returns:
            导航详情字典
        """
        try:
            response = self._send_request(
                self.PORT_STATUS,
                self.API_NAV_STATUS_QUERY,
                {}
            )
            return {
                'task_status': response.get('task_status', 0),
                'running_status': response.get('running_status', 0),
                'target_id': response.get('target_id', ''),
                'target_label': response.get('target_label', ''),
                'target_dist': response.get('target_dist', 0.0),
                'target_point': response.get('target_point', []),
                'unfinished_path': response.get('unfinished_path', []),
            }

        except Exception as e:
            logger.error(f"Failed to get navigation detail: {e}")
            return {}

    def get_task_progress(self) -> dict:
        """获取任务进度信息.

        使用 API_TASK_PKG_QUERY (0x0456) 获取任务进度包，
        包含 percentage, distance, closest_target 等。

        Returns:
            任务进度字典
        """
        try:
            response = self._send_request(
                self.PORT_STATUS,
                self.API_TASK_PKG_QUERY,
                {}
            )
            pkg = response.get('task_status_package', {})
            return {
                'percentage': pkg.get('percentage', 0),
                'distance': pkg.get('distance', -1),
                'closest_target': pkg.get('closest_target', ''),
                'target_name': pkg.get('target_name', ''),
                'source_name': pkg.get('source_name', ''),
            }

        except Exception as e:
            logger.error(f"Failed to get task progress: {e}")
            return {}

    def get_status(self, use_cache: bool = True) -> AGVStatus:
        """获取综合状态信息.

        Args:
            use_cache: 是否使用缓存 (避免频繁查询)

        Returns:
            AGVStatus对象
        """
        # 检查缓存
        if use_cache and self._last_status:
            if time.time() - self._last_update_time < self._status_cache_ttl:
                return self._last_status

        try:
            # 使用超级聚合查询 (0x044C) 一次获取所有数据
            # 比单独调用 get_battery() + get_position() + get_velocity() 等效率更高
            response = self._send_request(
                self.PORT_STATUS,
                self.API_AGGREGATE_QUERY,
                {}
            )

            # 位置
            x = float(response.get('x', 0.0))
            y = float(response.get('y', 0.0))
            theta = float(response.get('angle', 0.0))
            position = AGVPosition(x=x, y=y, theta=theta)
            self._last_position = position

            # 速度
            vx = float(response.get('vx', 0.0))
            vy = float(response.get('vy', 0.0))
            vtheta = float(response.get('w', 0.0))

            # 电量 (battery_level 是 0.0-1.0 比例)
            battery_level = response.get('battery_level', 0.0)
            battery = int(battery_level * 100)

            # 站点
            station = str(response.get('current_station', ''))

            # 状态码
            running_status = response.get('running_status', 0)

            # EMC状态
            emergency = response.get('emergency', False)

            # 判断是否在移动 - 速度非零 或 状态码=执行中
            is_moving = (abs(vx) > 0.01 or abs(vy) > 0.01 or abs(vtheta) > 0.01 or
                        running_status == self.STATUS_EXECUTING)

            self._last_status = AGVStatus(
                battery=battery,
                status_code=running_status,
                current_station=station,
                position=position,
                is_moving=is_moving,
                error_code=0 if not emergency else 1,
                error_message='' if not emergency else 'EMC active',
                vx=vx,
                vy=vy,
                vtheta=vtheta,
            )
            self._last_update_time = time.time()

            return self._last_status

        except Exception as e:
            logger.error(f"Failed to get status: {e}")
            return AGVStatus(
                battery=0,
                status_code=self.STATUS_ERROR,
                current_station='',
                position=AGVPosition(0.0, 0.0, 0.0),
                is_moving=False,
                error_code=-1,
                error_message=str(e),
            )

    # ========== 导航控制API ==========

    def move_to_station(self, station_id: str, source_id: str = "SELF_POSITION") -> bool:
        """导航到指定站点.

        使用官方API: robot_task_gotarget_req (3051), 端口19206.
        官方文档参数: id(目标站点) + source_id(起始站点, "SELF_POSITION"表示当前位置)

        Args:
            station_id: 目标站点ID (如 "LM1", "LM8")
            source_id: 起始站点ID, 默认"SELF_POSITION"(从当前位置出发)

        Returns:
            True if navigation started successfully
        """
        try:
            response = self._send_request(
                self.PORT_NAVIGATION,
                self.API_NAVIGATE_STATION,
                {"id": station_id, "source_id": source_id}
            )

            # 检查响应 - AGV 返回 ret_code 字段
            ret_code = response.get('ret_code', -1)
            success = ret_code == 0

            if success:
                self._current_navigation_target = station_id
                self._navigation_start_time = time.time()
                logger.info(f"Navigation started: target={station_id}")
                return True
            else:
                error_msg = response.get('err_msg', 'Unknown error')
                logger.error(f"Navigation failed: {error_msg}")
                return False

        except Exception as e:
            logger.error(f"Failed to start navigation: {e}")
            return False

    def move_to_position(self, x: float, y: float, theta: float = 0.0) -> bool:
        """导航到指定坐标 (freeGo模式).

        使用官方API: robot_task_gotarget_req (3051), 端口19206.
        freeGo模式通过发送坐标而非站点名来实现导航。
        JSON格式: {"id": "SELF_POSITION", "freeGo": {"x": ..., "y": ..., "theta": ...}}

        注意: freeGo仅支持双轮差速底盘模型。目标坐标为世界坐标系(地图坐标)。

        Args:
            x: 目标x坐标 (米, 世界坐标系)
            y: 目标y坐标 (米, 世界坐标系)
            theta: 目标航向角 (弧度), 默认0.0

        Returns:
            True if navigation started successfully
        """
        try:
            data = {
                "id": "SELF_POSITION",
                "freeGo": {
                    "x": x,
                    "y": y,
                    "theta": theta,
                }
            }

            response = self._send_request(
                self.PORT_NAVIGATION,
                self.API_NAVIGATE_STATION,
                data
            )

            ret_code = response.get('ret_code', -1)
            if ret_code == 0:
                self._current_navigation_target = f"freeGo({x:.2f},{y:.2f},{theta:.2f})"
                self._navigation_start_time = time.time()
                logger.info(f"freeGo navigation started: target=({x:.3f}, {y:.3f}, {theta:.3f})")
                return True
            else:
                error_msg = response.get('err_msg', 'Unknown error')
                logger.error(f"freeGo navigation failed: {error_msg}")
                return False

        except Exception as e:
            logger.error(f"Failed to start freeGo navigation: {e}")
            return False

    def cancel_navigation(self) -> bool:
        """取消当前导航任务.

        使用官方API: robot_task_cancel_req (3003), 端口19206.

        Returns:
            True if cancelled successfully
        """
        try:
            response = self._send_request(
                self.PORT_NAVIGATION,
                self.API_CANCEL_NAV,
                {}
            )

            # AGV 返回 ret_code 字段
            ret_code = response.get('ret_code', -1)
            success = ret_code == 0

            if success:
                self._current_navigation_target = None
                logger.info("Navigation cancelled")
                return True
            else:
                error_msg = response.get('err_msg', 'Unknown error')
                logger.warning(f"Failed to cancel navigation: {error_msg}")
                return False

        except Exception as e:
            logger.error(f"Failed to cancel navigation: {e}")
            return False

    # ========== 相对移动API ==========

    def translate(
        self,
        dist: float,
        vx: float | None = None,
        vy: float | None = None,
        mode: int = 0,
    ) -> bool:
        """平动 — 以固定速度直线移动固定距离.

        使用官方API: robot_task_translate_req (3055/0x0BEF), 端口19206.
        AGV移动完dist距离后自动停止。

        坐标系为机器人坐标系（AGV自身前后左右）:
        - vx: 正=前进, 负=后退 (m/s)
        - vy: 正=左移, 负=右移 (m/s)
        - 如果vx和vy都有值，速度会合成

        注意: 3055(平动)和3056(转动)不能同时进行。
        下发平动指令会取消当前正在执行的导航任务。

        Args:
            dist: 移动距离 (米, 绝对值)
            vx: 机器人坐标系X方向速度 (m/s, 正=前, 负=后)。缺省=0
            vy: 机器人坐标系Y方向速度 (m/s, 正=左, 负=右)。缺省=0
            mode: 0=里程模式(不需要定位精准, 但误差随距离增大),
                  1=定位模式(需要定位稳定, 精度更高)

        Returns:
            True if translate started successfully
        """
        try:
            data: dict = {"dist": dist, "mode": mode}
            if vx is not None:
                data["vx"] = vx
            if vy is not None:
                data["vy"] = vy

            response = self._send_request(
                self.PORT_NAVIGATION,
                self.API_TRANSLATE,
                data,
            )

            ret_code = response.get('ret_code', -1)
            if ret_code == 0:
                self._current_navigation_target = f"translate({dist}m)"
                self._navigation_start_time = time.time()
                direction = ""
                if vx is not None:
                    direction += f"vx={vx:.2f}(前/后) "
                if vy is not None:
                    direction += f"vy={vy:.2f}(左/右) "
                logger.info(f"Translate started: dist={dist}m {direction}mode={mode}")
                return True
            else:
                error_msg = response.get('err_msg', 'Unknown error')
                logger.error(f"Translate failed: {error_msg}")
                return False

        except Exception as e:
            logger.error(f"Failed to start translate: {e}")
            return False

    def turn(
        self,
        angle: float,
        vw: float,
        mode: int = 0,
    ) -> bool:
        """转动 — 以固定角速度旋转固定角度.

        使用官方API: robot_task_turn_req (3056/0x0BF0), 端口19206.
        AGV旋转完angle角度后自动停止。

        Args:
            angle: 旋转角度 (弧度, 绝对值, 可以大于2π)
            vw: 旋转角速度 (rad/s, 正=逆时针, 负=顺时针)
            mode: 0=里程模式, 1=定位模式

        Returns:
            True if turn started successfully
        """
        try:
            data = {"angle": angle, "vw": vw, "mode": mode}

            response = self._send_request(
                self.PORT_NAVIGATION,
                self.API_TURN,
                data,
            )

            ret_code = response.get('ret_code', -1)
            if ret_code == 0:
                self._current_navigation_target = f"turn({angle:.2f}rad)"
                self._navigation_start_time = time.time()
                logger.info(f"Turn started: angle={angle:.2f}rad vw={vw:.2f}rad/s mode={mode}")
                return True
            else:
                error_msg = response.get('err_msg', 'Unknown error')
                logger.error(f"Turn failed: {error_msg}")
                return False

        except Exception as e:
            logger.error(f"Failed to start turn: {e}")
            return False

    def wait_for_translate_complete(
        self,
        timeout: float = 30.0,
        poll_interval: float = 0.2,
    ) -> bool:
        """等待平动完成 — AGV移动完距离后变为空闲状态.

        Args:
            timeout: 最大等待时间 (秒)
            poll_interval: 状态轮询间隔 (秒)

        Returns:
            True if translate completed, False if timeout or error
        """
        start_time = time.time()
        logger.info(f"Waiting for translate to complete (timeout={timeout}s)")

        # Give a brief window for the task to start executing
        time.sleep(1.0)

        while time.time() - start_time < timeout:
            status = self.get_status(use_cache=False)

            # Translate completed: AGV returns to IDLE
            if status.status_code == self.STATUS_IDLE and not status.is_moving:
                elapsed = time.time() - start_time
                logger.info(f"Translate completed in {elapsed:.1f}s")
                self._current_navigation_target = None
                return True

            # Error during translate
            if status.error_code != 0:
                logger.error(f"AGV error during translate: {status.error_code} - {status.error_message}")
                self._current_navigation_target = None
                return False

            time.sleep(poll_interval)

        logger.warning(f"Timeout waiting for translate to complete")
        self._current_navigation_target = None
        return False

    def wait_for_turn_complete(
        self,
        timeout: float = 30.0,
        poll_interval: float = 0.2,
    ) -> bool:
        """等待转动完成 — AGV旋转完角度后变为空闲状态.

        Args:
            timeout: 最大等待时间 (秒)
            poll_interval: 状态轮询间隔 (秒)

        Returns:
            True if turn completed, False if timeout or error
        """
        start_time = time.time()
        logger.info(f"Waiting for turn to complete (timeout={timeout}s)")

        time.sleep(1.0)

        while time.time() - start_time < timeout:
            status = self.get_status(use_cache=False)

            if status.status_code == self.STATUS_IDLE and not status.is_moving:
                elapsed = time.time() - start_time
                logger.info(f"Turn completed in {elapsed:.1f}s")
                self._current_navigation_target = None
                return True

            if status.error_code != 0:
                logger.error(f"AGV error during turn: {status.error_code} - {status.error_message}")
                self._current_navigation_target = None
                return False

            time.sleep(poll_interval)

        logger.warning(f"Timeout waiting for turn to complete")
        self._current_navigation_target = None
        return False

    # ========== 控制API ==========

    def stop(self) -> bool:
        """急停 - 立即停止AGV.

        使用官方API: robot_control_stop_req (2000/0x07D0), 端口19205.
        如果急停失败，尝试取消导航任务(3003)作为备选。

        Returns:
            True if stop successful
        """
        try:
            # 先尝试停止开环运动 (官方API 2000)
            response = self._send_request(
                self.PORT_CONTROL,
                self.API_EMERGENCY_STOP,
                {}
            )

            ret_code = response.get('ret_code', -1)
            success = ret_code == 0

            if success:
                self._current_navigation_target = None
                logger.warning("AGV stop command executed (stop open-loop motion)")
                return True

            # 如果停止失败，尝试取消当前导航任务
            logger.warning(f"Stop ret_code={ret_code}, trying cancel_navigation")
            cancel_response = self._send_request(
                self.PORT_NAVIGATION,
                self.API_CANCEL_NAV,
                {}
            )
            cancel_ret = cancel_response.get('ret_code', -1)
            if cancel_ret == 0:
                self._current_navigation_target = None
                logger.warning("AGV stopped via cancel_navigation")
                return True

            error_msg = response.get('err_msg', 'Unknown error')
            logger.error(f"Emergency stop and cancel_navigation both failed: {error_msg}")
            return False

        except Exception as e:
            logger.error(f"Failed to stop AGV: {e}")
            return False

    def pause(self) -> bool:
        """暂停当前导航任务.

        使用官方API: robot_task_pause_req (3001/0x0BB9), 端口19206.

        Returns:
            True if pause successful
        """
        try:
            response = self._send_request(
                self.PORT_NAVIGATION,
                self.API_PAUSE_NAV,
                {}
            )

            ret_code = response.get('ret_code', -1)
            success = ret_code == 0

            if success:
                logger.info("AGV paused")
                return True
            else:
                error_msg = response.get('err_msg', 'Unknown error')
                logger.warning(f"Failed to pause AGV: {error_msg}")
                return False

        except Exception as e:
            logger.error(f"Failed to pause: {e}")
            return False

    def resume(self) -> bool:
        """继续执行暂停的导航任务.

        使用官方API: robot_task_resume_req (3002/0x0BBA), 端口19206.

        Returns:
            True if resume successful
        """
        try:
            response = self._send_request(
                self.PORT_NAVIGATION,
                self.API_RESUME_NAV,
                {}
            )

            ret_code = response.get('ret_code', -1)
            success = ret_code == 0

            if success:
                logger.info("AGV resumed")
                return True
            else:
                error_msg = response.get('err_msg', 'Unknown error')
                logger.warning(f"Failed to resume AGV: {error_msg}")
                return False

        except Exception as e:
            logger.error(f"Failed to resume: {e}")
            return False

    def cancel_task(self, task_id: str = "") -> bool:
        """取消当前导航任务.

        使用官方API: robot_task_cancel_req (3003/0x0BBB), 端口19206.

        Args:
            task_id: 任务ID (未使用, 官方API为空{})

        Returns:
            True if cancelled successfully
        """
        try:
            response = self._send_request(
                self.PORT_NAVIGATION,
                self.API_CANCEL_NAV,
                {}
            )

            ret_code = response.get('ret_code', -1)
            success = ret_code == 0

            if success:
                self._current_navigation_target = None
                logger.info("AGV task cancelled")
                return True
            else:
                error_msg = response.get('err_msg', 'Unknown error')
                logger.warning(f"Failed to cancel task: {error_msg}")
                return False

        except Exception as e:
            logger.error(f"Failed to cancel task: {e}")
            return False

    def set_speed(self, speed: float, task_id: str = "") -> bool:
        """设置AGV速度.

        注意: 官方文档中没有19205端口上的专用速度设置API。
        0x07D6在扫描中需要'id'参数，但官方文档中未定义此API码。
        暂时使用开环运动API(2010)中的速度控制能力。

        Args:
            speed: 速度比例 (0.0-1.0) 或绝对速度 (m/s)
            task_id: 未使用

        Returns:
            True if successful
        """
        try:
            # 开环运动API可设置速度，但需要完整参数(vx, vy, w)
            # 目前暂不实现，返回False
            logger.warning("set_speed not implemented - no official API available")
            return False

            ret_code = response.get('ret_code', -1)
            success = ret_code == 0
            if not success:
                error_msg = response.get('err_msg', 'Unknown error')
                logger.warning(f"Failed to set speed: {error_msg}")
            return success

        except Exception as e:
            logger.error(f"Failed to set speed: {e}")
            return False

    # ========== 控制权管理API (端口19207) ==========

    def lock_control(self, nick_name: str = "lerobot") -> bool:
        """抢占控制权.

        使用官方API: robot_config_lock_req (4005), 端口19207.
        必须抢占控制权后才能下发导航/控制指令。如果其他客户端(如Roboshop)持有控制权，
        导航指令会被静默拒绝。

        Args:
            nick_name: 控制权抢占者名称，用于标识

        Returns:
            True if lock acquired
        """
        try:
            response = self._send_request(
                self.PORT_CONFIG,
                self.API_LOCK_CONTROL,
                {"nick_name": nick_name}
            )

            ret_code = response.get('ret_code', -1)
            if ret_code == 0:
                logger.info(f"Control lock acquired: nick_name={nick_name}")
                return True
            else:
                error_msg = response.get('err_msg', 'Unknown error')
                logger.warning(f"Failed to acquire control lock: {error_msg}")
                return False

        except Exception as e:
            logger.error(f"Failed to lock control: {e}")
            return False

    def unlock_control(self) -> bool:
        """释放控制权.

        使用官方API: robot_config_unlock_req (4006), 端口19207.
        只能释放自己抢占的控制权，不能释放别人的控制权。

        Returns:
            True if lock released
        """
        try:
            response = self._send_request(
                self.PORT_CONFIG,
                self.API_UNLOCK_CONTROL,
                {}
            )

            ret_code = response.get('ret_code', -1)
            if ret_code == 0:
                logger.info("Control lock released")
                return True
            else:
                error_msg = response.get('err_msg', 'Unknown error')
                logger.warning(f"Failed to release control lock: {error_msg}")
                return False

        except Exception as e:
            logger.error(f"Failed to unlock control: {e}")
            return False

    def clear_all_errors(self) -> bool:
        """清除AGV当前所有报错.

        使用官方API: robot_config_clearallerrors_req (4009), 端口19207.
        仅清除error级别的错误。

        Returns:
            True if errors cleared
        """
        try:
            response = self._send_request(
                self.PORT_CONFIG,
                self.API_CLEAR_ALL_ERRORS,
                {}
            )

            ret_code = response.get('ret_code', -1)
            if ret_code == 0:
                logger.info("All errors cleared")
                return True
            else:
                error_msg = response.get('err_msg', 'Unknown error')
                logger.warning(f"Failed to clear errors: {error_msg}")
                return False

        except Exception as e:
            logger.error(f"Failed to clear errors: {e}")
            return False

    # ========== 站点查询API ==========

    def query_stations(self) -> list[dict]:
        """查询当前载入地图中的所有站点信息.

        使用官方API: robot_status_station_req (1301), 端口19204.
        返回所有站点的id/x/y/r/type/desc信息，可用于:
        - 发现有效的站点名称用于导航
        - 构建站点坐标地图
        - 确认目标站点是否存在于地图中

        Returns:
            站点列表，每个站点包含 id, type, x, y, r, desc
            站点类型: LocationMark(导航点), ChargePoint(充电点), ActionPoint(动作点)
        """
        try:
            response = self._send_request(
                self.PORT_STATUS,
                self.API_STATION_QUERY,
                {}
            )

            stations = response.get('stations', [])

            # 更新站点地图缓存
            for station in stations:
                station_id = station.get('id', '')
                if station_id and station.get('type') == 'LocationMark':
                    self._station_map[station_id] = AGVPosition(
                        x=float(station.get('x', 0.0)),
                        y=float(station.get('y', 0.0)),
                        theta=float(station.get('r', 0.0)),
                    )

            logger.info(f"Queried {len(stations)} stations, cached {len(self._station_map)} navigation points")
            return stations

        except Exception as e:
            logger.error(f"Failed to query stations: {e}")
            return []

    def query_navigation_path(self, target_station: str) -> list[str]:
        """获取路径导航的规划路径(不实际执行导航).

        使用官方API: robot_task_target_path_req (3053), 端口19206.
        只返回规划路径上的站点序列，不触发导航。

        Args:
            target_station: 目标站点ID

        Returns:
            路径上的站点ID列表(从当前位置到目标)
        """
        try:
            response = self._send_request(
                self.PORT_NAVIGATION,
                self.API_GET_PATH,
                {"id": target_station}
            )

            ret_code = response.get('ret_code', -1)
            if ret_code == 0:
                path = response.get('path', [])
                logger.info(f"Path to {target_station}: {path}")
                return path
            else:
                error_msg = response.get('err_msg', 'Unknown error')
                logger.warning(f"Failed to get path: {error_msg}")
                return []

        except Exception as e:
            logger.error(f"Failed to query navigation path: {e}")
            return []

    # ========== 辅助方法 ==========

    def wait_for_arrival(
        self,
        target_station: str,
        timeout: float = 60.0,
        poll_interval: float = 0.2,
        tolerance: float = 0.3,
        wait_for_orientation: bool = True,
    ) -> bool:
        """等待到达目标站点并完成姿态调整.

        Seer AGV路径导航到达站点分两步：
        1. 移动到XY位置 → current_station切换为目标站点
        2. 原地旋转调整朝向角(theta) → running_status变为IDLE

        如果wait_for_orientation=True，到达站点后还需等待AGV变为IDLE
        才算真正到达（确保旋转完成）。否则只检查站点名匹配。

        Args:
            target_station: 目标站点ID
            timeout: 最大等待时间 (秒)
            poll_interval: 状态轮询间隔 (秒)
            tolerance: 距离容差 (米)
            wait_for_orientation: 是否等待姿态调整完成(旋转到位后AGV变为IDLE)

        Returns:
            True if arrived, False if timeout or error
        """
        start_time = time.time()
        logger.info(f"Waiting for arrival at {target_station} (timeout={timeout}s, wait_orientation={wait_for_orientation})")

        station_reached = False

        while time.time() - start_time < timeout:
            status = self.get_status(use_cache=False)

            # 检查站点ID匹配 (XY位置已到达)
            if status.current_station == target_station:
                if not station_reached:
                    station_reached = True
                    logger.info(f"Station reached: {target_station} (position arrived)")

                if wait_for_orientation:
                    # 等待姿态调整完成：AGV从EXECUTING变为IDLE
                    if status.status_code == self.STATUS_IDLE and not status.is_moving:
                        logger.info(f"Arrived at station: {target_station} (orientation complete)")
                        self._current_navigation_target = None
                        return True
                    else:
                        logger.debug(f"Station reached but still adjusting orientation: status={status.status_code}, moving={status.is_moving}")
                        time.sleep(poll_interval)
                        continue
                else:
                    logger.info(f"Arrived at station: {target_station}")
                    self._current_navigation_target = None
                    return True

            # 检查是否到达目标位置 (如果有坐标信息)
            if target_station in self._station_map:
                target_pos = self._station_map[target_station]
                current_pos = status.position
                distance = ((current_pos.x - target_pos.x) ** 2 +
                           (current_pos.y - target_pos.y) ** 2) ** 0.5
                if distance <= tolerance:
                    if not station_reached:
                        station_reached = True
                        logger.info(f"Position reached near {target_station}: distance={distance:.3f}m")

                    if wait_for_orientation:
                        if status.status_code == self.STATUS_IDLE and not status.is_moving:
                            logger.info(f"Arrived near {target_station} (orientation complete)")
                            self._current_navigation_target = None
                            return True
                        else:
                            logger.debug(f"Position reached but still adjusting orientation")
                            time.sleep(poll_interval)
                            continue
                    else:
                        logger.info(f"Arrived near {target_station}: distance={distance:.3f}m")
                        self._current_navigation_target = None
                        return True

            # 使用导航详情检查目标是否匹配
            nav_detail = self.get_navigation_detail()
            target_id = nav_detail.get('target_id', '')
            target_dist = nav_detail.get('target_dist', -1)
            if target_id:
                logger.debug(f"Navigation detail: target_id={target_id}, dist={target_dist:.2f}m")

            # 使用障碍物检测检查是否被阻挡
            obstacle = self.get_obstacle_status()
            if obstacle.get('blocked', False):
                logger.warning(f"AGV blocked by obstacle at ({obstacle['block_x']:.2f}, {obstacle['block_y']:.2f})")

            # 使用任务进度获取百分比
            progress = self.get_task_progress()
            pct = progress.get('percentage', 0)
            if pct > 0:
                logger.debug(f"Navigation progress: {pct}%")

            # 检查异常
            if status.error_code != 0:
                logger.error(f"AGV error during navigation: {status.error_code} - {status.error_message}")
                return False

            # 检查是否已停止但未到达 (仅当站点还没到达时才算失败)
            if not station_reached and not status.is_moving and status.status_code == self.STATUS_IDLE:
                elapsed = time.time() - self._navigation_start_time
                if elapsed > 5.0:  # 给导航一点启动时间
                    logger.warning(f"AGV stopped (idle) but not at target {target_station}")
                    return False

            # 等待下次轮询
            time.sleep(poll_interval)

        # 超时
        if station_reached:
            logger.warning(f"Timeout: station {target_station} reached but orientation not complete")
        else:
            logger.warning(f"Timeout waiting for arrival at {target_station}")
        return False

    def wait_for_idle(self, timeout: float = 30.0) -> bool:
        """等待AGV变为空闲状态.

        Args:
            timeout: 最大等待时间

        Returns:
            True if idle, False if timeout
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            status = self.get_status(use_cache=False)
            if status.status_code == self.STATUS_IDLE and not status.is_moving:
                return True
            time.sleep(0.5)

        logger.warning("Timeout waiting for AGV idle")
        return False

    def set_station_map(self, station_map: dict[str, tuple[float, float, float]]) -> None:
        """设置站点坐标地图.

        Args:
            station_map: {station_id: (x, y, theta)} 字典
        """
        self._station_map = {
            station_id: AGVPosition(x=x, y=y, theta=theta)
            for station_id, (x, y, theta) in station_map.items()
        }
        logger.info(f"Station map loaded: {len(self._station_map)} stations")

    def is_navigating(self) -> bool:
        """检查是否正在导航."""
        status = self.get_status()
        return status.is_moving or status.status_code == self.STATUS_EXECUTING

    def get_navigation_progress(self) -> dict:
        """获取导航进度信息.

        Returns:
            进度信息字典
        """
        if not self._current_navigation_target:
            return {"navigating": False}

        elapsed = time.time() - self._navigation_start_time
        status = self.get_status(use_cache=False)

        return {
            "navigating": True,
            "target": self._current_navigation_target,
            "elapsed_time": elapsed,
            "current_position": status.position,
            "current_station": status.current_station,
            "is_moving": status.is_moving,
            "battery": status.battery,
        }

    def get_info(self) -> dict:
        """获取控制器信息."""
        return {
            "host": self.host,
            "connected": self.is_connected(),
            "ports_connected": list(self._sockets.keys()),
            "current_navigation_target": self._current_navigation_target,
            "last_status": self._last_status,
        }