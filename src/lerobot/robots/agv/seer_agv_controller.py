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

Port allocation:
- 19204: Status query
- 19205: Control (stop/pause/resume)
- 19206: Navigation
- 19207: Task management

Reference: /root/workspace/dc_dir/ros2_ws/src/sm_test_tcp_bridge/sm_test_tcp_bridge/tcp_bridge_node.py
"""

import json
import logging
import socket
import struct
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

    # ========== API类型码定义 (参考tcp_bridge_node.py) ==========

    # 状态查询类 API (端口19204)
    API_STATUS_QUERY = 0x03E8      # 1000 - 综合状态查询 (系统版本、地图名、vehicle_id等)
    API_BATTERY_QUERY = 0x03EA     # 1002 - 电量查询 (battery_level, controller_voltage等)
    API_TASK_STATUS_QUERY = 0x03EC # 1004 - 任务状态查询 ✅ 包含位置x, y, angle和current_station
    API_VELOCITY_QUERY = 0x03F4    # 1012 - ❌ 实际返回EMC急停状态 (emergency, soft_emc)，不是速度！
    API_STATION_QUERY = 0x044E     # 1102 - ❌ 实际返回电池/充电状态 (battery_level, charging)，不是站点名！
    # 注意: 0x03F2 (1010) 返回的是 path 路径数据，不是位置
    # TODO: 需要找到正确的速度查询API (vx, vy, vtheta)

    # 控制类 API (端口19205)
    API_STOP = 0x07D2              # 2002 - 急停
    API_PAUSE = 0x07D3             # 2003 - 暂停
    API_RESUME = 0x07D4            # 2004 - 继续
    API_CANCEL_TASK = 0x07D5       # 2005 - 取消当前任务
    API_SET_SPEED = 0x07D6         # 2006 - 设置速度

    # 导航类 API (端口19206)
    API_NAVIGATE_STATION = 0x07E8  # 2024 - 导航到站点
    API_NAVIGATE_POSITION = 0x07E9 # 2025 - 导航到坐标
    API_NAVIGATE_P2P = 0x07EA      # 2026 - 点到点导航
    API_CANCEL_NAVIGATION = 0x07EB # 2027 - 取消导航

    # 任务管理类 API (端口19207)
    API_CREATE_TASK = 0x07F0       # 2032 - 创建任务
    API_QUERY_TASK = 0x07F1        # 2033 - 查询任务

    # ========== 端口分配 ==========
    PORT_STATUS = 19204     # 状态查询
    PORT_CONTROL = 19205    # 控制(急停/暂停/继续)
    PORT_NAVIGATION = 19206 # 导航
    PORT_TASK = 19207       # 任务管理

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
            # 需要连接的端口列表
            ports_to_connect = [
                self.PORT_STATUS,
                self.PORT_NAVIGATION,
                self.PORT_CONTROL,
            ]

            connected_ports = []
            for port in ports_to_connect:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(self.connection_timeout)
                    sock.connect((self.host, port))
                    sock.settimeout(self.read_timeout)
                    self._sockets[port] = sock
                    connected_ports.append(port)
                    logger.info(f"Connected to AGV at {self.host}:{port}")
                except Exception as e:
                    logger.warning(f"Failed to connect to port {port}: {e}")
                    # 继续尝试其他端口

            if len(connected_ports) >= 2:
                # 至少需要状态和导航端口
                logger.info(f"AGV connection established on ports: {connected_ports}")
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
        """断开所有TCP连接."""
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

        for attempt in range(retry_count + 1):
            try:
                # 构建请求包
                packet = self._build_packet(api_type, data)
                logger.debug(f"Sending packet to port {port}: api={api_type:#x}, len={len(packet)}")

                # 发送
                sock.sendall(packet)

                # 接收响应header (16 bytes)
                header_bytes = self._recv_exact(sock, 16)

                # 解析header
                seq, data_len, api_type_resp = self._parse_response_header(header_bytes)

                # 接收payload
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

        API 0x03EA 返回的字段: battery_level, controller_voltage, battery_temp 等

        Returns:
            电量百分比 (0-100)
        """
        try:
            response = self._send_request(
                self.PORT_STATUS,
                self.API_BATTERY_QUERY,
                {}
            )
            # 电量字段名是 battery_level (百分比)
            battery = response.get('battery_level', response.get('controller_voltage', 0))
            return int(battery)
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

        ⚠️ WARNING: API_VELOCITY_QUERY (0x03F4) 实际返回的是急停/EMC状态，不是速度！
        返回字段: emergency, soft_emc, driver_emc

        TODO: 需要找到正确的速度查询 API。

        Returns:
            (vx, vy, vtheta) 单位: m/s, m/s, rad/s
            当前返回 (0, 0, 0) 因为 API 错误
        """
        # TODO: 找到正确的速度查询 API
        # 0x03F4 返回的是 EMC 状态，不是速度
        logger.warning("get_velocity: API 0x03F4 returns EMC status, not velocity. Returning (0, 0, 0)")
        return (0.0, 0.0, 0.0)

    def get_emc_status(self) -> dict:
        """获取急停/EMC状态.

        API_VELOCITY_QUERY (0x03F4) 实际返回的是 EMC 状态。

        Returns:
            {'emergency': bool, 'soft_emc': bool, 'driver_emc': bool}
        """
        try:
            response = self._send_request(
                self.PORT_STATUS,
                self.API_VELOCITY_QUERY,
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

        Returns:
            任务状态字典
        """
        try:
            response = self._send_request(
                self.PORT_STATUS,
                self.API_TASK_STATUS_QUERY,
                {}
            )
            return response.get('data', response)

        except Exception as e:
            logger.error(f"Failed to get task status: {e}")
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
            # 综合查询
            battery = self.get_battery()
            position = self.get_position()
            station = self.get_current_station()
            vx, vy, vtheta = self.get_velocity()
            task_status = self.get_task_status()

            # 解析状态码
            status_code = task_status.get('status', task_status.get('state', 0))
            error_code = task_status.get('error_code', task_status.get('err_code', 0))
            error_message = task_status.get('error_msg', task_status.get('message', ''))

            # 判断是否在移动
            is_moving = (abs(vx) > 0.01 or abs(vy) > 0.01 or abs(vtheta) > 0.01 or
                        status_code == self.STATUS_EXECUTING)

            self._last_status = AGVStatus(
                battery=battery,
                status_code=status_code,
                current_station=station,
                position=position,
                is_moving=is_moving,
                error_code=error_code,
                error_message=error_message,
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

    def move_to_station(self, station_id: str) -> bool:
        """导航到指定站点.

        Args:
            station_id: 目标站点ID

        Returns:
            True if navigation started successfully
        """
        try:
            response = self._send_request(
                self.PORT_NAVIGATION,
                self.API_NAVIGATE_STATION,
                {"station_id": station_id}
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
                error_msg = response.get('err_msg', response.get('msg', 'Unknown error'))
                logger.error(f"Navigation failed: {error_msg}")
                return False

        except Exception as e:
            logger.error(f"Failed to start navigation: {e}")
            return False

    def move_to_position(self, x: float, y: float, theta: float = None) -> bool:
        """导航到指定坐标.

        Args:
            x: 目标x坐标 (米)
            y: 目标y坐标 (米)
            theta: 目标航向角 (弧度)，可选

        Returns:
            True if navigation started successfully
        """
        try:
            data = {"x": x, "y": y}
            if theta is not None:
                data["theta"] = theta

            response = self._send_request(
                self.PORT_NAVIGATION,
                self.API_NAVIGATE_POSITION,
                data
            )

            result = response.get('ret_code', response.get('ret', -1))
            success = result == 0

            if success:
                self._current_navigation_target = f"({x:.2f}, {y:.2f})"
                self._navigation_start_time = time.time()
                logger.info(f"Navigation started: target={x:.2f}, {y:.2f}")
                return True
            else:
                error_msg = response.get('err_msg', response.get('msg', 'Unknown error'))
                logger.error(f"Navigation to position failed: {error_msg}")
                return False

        except Exception as e:
            logger.error(f"Failed to navigate to position: {e}")
            return False

    def cancel_navigation(self) -> bool:
        """取消当前导航任务.

        Returns:
            True if cancelled successfully
        """
        try:
            response = self._send_request(
                self.PORT_NAVIGATION,
                self.API_CANCEL_NAVIGATION,
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

    # ========== 控制API ==========

    def stop(self) -> bool:
        """急停 - 立即停止AGV.

        Returns:
            True if stop successful
        """
        try:
            response = self._send_request(
                self.PORT_CONTROL,
                self.API_STOP,
                {}
            )

            ret_code = response.get('ret_code', -1)
            success = ret_code == 0

            if success:
                self._current_navigation_target = None
                logger.warning("AGV emergency stop executed")
                return True
            else:
                error_msg = response.get('err_msg', 'Unknown error')
                logger.error(f"Emergency stop failed: {error_msg}")
                return False

        except Exception as e:
            logger.error(f"Failed to stop AGV: {e}")
            return False

    def pause(self) -> bool:
        """暂停当前任务.

        Returns:
            True if pause successful
        """
        try:
            response = self._send_request(
                self.PORT_CONTROL,
                self.API_PAUSE,
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
        """继续执行暂停的任务.

        Returns:
            True if resume successful
        """
        try:
            response = self._send_request(
                self.PORT_CONTROL,
                self.API_RESUME,
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

    def cancel_task(self) -> bool:
        """取消当前任务.

        Returns:
            True if cancelled successfully
        """
        try:
            response = self._send_request(
                self.PORT_CONTROL,
                self.API_CANCEL_TASK,
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

    def set_speed(self, speed: float) -> bool:
        """设置AGV速度.

        Args:
            speed: 速度比例 (0.0-1.0) 或绝对速度 (m/s)

        Returns:
            True if successful
        """
        try:
            response = self._send_request(
                self.PORT_CONTROL,
                self.API_SET_SPEED,
                {"speed": speed}
            )

            ret_code = response.get('ret_code', -1)
            success = ret_code == 0
            if not success:
                error_msg = response.get('err_msg', 'Unknown error')
                logger.warning(f"Failed to set speed: {error_msg}")
            return success

        except Exception as e:
            logger.error(f"Failed to set speed: {e}")
            return False

    # ========== 辅助方法 ==========

    def wait_for_arrival(
        self,
        target_station: str,
        timeout: float = 60.0,
        poll_interval: float = 1.0,
        tolerance: float = 0.3,
    ) -> bool:
        """等待到达目标站点.

        Args:
            target_station: 目标站点ID
            timeout: 最大等待时间 (秒)
            poll_interval: 状态轮询间隔 (秒)
            tolerance: 距离容差 (米)

        Returns:
            True if arrived, False if timeout or error
        """
        start_time = time.time()
        logger.info(f"Waiting for arrival at {target_station} (timeout={timeout}s)")

        while time.time() - start_time < timeout:
            status = self.get_status(use_cache=False)

            # 检查站点ID匹配
            if status.current_station == target_station:
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
                    logger.info(f"Arrived near {target_station}: distance={distance:.3f}m")
                    self._current_navigation_target = None
                    return True

            # 检查异常
            if status.error_code != 0:
                logger.error(f"AGV error during navigation: {status.error_code} - {status.error_message}")
                return False

            # 检查是否已停止但未到达
            if not status.is_moving and status.status_code == self.STATUS_IDLE:
                # 空闲状态但未到达目标
                elapsed = time.time() - self._navigation_start_time
                if elapsed > 5.0:  # 给导航一点启动时间
                    logger.warning(f"AGV stopped (idle) but not at target {target_station}")
                    return False

            # 等待下次轮询
            time.sleep(poll_interval)

        # 超时
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