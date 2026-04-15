import time
import math
from typing import List, Dict, Any, Tuple, Optional
import datetime
# 导入更新后的 eu_motor_py 绑定
import eu_motor_py 
from .hardware_interface import HardwareInterface
from lerobot.utils.monitor_utils import monitor_performance

class EyouMotorHardware(HardwareInterface):
    """
    一个模仿 supre_robot_control::EyouSystemInterface 的 Python 类。
    
    该类采用混合设计模式：
    1. 它作为有状态对象，在内部维护 hw_states_* 和 hw_commands_* 变量。
    2. read() 和 write() 方法同时提供清晰的参数和返回值，以方便控制循环。
    """

    def __init__(self):
        """构造函数。初始化内部状态存储。"""
        self.can_manager_: Optional[eu_motor_py.CanNetworkManager] = None
        self.feedback_manager_: Optional[eu_motor_py.MotorFeedbackManager] = None
        self.motor_nodes_: List[eu_motor_py.EuMotorNode] = []
        self.joint_names_: List[str] = []

        # --- 恢复内部状态和指令存储 ---
        self.hw_states_positions_: List[float] = []
        self.hw_states_velocities_: List[float] = []
        self.hw_states_torques_: List[float] = []

        self.hw_commands_positions_: List[float] = []
        self.hw_start_enabled_: List[bool] = []

        # --- 软件速度计算（因为硬件反馈速度始终为0）---
        self._prev_positions_: List[float] = []
        self._prev_time_: float = 0.0
        self._enable_velocity_calculation: bool = False  # 默认关闭，避免每帧计算开销

        self._config: Dict[str, Any] = {}
        self._last_log_time = time.monotonic()
        self._max_write_duration_us = 0.0

    def init(self, config: Dict[str, Any]) -> bool:
        """
        初始化硬件接口。
        【已更新以从 'parameters' 中读取 start_enabled】
        """
        print("Initializing EyouMotorHardware...")
        self._config = config
        
        try:
            # 读取顶层参数
            can_device_index = int(self._config["can_device_index"])
            baud_rate_str = self._config["can_baud_rate"]
            
            baud_rate_map = {
                "1M": eu_motor_py.Baudrate.BPS_1M,
                "500K": eu_motor_py.Baudrate.BPS_500K,
                "250K": eu_motor_py.Baudrate.BPS_250K,
            }
            can_baud_rate = baud_rate_map[baud_rate_str]

            print(f"CAN Device Index: {can_device_index}, Baud Rate: {baud_rate_str}")
            
            # 验证并初始化内部存储
            if "joints" not in self._config or not self._config["joints"]:
                raise KeyError("config must contain a non-empty 'joints' list")
            num_joints = len(self._config["joints"])
            self.hw_states_positions_ = [0.0] * num_joints
            self.hw_states_velocities_ = [0.0] * num_joints
            self.hw_states_torques_ = [0.0] * num_joints

            self.hw_commands_positions_ = [0.0] * num_joints
            self.hw_start_enabled_ = [True] * num_joints # 默认全部启用

            # 软件速度计算初始化
            self._prev_positions_ = [0.0] * num_joints
            self._prev_time_ = 0.0

            # 从配置读取是否启用速度计算（默认关闭以优化性能）
            self._enable_velocity_calculation = self._config.get("enable_velocity_calculation", False)
            if self._enable_velocity_calculation:
                print("Velocity calculation is ENABLED.")
            else:
                print("Velocity calculation is DISABLED (for performance optimization).")

            # 初始化CAN总线
            self.can_manager_ = eu_motor_py.CanNetworkManager()
            self.can_manager_.init_device(eu_motor_py.DeviceType.Canable, can_device_index, can_baud_rate)
            print("CAN device initialized successfully.")

            # 初始化电机节点
            self.motor_nodes_ = []
            self.joint_names_ = []
            
            for i, joint_info in enumerate(self._config["joints"]):
                joint_name = joint_info.get("name")
                if not joint_name:
                    raise ValueError(f"Joint at index {i} is missing a 'name'.")

                parameters = joint_info.get("parameters", {})
                if "node_id" not in parameters:
                    raise KeyError(f"'node_id' is missing in 'parameters' for joint '{joint_name}'.")
                node_id = int(parameters["node_id"])
                
                self.joint_names_.append(joint_name)
                
                print(f"Initializing motor for joint '{joint_name}' with Node ID {node_id}")
                motor = eu_motor_py.EuMotorNode(can_device_index, node_id)
                self.motor_nodes_.append(motor)

                # --- 核心修改在这里 ---
                # 从 parameters 字典中读取 start_enabled
                # .get(key, default_value) 使得这个参数是可选的
                # 如果 'start_enabled' 不存在，默认值为 True
                start_enabled = parameters.get("start_enabled", True)
                
                # YAML解析器可能会将 'false' 读为布尔值 False，但为了安全，我们还是处理字符串
                if str(start_enabled).lower() == 'false':
                    self.hw_start_enabled_[i] = False
                    print(f"  -> Joint '{joint_name}' is configured to be DISABLED on start.")
                else:
                    self.hw_start_enabled_[i] = True
                    # 默认启用时可以不打印，保持日志简洁
                    # print(f"  -> Joint '{joint_name}' is configured to be ENABLED on start (default).")


        except (KeyError, ValueError, RuntimeError, TypeError) as e:
            import traceback
            print(f"Error during EyouMotorHardware initialization: {e}")
            traceback.print_exc()
            return False

        print("Initialization successful.")
        return True

    def activate(self) -> bool:
        """
        模仿 on_activate。激活硬件并更新内部状态为初始值。
        
        :return: 如果成功则返回 True。
        """
        print("Activating EyouMotorHardware...")
        
        try:
            # 1. 读取初始状态并更新内部成员变量
            for i, motor in enumerate(self.motor_nodes_):
                pos = motor.get_position()
                vel = motor.get_velocity()
                torque = motor.get_torque()

                self.hw_states_positions_[i] = pos
                self.hw_states_velocities_[i] = vel
                self.hw_states_torques_[i] = float(torque)

                self.hw_commands_positions_[i] = pos # 防止启动时跳动
                print(f"Initial state for {self.joint_names_[i]}: Pos={pos:.2f}, Vel={vel:.2f}")

            # 2. 配置并使能电机 (逻辑与之前版本相同)
            for i, motor in enumerate(self.motor_nodes_):
                joint_name = self.joint_names_[i]
                if self.hw_start_enabled_[i]:
                    print(f"Enabling motor for joint {joint_name}...")
                    if not all([motor.clear_fault(),
                                motor.configure_csp_mode(0, False),
                                motor.start_auto_feedback(0, 255, 20),
                                motor.start_error_feedback_tpdo(1, 255, 60)]):
                        print(f"Error: Failed to configure enabled joint {joint_name}")
                        return False
                else:
                    print(f"Skipping activation for joint {joint_name} as it is disabled.")
                    motor.disable()
                    if not all([motor.clear_fault(),
                                motor.start_auto_feedback(0, 255, 20),
                                motor.start_error_feedback_tpdo(1, 255, 60)]):
                         print(f"Warning: Failed to configure disabled joint {joint_name}")
            
            self.feedback_manager_ = eu_motor_py.MotorFeedbackManager.get_instance()
            self.feedback_manager_.register_callback()
            print("Global feedback callback registered.")

        except RuntimeError as e:
            print(f"Error during activation: {e}")
            return False

        print("Activation successful.")
        return True
    def read(self) -> List[Tuple[Optional[float], Optional[float]]]:
        """
        更新内部状态并返回一份新的状态拷贝。

        :return: (positions, torques) 元组列表。
        """
        # 只在启用速度计算时才获取时间和计算 dt
        if self._enable_velocity_calculation:
            current_time = time.perf_counter()
            dt = current_time - self._prev_time_ if self._prev_time_ > 0 else 0.001

        for i, motor in enumerate(self.motor_nodes_):
            feedback = motor.get_latest_feedback()

            if feedback.last_update_time > datetime.timedelta(0):
                new_position = feedback.position_deg

                # 只在启用时计算速度，避免每帧不必要的计算开销
                if self._enable_velocity_calculation:
                    if dt > 0.0001:  # 避免除以0
                        velocity = (new_position - self._prev_positions_[i]) / dt
                    else:
                        velocity = 0.0
                    self.hw_states_velocities_[i] = velocity
                    self._prev_positions_[i] = new_position

                self.hw_states_positions_[i] = new_position
                self.hw_states_torques_[i] = float(feedback.torque_milli)/1000.0

        # 只在启用时更新时间
        if self._enable_velocity_calculation:
            self._prev_time_ = current_time

        # 返回内部状态的拷贝，防止外部代码意外修改
        return list(zip(self.hw_states_positions_, self.hw_states_torques_))

    def read_velocities(self) -> List[float]:
        """
        返回所有关节的速度数据。

        :return: 速度列表 (°/s)
        """
        return list(self.hw_states_velocities_)
    def busy_wait(self, wait_time_s):
        end_time = time.perf_counter() + wait_time_s
        while time.perf_counter() < end_time:
            pass    
    def write(self, commands_positions: List[float]):
        """
        用传入的指令更新内部指令，然后发送到硬件。
        
        :param commands_positions: 要发送的目标位置列表。
        """
        start_time = time.perf_counter()

        # 1. 使用传入的参数更新内部指令变量
        self.hw_commands_positions_ = commands_positions

        any_motor_enabled = False
        # 2. 从内部指令变量读取数据并发送
        for i, motor in enumerate(self.motor_nodes_):
            if self.hw_start_enabled_[i]:
                result = motor.send_csp_target_position(self.hw_commands_positions_[i],0, False)
                if result != 0:
                    print(f"Error: Failed to send command to joint {self.joint_names_[i]}")
                if i!=0 and i%6==0:
                    self.busy_wait(0.001)
                any_motor_enabled = True

        #if any_motor_enabled:
        #    for i, motor in enumerate(self.motor_nodes_):
        #        if self.hw_start_enabled_[i]:
        #            motor.send_sync()
        #            break
        
        # 3. 性能日志
        end_time = time.perf_counter()
        current_duration_us = (end_time - start_time) * 1_000_000
        
        if current_duration_us > self._max_write_duration_us:
            self._max_write_duration_us = current_duration_us
            
        now = time.monotonic()
        if (now - self._last_log_time) >= 1.0:
            print(f"Max write() duration in last second: {self._max_write_duration_us:.0f} us")
            self._max_write_duration_us = 0.0
            self._last_log_time = now

    def deactivate(self):
        """停用硬件。"""
        print("Deactivating EyouMotorHardware...")
        try:
            for motor in self.motor_nodes_:
                motor.disable()
        except RuntimeError as e:
            print(f"Error during deactivation: {e}")
        print("Deactivation successful.")

    # ==================== CST 力矩控制模式支持 ====================
    # 用于力反馈功能：将 Follower 的力数据转换为 Leader 的阻尼力矩

    def configure_cst_mode(self, interpolation_period_ms: int = 10) -> bool:
        """
        配置所有电机为 CST (Cyclic Synchronous Torque) 力矩控制模式。

        用于力反馈功能，使 Leader 电机能够接收力矩指令产生阻尼感。

        注意：此方法会配置所有电机，不受 start_enabled 参数影响。
        因为 Leader 作为摇操设备需要特殊处理：启动时不使能，但力反馈时需要配置 CST。

        Args:
            interpolation_period_ms: 插补周期（毫秒），默认10ms

        Returns:
            bool: 配置成功返回 True
        """
        print(f"Configuring CST mode with interpolation period {interpolation_period_ms}ms...")

        try:
            for i, motor in enumerate(self.motor_nodes_):
                joint_name = self.joint_names_[i]
                print(f"Configuring CST mode for {joint_name} (node_id={motor.get_node_id()})...")

                # 1. 清除故障
                print(f"  Step 1: Clearing fault...")
                if not motor.clear_fault():
                    print(f"Error: Failed to clear fault for {joint_name}")
                    return False

                # 2. 配置反馈 (即使 start_enabled=false 也需要反馈来读取位置)
                print(f"  Step 2: Configuring feedback...")
                if not motor.start_auto_feedback(0, 255, 20):
                    print(f"Warning: Failed to start auto feedback for {joint_name}")
                if not motor.start_error_feedback_tpdo(1, 255, 60):
                    print(f"Warning: Failed to start error feedback for {joint_name}")

                # 3. 使能电机并切换到 CST 模式
                print(f"  Step 3: Enabling motor in CST mode...")
                result = motor.enable(eu_motor_py.OperateMode.CST)
                if not result:
                    print(f"Error: Failed to enable motor in CST mode for {joint_name}")
                    return False

                # 4. 配置 CST 模式参数
                print(f"  Step 4: Configuring CST PDO...")
                result = motor.configure_cst_mode(interpolation_period_ms, 0, True)
                if not result:  # 返回 bool，False 表示失败
                    print(f"Error: configure_cst_mode failed for {joint_name}")
                    return False

                print(f"  CST mode configured successfully for {joint_name}")

            print("CST mode configuration successful.")
            return True

        except Exception as e:
            print(f"Error during CST mode configuration: {e}")
            return False

    def configure_csp_mode(self) -> bool:
        """
        重新配置所有电机为 CSP (Cyclic Synchronous Position) 位置控制模式。

        从 CST 模式切换回位置控制模式。

        Returns:
            bool: 配置成功返回 True
        """
        print("Reconfiguring CSP (Position) mode...")

        try:
            for i, motor in enumerate(self.motor_nodes_):
                if self.hw_start_enabled_[i]:
                    result = motor.configure_csp_mode(0, True)
                    if result != 0:
                        print(f"Error: Failed to configure CSP mode for joint {self.joint_names_[i]}")
                        return False

            print("CSP mode configuration successful.")
            return True

        except Exception as e:
            print(f"Error during CSP mode configuration: {e}")
            return False

    def write_torques(self, torques: List[float], rated_torque: float = 2.0) -> None:
        """
        发送力矩指令到所有电机（CST 模式）。

        注意：此方法会向所有电机发送力矩指令，不受 start_enabled 参数影响。
        因为 Leader 作为摇操设备需要特殊处理：启动时不使能，但力反馈时需要发送力矩。

        Args:
            torques: 目标力矩列表，单位：Nm
            rated_torque: 电机额定力矩，单位：Nm，默认2.0Nm
                         用于将 Nm 转换为千分之额定力矩
        """
        if len(torques) != len(self.motor_nodes_):
            raise ValueError(f"Torque list length {len(torques)} does not match motor count {len(self.motor_nodes_)}")

        any_nonzero_torque = False
        for i, motor in enumerate(self.motor_nodes_):
            # 单位转换：Nm → 千分之额定力矩 (permille)
            # 公式：torque_permille = torque_Nm / rated_torque * 1000
            torque_permille = int(torques[i] / rated_torque * 1000)

            # 发送力矩指令（第三个参数 False 表示不自动SYNC）
            result = motor.send_cst_target_torque(torque_permille, 0, False)

            if torques[i] != 0.0:
                any_nonzero_torque = True
                # 只打印非零力矩的调试信息
                print(f"DEBUG: Joint {self.joint_names_[i]}, torque={torques[i]:.3f}Nm, permille={torque_permille}, result={result}")

        # 发送SYNC信号（CST模式需要SYNC才能生效）
        if any_nonzero_torque:
            for motor in self.motor_nodes_:
                motor.send_sync()
                break  # 只需要发送一次SYNC，所有电机共享CAN总线

    def get_joint_count(self) -> int:
        """返回硬件中的电机数量。"""
        return len(self.motor_nodes_)

# --- 主程序：演示如何使用混合模式的硬件接口 ---
if __name__ == "__main__":
    # ==================== 配置修改开始 ====================
    # 根据 EyouMotorHardware.init 的更新，修改了此处的配置结构。
    # 主要变化：
    # 1. 'node_id' 现在位于 'parameters' 字典内部。
    # 2. 'parameters' 中新增了可选的 'start_enabled' 参数。
    # 3. 移除了不再使用的 'can_device_type'。
    robot_config = {
        "can_device_index": 1,
        "can_baud_rate": "1M",
        "joints": [
            {
                "name": "right_arm_joint_2",
                "parameters": {
                    "node_id": 12,
                    # "start_enabled" 在此处被省略，将使用默认值 True
                }
            },
            {
                "name": "right_arm_joint_4",
                "parameters": {
                    "node_id": 14,
                    "start_enabled": True  # 明确设置为 True (与默认行为相同)
                }
            },
            # 添加第三个关节，以演示 "start_enabled": False 的效果
            # 这个关节将被初始化，但不会在 activate() 中被使能
            {
                "name": "right_arm_joint_6",
                "parameters": {
                    "node_id": 16,
                    "start_enabled": False # 设置为在启动时不使能
                }
            }
        ]
    }
    # ==================== 配置修改结束 ====================

    robot = EyouMotorHardware()

    if not robot.init(robot_config):
        print("Failed to initialize robot hardware. Exiting.")
        exit(1)

    try:
        if not robot.activate():
            print("Failed to activate robot hardware. Exiting.")
            exit(1)
        
        initial_positions = list(robot.hw_commands_positions_)
        print(f"\n--- Starting Control Loop (Press Ctrl+C to exit) ---")
        print(f"Initial positions: {[f'{p:.2f}' for p in initial_positions]}")
        
        target_positions = list(initial_positions)
        
        control_frequency = 100
        control_period = 1.0 / control_frequency
        start_loop_time = time.time()
        
        while True:
            loop_start = time.perf_counter()
            
            # a. 读取硬件状态 (更新内部状态并返回拷贝)
            current_positions = robot.read()
            
            # 定期打印状态
            if int(loop_start * 10) % 10 == 0:
                pos_str = ", ".join([f"{p:7.2f}" for p in current_positions])
                # 标记出被禁用的电机
                enabled_status = [" " if robot.hw_start_enabled_[i] else "D" for i in range(len(robot.hw_start_enabled_))]
                status_str = "".join(enabled_status)
                print(f"Time: {time.time() - start_loop_time:5.2f}s | Pos: [{pos_str}] | Enabled: [{status_str}]")


            # b. Controller Logic: 计算新的目标位置
            elapsed_time = time.time() - start_loop_time
            
            amplitude = 30.0 # 幅度增大以便观察
            frequency = 0.2
            
            for i in range(len(target_positions)):
                # 只对已使能的电机应用运动指令
                if robot.hw_start_enabled_[i]:
                    phase = i * (math.pi / 2)
                    
                    # 1. 创建一个在 [0, 1] 范围内振荡的归一化值
                    normalized_oscillation = (math.sin(2 * math.pi * frequency * elapsed_time + phase) + 1) / 2
                    
                    # 2. 计算始终为正的偏移量
                    offset = amplitude * normalized_oscillation
                    
                    # 3. 将偏移量加到初始位置上
                    target_positions[i] = initial_positions[i] + offset
                else:
                    # 对于禁用的电机，保持其目标位置不变
                    target_positions[i] = initial_positions[i]

            # c. 写入硬件
            robot.write(target_positions)
            
            # d. 维持控制频率
            loop_end = time.perf_counter()
            sleep_time = control_period - (loop_end - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nCtrl+C pressed. Shutting down.")
    except Exception as e:
        import traceback
        print(f"\nAn unexpected error occurred: {e}")
        traceback.print_exc()
    finally:
        robot.deactivate()