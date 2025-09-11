import time
import math
from typing import List, Dict, Any, Tuple, Optional
import datetime
# 导入更新后的 eu_motor_py 绑定
import eu_motor_py 

class EyouMotorHardware:
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
        self.hw_commands_positions_: List[float] = []
        self.hw_start_enabled_: List[bool] = []
        
        self._config: Dict[str, Any] = {}
        self._last_log_time = time.monotonic()
        self._max_write_duration_us = 0.0

    def init(self, config: Dict[str, Any]) -> bool:
        """
        模仿 on_init。初始化硬件接口并调整内部存储的大小。
        """
        print("Initializing EyouMotorHardware...")
        self._config = config
        
        try:
            can_device_index = int(self._config["can_device_index"])
            baud_rate_str = self._config["can_baud_rate"]
            
            baud_rate_map = {
                "1M": eu_motor_py.Baudrate.BPS_1M,
                "500K": eu_motor_py.Baudrate.BPS_500K,
                "250K": eu_motor_py.Baudrate.BPS_250K,
            }
            if baud_rate_str not in baud_rate_map:
                print(f"Error: Invalid baud rate '{baud_rate_str}'")
                return False
            can_baud_rate = baud_rate_map[baud_rate_str]

            print(f"CAN Device Index: {can_device_index}, Baud Rate: {baud_rate_str}")
            
            # --- 恢复内部存储的初始化 ---
            num_joints = len(self._config["joints"])
            self.hw_states_positions_ = [0.0] * num_joints
            self.hw_states_velocities_ = [0.0] * num_joints
            self.hw_commands_positions_ = [0.0] * num_joints
            self.hw_start_enabled_ = [True] * num_joints

            self.can_manager_ = eu_motor_py.CanNetworkManager()
            self.can_manager_.init_device(eu_motor_py.DeviceType.Canable, can_device_index, can_baud_rate)
            print("CAN device initialized successfully.")

            self.motor_nodes_ = []
            self.joint_names_ = []
            for i, joint_info in enumerate(self._config["joints"]):
                node_id = int(joint_info["node_id"])
                joint_name = joint_info["name"]
                self.joint_names_.append(joint_name)
                
                print(f"Initializing motor for joint '{joint_name}' with Node ID {node_id}")
                motor = eu_motor_py.EuMotorNode(can_device_index, node_id)
                self.motor_nodes_.append(motor)

                if "start_enabled" in joint_info and joint_info["start_enabled"].lower() == "false":
                    self.hw_start_enabled_[i] = False
                    print(f"Joint '{joint_name}' is configured to be disabled on start.")

        except (KeyError, ValueError, RuntimeError) as e:
            print(f"Error during initialization: {e}")
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
                self.hw_states_positions_[i] = pos
                self.hw_states_velocities_[i] = vel
                self.hw_commands_positions_[i] = pos # 防止启动时跳动
                print(f"Initial state for {self.joint_names_[i]}: Pos={pos:.2f}, Vel={vel:.2f}")

            # 2. 配置并使能电机 (逻辑与之前版本相同)
            for i, motor in enumerate(self.motor_nodes_):
                joint_name = self.joint_names_[i]
                if self.hw_start_enabled_[i]:
                    print(f"Enabling motor for joint {joint_name}...")
                    if not all([motor.clear_fault(),
                                motor.configure_csp_mode(),
                                motor.start_auto_feedback(0, 255, 10),
                                motor.start_error_feedback_tpdo(1, 255, 60)]):
                        print(f"Error: Failed to configure enabled joint {joint_name}")
                        return False
                else:
                    print(f"Skipping activation for joint {joint_name} as it is disabled.")
                    motor.disable()
                    if not all([motor.clear_fault(),
                                motor.start_auto_feedback(0, 255, 10),
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

    def read(self) -> list[float | None]:
        """
        更新内部状态并返回一份新的状态拷贝。
        
        :return: (new_positions, new_velocities) 元组。
        """
        for i, motor in enumerate(self.motor_nodes_):
            feedback = motor.get_latest_feedback()
            
            if feedback.last_update_time > datetime.timedelta(0):
                self.hw_states_positions_[i] = feedback.position_deg
                self.hw_states_velocities_[i] = feedback.velocity_dps
        
        # 返回内部状态的拷贝，防止外部代码意外修改
        return list(self.hw_states_positions_)

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
                motor.send_csp_target_position(self.hw_commands_positions_[i])
                any_motor_enabled = True

        if any_motor_enabled:
            for i, motor in enumerate(self.motor_nodes_):
                if self.hw_start_enabled_[i]:
                    motor.send_sync()
                    break
        
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


# --- 主程序：演示如何使用混合模式的硬件接口 ---
if __name__ == "__main__":
    robot_config = {
        "can_device_type": "Canable",
        "can_device_index": 1,
        "can_baud_rate": "1M",
        "joints": [
            {"name": "right_arm_joint_2", "node_id": 12},
            {"name": "right_arm_joint_4", "node_id": 14},
        ]
    }

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
            
            current_positions, current_velocities = robot.read()
            
            if int(loop_start * 10) % 10 == 0:
                pos_str = ", ".join([f"{p:7.2f}" for p in current_positions])
                vel_str = ", ".join([f"{v:7.2f}" for v in current_velocities])
                print(f"Time: {time.time() - start_loop_time:5.2f}s | Pos: [{pos_str}] | Vel: [{vel_str}]")

            # b. Controller Logic: 计算新的目标位置
            elapsed_time = time.time() - start_loop_time
            
            amplitude = 5.0
            frequency = 0.2
            
            for i in range(len(target_positions)):
                phase = i * (math.pi / 2)
                
                # --- 核心修改：确保偏移量非负 ---
                # 1. 创建一个在 [0, 1] 范围内振荡的归一化值
                normalized_oscillation = (math.sin(2 * math.pi * frequency * elapsed_time + phase) + 1) / 2
                
                # 2. 计算始终为正的偏移量
                offset = amplitude * normalized_oscillation
                
                # 3. 将偏移量加到初始位置上
                target_positions[i] = initial_positions[i] + offset

            robot.write(target_positions)
            
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