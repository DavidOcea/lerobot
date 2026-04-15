import yaml

import os
from typing import List, Dict, Any, Tuple

# 导入你提供的两个硬件类
# 假设它们在名为 eyou_hardware.py 和 gripper_hardware.py 的文件中
from lerobot.motors.eyou import EyouMotorHardware,AsyncInterpolator
from lerobot.motors.gripper import JodellGripperHardware
from lerobot.utils.monitor_utils import monitor_performance

class SupreRobotHardwareManager:
    """
    一个聚合器类，用于管理多个异构硬件接口，
    并为上层控制程序提供一个统一的接口。
    """

    # 映射配置中的类型字符串到实际的类
    HARDWARE_TYPE_MAP = {
        "EyouMotorHardware": EyouMotorHardware,
        "JodellGripperHardware": JodellGripperHardware,
    }

    def __init__(self, config_path: str, control_frequency: float = 30, use_interpolation: bool = False, enable_velocity_read: bool = False):
        """
        构造函数。
        :param config_path: 指向 robot_config.yaml 文件的路径。
        :param control_frequency: 控制频率 (Hz)
        :param use_interpolation: 是否启用插值模式
        :param enable_velocity_read: 是否读取速度数据（默认关闭以优化性能）
        """
        print("Initializing SupreRobotHardwareManager...")
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)

        self.joint_order: List[str] = self._config["joint_order"]
        self.num_joints = len(self.joint_order)
        print(f"Total number of joints configured: {self.num_joints}")
        print(f"Joint order: {self.joint_order}")

        # 存储硬件实例
        self._hardware_instances: List[Any] = []
        
        # 核心映射表：
        # key: 全局关节索引 (在 self.joint_order 中的位置)
        # value: dict{'instance': 硬件实例, 'hw_index': 该关节在硬件实例内部的索引}
        self._joint_map: Dict[int, Dict[str, Any]] = {}

        # 全局状态和指令向量
        self.positions = [0.0] * self.num_joints
        self.velocities = [0.0] * self.num_joints
        self.forces = [0.0] * self.num_joints

        self.commands = [0.0] * self.num_joints

        self.use_interpolation = use_interpolation
        self.control_frequency = control_frequency
        self.enable_velocity_read = enable_velocity_read  # 速度读取开关（默认关闭）

        if self.enable_velocity_read:
            print("Velocity reading is ENABLED.")
        else:
            print("Velocity reading is DISABLED (for performance optimization).") 
    def init(self) -> bool:
        """
        根据配置初始化所有硬件接口并构建映射表。
        """
        print("\n--- Initializing all hardware interfaces ---")
        # 1. 创建硬件实例
        for hw_info in self._config["hardware_interfaces"]:
            hw_type_str = hw_info["type"]
            if hw_type_str not in self.HARDWARE_TYPE_MAP:
                print(f"Error: Unknown hardware type '{hw_type_str}'")
                return False
            
            # 动态创建实例
            hw_class = self.HARDWARE_TYPE_MAP[hw_type_str]
            instance = hw_class()
            

            # 如果需要，创建插值器
            if self.use_interpolation:
                interpolation_config = {}
                interpolation_config["control_frequency"]=self.control_frequency
                interpolation_config["interpolation_n"]=hw_info["interpolation"]["interpolation_n"]
                instance = AsyncInterpolator(instance, interpolation_config)

            # 初始化硬件
            # 将 enable_velocity_read 传递给硬件配置（用于控制速度计算）
            hw_config = hw_info["config"].copy()
            hw_config["enable_velocity_calculation"] = self.enable_velocity_read
            if not instance.init(hw_config):
                print(f"Error: Failed to initialize hardware '{hw_info['name']}'")
                return False
                        
            self._hardware_instances.append(instance)
            print(f"Successfully created and initialized '{hw_info['name']}' of type '{hw_type_str}'")

            # 2. 构建关节映射
            # 获取该硬件控制的关节列表
            hw_joint_list = [j['name'] for j in hw_info["config"]["joints"]]
            for hw_index, joint_name in enumerate(hw_joint_list):
                if joint_name not in self.joint_order:
                    print(f"Warning: Joint '{joint_name}' from hardware '{hw_info['name']}' is not in the global joint_order. Ignoring.")
                    continue
                
                # 找到该关节在全局列表中的索引
                global_index = self.joint_order.index(joint_name)
                
                # 存储映射关系
                self._joint_map[global_index] = {
                    'instance': instance,
                    'hw_index': hw_index
                }
        
        # 3. 验证所有关节是否都被映射
        if len(self._joint_map) != self.num_joints:
            print("Error: Mismatch between joints in joint_order and joints defined in hardware_interfaces.")
            mapped_joints = {self.joint_order[i] for i in self._joint_map.keys()}
            unmapped_joints = set(self.joint_order) - mapped_joints
            print(f"Unmapped joints: {unmapped_joints}")
            return False

        print("\n--- Joint mapping completed successfully ---")
        return True

    def activate(self) -> bool:
        """激活所有硬件接口。"""
        print("\n--- Activating all hardware interfaces ---")
        for instance in self._hardware_instances:
            if not instance.activate():
                print(f"Error: Failed to activate hardware instance {instance.__class__.__name__}")
                return False
        print("All hardware activated.")
        # 首次读取以填充初始状态
        self.read()
        self.commands = list(self.positions) # 将初始指令设置为当前位置，防止跳动
        return True

    def deactivate(self):
        """停用所有硬件接口。"""
        print("\n--- Deactivating all hardware interfaces ---")
        for instance in self._hardware_instances:
            instance.deactivate()
        print("All hardware deactivated.")
    def read(self) -> Tuple[List[float], List[float]]:
        """
        从所有硬件读取数据，并聚合成全局状态向量。
        """
        # 1. 从每个硬件读取数据
        hw_results = {inst: inst.read() for inst in self._hardware_instances}

        # 只在启用时读取速度数据（避免额外开销）
        if self.enable_velocity_read:
            hw_velocities = {inst: inst.read_velocities() if hasattr(inst, 'read_velocities') else [] for inst in self._hardware_instances}

        # 2. 使用映射表填充全局状态向量
        for global_index in range(self.num_joints):
            mapping = self._joint_map[global_index]
            instance = mapping['instance']
            hw_index = mapping['hw_index']

            result = hw_results[instance]

            # 根据硬件类型适配不同的返回值
            if isinstance(instance, EyouMotorHardware):
                # EyouMotorHardware.read() -> Tuple[List[float], List[float]]
                self.positions[global_index] = result[hw_index][0]
                self.forces[global_index] = result[hw_index][1]
                # 只在启用时读取速度
                if self.enable_velocity_read and hw_velocities.get(instance):
                    self.velocities[global_index] = hw_velocities[instance][hw_index]
                else:
                    self.velocities[global_index] = 0.0

            elif isinstance(instance, JodellGripperHardware):
                # JodellGripperHardware.read() -> list[float | None]
                pos = result[hw_index][0]
                force = result[hw_index][1]
                self.positions[global_index] = pos if pos is not None else self.positions[global_index] # 保持旧值如果读取失败
                self.velocities[global_index] = 0.0 # 夹爪没有速度反馈
                self.forces[global_index] = force if force is not None else self.forces[global_index]
            else:
                pos = result[hw_index]
                force = result[hw_index][1]
                self.velocities[global_index] = 0.0
                self.positions[global_index] = pos if pos is not None else self.positions[global_index] # 保持旧值如果读取失败
                self.forces[global_index] = force if force is not None else self.forces[global_index]

        return (list(self.positions),list(self.forces))

    def read_velocities(self) -> List[float]:
        """
        返回所有关节的速度数据。

        Returns:
            List[float]: 速度列表 (°/s)
        """
        return list(self.velocities)
    def write(self, command_positions: List[float]):
        """
        接收全局指令向量，并分发到各个硬件。
        """
        if len(command_positions) != self.num_joints:
            raise ValueError(f"Command vector length ({len(command_positions)}) does not match number of joints ({self.num_joints}).")

        self.commands = command_positions
        
        # 1. 准备分发给每个硬件的指令字典
        #    key: 硬件实例
        #    value: 该硬件的指令列表
        hw_commands = {}
        for instance in self._hardware_instances:
            # 获取该硬件控制的关节数量并创建指令列表
            #num_hw_joints = len(instance.joint_names_) if hasattr(instance, 'joint_names_') else len(instance.slave_ids)
            num_hw_joints = instance.get_joint_count()
            hw_commands[instance] = [None] * num_hw_joints

        # 2. 遍历全局指令，使用映射表分发
        for global_index, command_value in enumerate(self.commands):
            mapping = self._joint_map[global_index]
            instance = mapping['instance']
            hw_index = mapping['hw_index']
            
            hw_commands[instance][hw_index] = command_value
            
        # 3. 将准备好的指令发送到每个硬件
        for instance, commands in hw_commands.items():
            # JodellGripperHardware.write() 接受 None 值，而 EyouMotorHardware 需要一个完整的浮点数列表。
            # 我们的分发逻辑保证了 Eyou 的列表是完整的。
            print(f"Sending command to {instance.__class__.__name__}: {commands}")
            instance.write(commands)

    # ==================== CST 力矩控制支持 ====================
    # 用于力反馈功能

    def configure_cst_mode(self, interpolation_period_ms: int = 10) -> bool:
        """
        配置所有电机为 CST 力矩控制模式。

        Args:
            interpolation_period_ms: 插补周期（毫秒）

        Returns:
            bool: 全部配置成功返回 True
        """
        print("Configuring CST mode for all hardware...")
        success = True
        for instance in self._hardware_instances:
            if hasattr(instance, 'configure_cst_mode'):
                if not instance.configure_cst_mode(interpolation_period_ms):
                    success = False
        return success

    def configure_csp_mode(self) -> bool:
        """
        重新配置所有电机为 CSP 位置控制模式。

        Returns:
            bool: 全部配置成功返回 True
        """
        print("Reconfiguring CSP mode for all hardware...")
        success = True
        for instance in self._hardware_instances:
            if hasattr(instance, 'configure_csp_mode'):
                if not instance.configure_csp_mode():
                    success = False
        return success

    def write_torques(self, torques: List[float], rated_torque: float = 2.0) -> None:
        """
        发送力矩指令到所有电机（CST 模式）。

        用于力反馈功能，将 Follower 的力数据转换为 Leader 的阻尼力矩。

        Args:
            torques: 全局力矩向量，单位：Nm
            rated_torque: 电机额定力矩，单位：Nm
        """
        if len(torques) != self.num_joints:
            raise ValueError(f"Torque vector length ({len(torques)}) does not match number of joints ({self.num_joints}).")

        # 分发力矩指令到各个硬件
        hw_torques = {}
        for instance in self._hardware_instances:
            num_hw_joints = instance.get_joint_count()
            hw_torques[instance] = [0.0] * num_hw_joints

        # 使用映射表分发
        for global_index, torque_value in enumerate(torques):
            mapping = self._joint_map[global_index]
            instance = mapping['instance']
            hw_index = mapping['hw_index']
            hw_torques[instance][hw_index] = torque_value

        # 发送到每个硬件
        for instance, torque_list in hw_torques.items():
            if hasattr(instance, 'write_torques'):
                instance.write_torques(torque_list, rated_torque)
