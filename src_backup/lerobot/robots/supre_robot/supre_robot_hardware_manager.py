import yaml
from typing import List, Dict, Any, Tuple

# 导入你提供的两个硬件类
# 假设它们在名为 eyou_hardware.py 和 gripper_hardware.py 的文件中
from lerobot.motors.eyou import EyouMotorHardware
from lerobot.motors.gripper import JodellGripperHardware

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

    def __init__(self, config_path: str):
        """
        构造函数。
        :param config_path: 指向 robot_config.yaml 文件的路径。
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
        self.commands = [0.0] * self.num_joints

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
            
            # 初始化硬件
            if not instance.init(hw_info["config"]):
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

    def read(self) ->  List[float]:
        """
        从所有硬件读取数据，并聚合成全局状态向量。
        """
        # 1. 从每个硬件读取数据
        hw_results = {inst: inst.read() for inst in self._hardware_instances}

        # 2. 使用映射表填充全局状态向量
        for global_index in range(self.num_joints):
            mapping = self._joint_map[global_index]
            instance = mapping['instance']
            hw_index = mapping['hw_index']
            
            result = hw_results[instance]

            # 根据硬件类型适配不同的返回值
            if isinstance(instance, EyouMotorHardware):
                # EyouMotorHardware.read() -> Tuple[List[float], List[float]]
                self.positions[global_index] = result[hw_index]
                self.velocities[global_index] = 0.0
            elif isinstance(instance, JodellGripperHardware):
                # JodellGripperHardware.read() -> list[float | None]
                pos = result[hw_index]
                self.positions[global_index] = pos if pos is not None else self.positions[global_index] # 保持旧值如果读取失败
                self.velocities[global_index] = 0.0 # 夹爪没有速度反馈
        
        return list(self.positions)

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
            num_hw_joints = len(instance.joint_names_) if hasattr(instance, 'joint_names_') else len(instance.slave_ids)
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
