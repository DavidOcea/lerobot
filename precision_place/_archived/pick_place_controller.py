"""
Precision Pick Place Controller - 精准抓放主控制器

整合ACT策略和视觉伺服，实现毫米级精准放置
"""

import numpy as np
import cv2
import time
import yaml
from pathlib import Path
from typing import Optional, Dict, Any

from .visual_servo import WristCameraVisualServo
from .slot_detector import SlotDetector


class PrecisionPickPlaceController:
    """
    精准抓放控制器
    
    流程：
    1. ACT策略：执行整体抓放动作（粗定位）
    2. 视觉伺服：放置前的精调（毫米级）
    """
    
    def __init__(self, robot, cameras: Dict, policy=None, config_path: str = None):
        """
        Args:
            robot: SupreRobotFollower 实例
            cameras: 相机字典 {'head': ..., 'left_wrist': ..., 'right_wrist': ...}
            policy: 训练好的ACT策略（可选，如果为None则使用手动模式）
            config_path: 配置文件路径
        """
        self.robot = robot
        self.cameras = cameras
        self.policy = policy
        
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 初始化视觉伺服控制器
        self.left_servo = WristCameraVisualServo(
            robot, cameras.get('left_wrist'), "left", config_path
        )
        self.right_servo = WristCameraVisualServo(
            robot, cameras.get('right_wrist'), "right", config_path
        )
        
        # 当前活动手臂
        self.active_arm = "right"
        
        # 统计信息
        self.stats = {
            'total_runs': 0,
            'successful_runs': 0,
            'avg_iterations': 0,
            'avg_final_error': 0
        }
        
    def _load_config(self, config_path: str) -> dict:
        """加载配置"""
        if config_path is None:
            config_path = Path(__file__).parent / "configs" / "precision_config.yaml"
        
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def connect(self):
        """连接机器人和相机"""
        print("正在连接机器人...")
        self.robot.connect()
        print("机器人已连接")
        
        # 连接相机
        for name, camera in self.cameras.items():
            if camera is not None and hasattr(camera, 'connect'):
                print(f"正在连接相机: {name}")
                camera.connect()
        print("所有相机已连接")
    
    def disconnect(self):
        """断开连接"""
        self.robot.disconnect()
        for name, camera in self.cameras.items():
            if camera is not None and hasattr(camera, 'disconnect'):
                camera.disconnect()
        print("已断开所有连接")
    
    def set_active_arm(self, arm: str):
        """设置活动手臂"""
        if arm not in ["left", "right"]:
            raise ValueError("arm must be 'left' or 'right'")
        self.active_arm = arm
        print(f"活动手臂已切换到: {arm}")
    
    def get_active_servo(self) -> WristCameraVisualServo:
        """获取当前活动手臂的视觉伺服控制器"""
        return self.left_servo if self.active_arm == "left" else self.right_servo
    
    def load_templates(self, left_path: str = None, right_path: str = None):
        """加载目标模板"""
        if left_path:
            self.left_servo.load_target_template(left_path)
        if right_path:
            self.right_servo.load_target_template(right_path)
    
    def save_current_as_template(self, arm: str = None):
        """
        将当前位置保存为目标模板
        
        使用方法：
        1. 手动将机器人移动到目标放置位置
        2. 调用此方法保存模板
        """
        if arm is None:
            arm = self.active_arm
        
        servo = self.left_servo if arm == "left" else self.right_servo
        servo.save_target_template()
    
    def calibrate_all(self):
        """执行完整标定流程"""
        print("\n" + "="*60)
        print("开始完整标定流程")
        print("="*60)
        
        # 1. 标定右臂
        print("\n--- 标定右臂 ---")
        self.right_servo.calibrate_pixel_to_mm(5.0)
        
        # 2. 标定左臂（如果需要）
        if self.cameras.get('left_wrist') is not None:
            print("\n--- 标定左臂 ---")
            self.left_servo.calibrate_pixel_to_mm(5.0)
        
        print("\n" + "="*60)
        print("标定完成!")
        print("="*60)
    
    def pick(self, use_policy: bool = True):
        """
        执行抓取
        
        Args:
            use_policy: 是否使用ACT策略
        """
        if use_policy and self.policy is not None:
            self._execute_policy_pick()
        else:
            print("\n=== 手动抓取模式 ===")
            print("请手动控制机器人完成抓取:")
            print("1. 移动到物体上方")
            print("2. 下降到抓取位置")
            print("3. 闭合夹爪")
            print("4. 抬起")
            input("\n抓取完成后按 Enter 继续...")
    
    def move_to_place_position(self, use_policy: bool = True):
        """移动到放置位置上方"""
        if use_policy and self.policy is not None:
            self._execute_policy_move_to_place()
        else:
            print("\n=== 手动移动模式 ===")
            print("请手动控制机器人移动到放置位置上方")
            input("到位后按 Enter 继续...")
    
    def place_with_precision(self, tolerance_mm: float = 2.0) -> bool:
        """
        精准放置
        
        执行视觉伺服精调，然后放置
        
        Args:
            tolerance_mm: 目标精度
        
        Returns:
            是否成功达到目标精度
        """
        print("\n" + "="*60)
        print(f"开始精准放置 (目标精度: {tolerance_mm}mm)")
        print("="*60)
        
        servo = self.get_active_servo()
        
        # 1. 视觉伺服精调XY位置
        print("\n[步骤1] 视觉伺服精调")
        success = servo.servo_to_target(tolerance_mm=tolerance_mm)
        
        if not success:
            print("警告: 未能达到目标精度，继续执行放置...")
        
        # 2. 下降到放置高度
        print("\n[步骤2] 下降到放置高度")
        self._descend_to_place()
        
        # 3. 松开夹爪
        print("\n[步骤3] 松开夹爪")
        servo.joint_controller.open_gripper()
        
        # 4. 抬起
        print("\n[步骤4] 抬起")
        self._ascend_after_place()
        
        print("\n" + "="*60)
        print("放置完成!")
        print("="*60)
        
        return success
    
    def run_full_sequence(self, tolerance_mm: float = 2.0, 
                          use_policy: bool = True) -> bool:
        """
        执行完整的抓放流程
        
        Args:
            tolerance_mm: 目标精度
            use_policy: 是否使用ACT策略
        
        Returns:
            是否成功
        """
        print("\n" + "#"*60)
        print("# 开始执行抓放任务")
        print("#"*60)
        
        start_time = time.time()
        
        # 1. 抓取
        print("\n[阶段1] 执行抓取")
        self.pick(use_policy=use_policy)
        
        # 2. 移动到放置位置
        print("\n[阶段2] 移动到放置位置")
        self.move_to_place_position(use_policy=use_policy)
        
        # 3. 精准放置
        print("\n[阶段3] 精准放置")
        success = self.place_with_precision(tolerance_mm=tolerance_mm)
        
        # 统计
        elapsed = time.time() - start_time
        self._update_stats(success, elapsed)
        
        print("\n" + "#"*60)
        print(f"# 任务完成!")
        print(f"# 耗时: {elapsed:.1f}秒")
        print(f"# 结果: {'成功' if success else '未达精度'}")
        print("#"*60)
        
        return success
    
    def run_manual_mode(self, tolerance_mm: float = 2.0) -> bool:
        """手动模式运行（不使用ACT策略）"""
        return self.run_full_sequence(tolerance_mm=tolerance_mm, use_policy=False)
    
    # === 以下方法需要根据实际机器人实现 ===
    
    def _execute_policy_pick(self):
        """使用ACT策略执行抓取"""
        # TODO: 实现ACT推理
        print("使用ACT策略执行抓取...")
        raise NotImplementedError("ACT策略推理待实现")
    
    def _execute_policy_move_to_place(self):
        """使用ACT策略移动到放置位置"""
        # TODO: 实现ACT推理
        print("使用ACT策略移动到放置位置...")
        raise NotImplementedError("ACT策略推理待实现")
    
    def _descend_to_place(self):
        """下降到放置高度"""
        # 获取当前关节位置
        current = self.robot.get_observation()
        joints = np.array(current.get('observation.state', []))
        
        if len(joints) != 16:
            print("无法获取关节位置，跳过下降")
            return
        
        # 简化策略：调整关节2（肩部俯仰）来下降
        # 实际需要根据机器人运动学计算
        if self.active_arm == "right":
            joints[8] += 2.0  # 右臂关节2
        else:
            joints[1] += 2.0  # 左臂关节2
        
        self.robot.send_action({'action': joints.tolist()})
        time.sleep(0.5)
        print("已下降到放置高度")
    
    def _ascend_after_place(self):
        """放置后抬起"""
        current = self.robot.get_observation()
        joints = np.array(current.get('observation.state', []))
        
        if len(joints) != 16:
            print("无法获取关节位置，跳过抬起")
            return
        
        if self.active_arm == "right":
            joints[8] -= 5.0  # 右臂关节2
        else:
            joints[1] -= 5.0  # 左臂关节2
        
        self.robot.send_action({'action': joints.tolist()})
        time.sleep(0.5)
        print("已抬起")
    
    def _update_stats(self, success: bool, elapsed: float):
        """更新统计信息"""
        self.stats['total_runs'] += 1
        if success:
            self.stats['successful_runs'] += 1
        
        servo = self.get_active_servo()
        if servo.iteration_count > 0:
            n = self.stats['total_runs']
            self.stats['avg_iterations'] = (
                (self.stats['avg_iterations'] * (n-1) + servo.iteration_count) / n
            )
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            **self.stats,
            'success_rate': (
                self.stats['successful_runs'] / self.stats['total_runs'] * 100
                if self.stats['total_runs'] > 0 else 0
            )
        }
    
    def print_stats(self):
        """打印统计信息"""
        stats = self.get_stats()
        print("\n" + "="*50)
        print("运行统计")
        print("="*50)
        print(f"总运行次数: {stats['total_runs']}")
        print(f"成功次数: {stats['successful_runs']}")
        print(f"成功率: {stats['success_rate']:.1f}%")
        print(f"平均迭代次数: {stats['avg_iterations']:.1f}")
        print("="*50)


# === 便捷函数 ===

def create_controller(robot_config_path: str = None, 
                      camera_indices: Dict = None,
                      policy_path: str = None) -> PrecisionPickPlaceController:
    """
    创建控制器的便捷函数
    
    Args:
        robot_config_path: 机器人配置文件路径
        camera_indices: 相机索引 {'head': 0, 'left_wrist': 1, 'right_wrist': 2}
        policy_path: ACT策略路径
    
    Returns:
        配置好的控制器实例
    """
    # 导入必要的类
    from lerobot.robots.supre_robot_follower import SupreRobotFollower
    from lerobot.robots.supre_robot_follower.supre_robot_follower_config import SupreRobotFollowerConfig
    from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    
    # 创建机器人
    if robot_config_path:
        config = SupreRobotFollowerConfig.from_yaml(robot_config_path)
    else:
        config = SupreRobotFollowerConfig()
    
    robot = SupreRobotFollower(config)
    
    # 创建相机
    cameras = {}
    if camera_indices:
        for name, idx in camera_indices.items():
            cam_config = OpenCVCameraConfig(
                index_or_path=idx,
                fps=30,
                width=640,
                height=480
            )
            cameras[name] = OpenCVCamera(cam_config)
    
    # 加载策略（如果提供）
    policy = None
    if policy_path:
        # TODO: 加载ACT策略
        pass
    
    return PrecisionPickPlaceController(robot, cameras, policy)
