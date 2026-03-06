"""
Joint Controller - 关节微调控制器

负责将像素偏移转换为关节角度调整量
"""

import numpy as np
from typing import Dict, Optional
import yaml
from pathlib import Path
import time


class JointController:
    """
    关节微调控制器
    
    将视觉检测到的像素偏移转换为机器人关节角度调整量。
    由于机器人只支持关节控制，需要建立像素偏移到关节调整的映射。
    """
    
    def __init__(self, robot, config_path: str = None, arm: str = "right"):
        """
        Args:
            robot: SupreRobotFollower 实例
            config_path: 配置文件路径
            arm: 使用的手臂 ("left" or "right")
        """
        self.robot = robot
        self.arm = arm
        self.config = self._load_config(config_path)
        
        # 关节灵敏度配置
        self.joint_sensitivity = self._get_joint_sensitivity()
        
        # 安全参数
        self.max_single_adjustment = self.config.get('max_single_adjustment', 3.0)
        self.min_adjustment = self.config.get('min_adjustment', 0.05)
        
        # 像素到毫米转换
        self.pixel_to_mm_ratio = 0.5  # 默认值，需要标定
        
    def _load_config(self, config_path: str) -> dict:
        """加载配置"""
        if config_path is None:
            config_path = Path(__file__).parent / "configs" / "precision_config.yaml"
        
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config.get('joint_controller', {})
        return {}
    
    def _get_joint_sensitivity(self) -> Dict[str, float]:
        """
        获取关节灵敏度配置
        
        灵敏度定义：末端移动1mm需要的关节角度变化（度）
        """
        arm_config = self.config.get(f'{self.arm}_arm', {})
        
        # 默认灵敏度值（需要根据实际机器人标定）
        default_sensitivity = {
            'joint_1': 0.15,  # 基座旋转
            'joint_2': 0.20,  # 肩部俯仰
            'joint_3': 0.30,  # 肩部旋转
            'joint_4': 0.50,  # 肘部
            'joint_5': 0.80,  # 前臂旋转
            'joint_6': 1.00,  # 腕部
        }
        
        # 合并配置
        for key in default_sensitivity:
            if key in arm_config:
                default_sensitivity[key] = arm_config[key]
        
        return default_sensitivity
    
    def set_pixel_to_mm_ratio(self, ratio: float):
        """设置像素到毫米的转换比例"""
        self.pixel_to_mm_ratio = ratio
        print(f"像素-毫米比例已设置: 1像素 = {ratio:.3f}mm")
    
    def pixel_offset_to_mm(self, offset_x: float, offset_y: float) -> tuple:
        """
        将像素偏移转换为毫米偏移
        
        注意：图像坐标系和机器人坐标系的关系需要根据相机安装确定
        """
        mm_x = offset_x * self.pixel_to_mm_ratio
        mm_y = offset_y * self.pixel_to_mm_ratio
        return mm_x, mm_y
    
    def mm_offset_to_joint_adjustment(self, mm_x: float, mm_y: float) -> np.ndarray:
        """
        将毫米偏移转换为16维关节调整量
        
        简化策略：
        - X方向偏移 → 主要调整关节1（基座旋转）
        - Y方向偏移 → 主要调整关节2（肩部俯仰）或关节3
        
        Returns:
            16维关节角度调整量
        """
        adjustment = np.zeros(16)
        
        # 限制单次调整量
        mm_x = np.clip(mm_x, -self.max_single_adjustment, self.max_single_adjustment)
        mm_y = np.clip(mm_y, -self.max_single_adjustment, self.max_single_adjustment)
        
        # 根据手臂选择关节索引
        if self.arm == "right":
            # 右臂关节索引：7-12（关节1-6），13是夹爪
            # X方向 → 关节1（索引7）
            if abs(mm_x) > self.min_adjustment:
                adjustment[7] = -mm_x * self.joint_sensitivity['joint_1']
            
            # Y方向 → 关节2（索引8）
            if abs(mm_y) > self.min_adjustment:
                adjustment[8] = -mm_y * self.joint_sensitivity['joint_2']
                
        else:  # left
            # 左臂关节索引：0-5（关节1-6），6是夹爪
            if abs(mm_x) > self.min_adjustment:
                adjustment[0] = mm_x * self.joint_sensitivity['joint_1']  # 注意左臂可能反向
            
            if abs(mm_y) > self.min_adjustment:
                adjustment[1] = -mm_y * self.joint_sensitivity['joint_2']
        
        return adjustment
    
    def apply_adjustment(self, adjustment: np.ndarray, wait_time: float = 0.3) -> bool:
        """
        应用关节调整
        
        Args:
            adjustment: 16维关节角度调整量
            wait_time: 等待机器人稳定的时间
        
        Returns:
            是否成功执行
        """
        try:
            # 获取当前关节位置
            current_obs = self.robot.get_observation()
            current_joints = np.array(current_obs.get('observation.state', []))
            
            if len(current_joints) != 16:
                print(f"错误: 关节数量不正确 ({len(current_joints)})")
                return False
            
            # 计算新目标位置
            new_target = current_joints + adjustment
            
            # 发送到机器人
            action_dict = {'action': new_target.tolist()}
            self.robot.send_action(action_dict)
            
            # 等待机器人稳定（因为没有反馈，使用固定等待时间）
            time.sleep(wait_time)
            
            return True
            
        except Exception as e:
            print(f"应用调整失败: {e}")
            return False
    
    def adjust_xy(self, pixel_offset_x: float, pixel_offset_y: float, 
                  gain: float = 1.0, wait_time: float = 0.3) -> bool:
        """
        一步调整XY位置
        
        Args:
            pixel_offset_x: X方向像素偏移
            pixel_offset_y: Y方向像素偏移
            gain: 增益系数
            wait_time: 等待时间
        
        Returns:
            是否成功执行
        """
        # 像素 → 毫米
        mm_x, mm_y = self.pixel_offset_to_mm(pixel_offset_x, pixel_offset_y)
        
        # 应用增益
        mm_x *= gain
        mm_y *= gain
        
        # 毫米 → 关节调整
        adjustment = self.mm_offset_to_joint_adjustment(mm_x, mm_y)
        
        print(f"调整: 像素({pixel_offset_x:.1f}, {pixel_offset_y:.1f}) → "
              f"毫米({mm_x:.2f}, {mm_y:.2f})")
        
        return self.apply_adjustment(adjustment, wait_time)
    
    def get_current_joint_positions(self) -> Optional[np.ndarray]:
        """获取当前关节位置"""
        try:
            obs = self.robot.get_observation()
            return np.array(obs.get('observation.state', []))
        except Exception as e:
            print(f"获取关节位置失败: {e}")
            return None
    
    def open_gripper(self, wait_time: float = 0.5) -> bool:
        """打开夹爪"""
        try:
            current = self.get_current_joint_positions()
            if current is None:
                return False
            
            # 夹爪开合位置
            if self.arm == "right":
                current[13] = 0.0  # 右臂夹爪
            else:
                current[6] = 0.0   # 左臂夹爪
            
            self.robot.send_action({'action': current.tolist()})
            time.sleep(wait_time)
            return True
            
        except Exception as e:
            print(f"打开夹爪失败: {e}")
            return False
    
    def close_gripper(self, position: float = 50.0, wait_time: float = 0.5) -> bool:
        """闭合夹爪"""
        try:
            current = self.get_current_joint_positions()
            if current is None:
                return False
            
            if self.arm == "right":
                current[13] = position  # 右臂夹爪
            else:
                current[6] = position   # 左臂夹爪
            
            self.robot.send_action({'action': current.tolist()})
            time.sleep(wait_time)
            return True
            
        except Exception as e:
            print(f"闭合夹爪失败: {e}")
            return False
