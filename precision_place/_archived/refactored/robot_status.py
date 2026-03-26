"""
机器人状态共享模块

通过文件实现跨进程的机器人状态共享，用于示教程序和标定程序之间的通信。

使用方法:
    # 示教程序 (写入)
    from precision_place.robot_status import RobotStatusWriter
    writer = RobotStatusWriter()
    writer.write(observation)

    # 标定程序 (读取)
    from precision_place.robot_status import RobotStatusReader
    reader = RobotStatusReader()
    joints = reader.read_joints()
"""

import json
import os
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any


# 状态文件路径
STATUS_FILE = Path(__file__).parent / "shared_robot_status.json"
TEMP_FILE = Path(__file__).parent / "shared_robot_status.tmp"

# 默认最大数据年龄 (毫秒)
DEFAULT_MAX_AGE_MS = 200


class RobotStatusWriter:
    """机器人状态写入器 (示教程序使用)"""

    def __init__(self, status_file: Path = None):
        self.status_file = status_file or STATUS_FILE
        self.temp_file = self.status_file.with_suffix('.tmp')
        self._write_count = 0

    def write(self, observation: Dict[str, Any]) -> bool:
        """
        原子写入机器人状态

        Args:
            observation: 机器人观测数据字典

        Returns:
            是否写入成功
        """
        try:
            # 提取关节位置
            joints = {}
            for key, value in observation.items():
                if key.endswith('.pos') and not key.startswith('images'):
                    joint_name = key.replace('.pos', '')
                    if isinstance(value, (int, float)):
                        joints[joint_name] = float(value)
                    elif hasattr(value, 'item'):  # numpy
                        joints[joint_name] = float(value.item())

            # 构建状态数据
            data = {
                "timestamp": time.time(),
                "write_count": self._write_count,
                "joints": joints,
                "joint_list": list(joints.values()),  # 有序列表
            }

            # 原子写入: 先写临时文件，再 rename
            with open(self.temp_file, 'w') as f:
                json.dump(data, f, indent=2)

            os.rename(self.temp_file, self.status_file)
            self._write_count += 1
            return True

        except Exception as e:
            print(f"⚠ 写入状态失败: {e}")
            return False

    def cleanup(self):
        """清理状态文件"""
        try:
            if self.status_file.exists():
                os.remove(self.status_file)
            if self.temp_file.exists():
                os.remove(self.temp_file)
        except:
            pass


class RobotStatusReader:
    """机器人状态读取器 (标定程序使用)"""

    def __init__(self, status_file: Path = None):
        self.status_file = status_file or STATUS_FILE
        self._last_count = -1

    def read(self, max_age_ms: float = DEFAULT_MAX_AGE_MS) -> Optional[Dict]:
        """
        读取机器人状态

        Args:
            max_age_ms: 最大数据年龄 (毫秒)，超过此时间的数据视为过期

        Returns:
            状态字典，如果数据无效则返回 None
        """
        try:
            if not self.status_file.exists():
                return None

            with open(self.status_file, 'r') as f:
                data = json.load(f)

            # 检查数据年龄
            age_ms = (time.time() - data.get("timestamp", 0)) * 1000
            if age_ms > max_age_ms:
                return None

            return data

        except (json.JSONDecodeError, FileNotFoundError, KeyError):
            return None

    def read_joints(self, max_age_ms: float = DEFAULT_MAX_AGE_MS) -> Optional[Dict[str, float]]:
        """
        读取关节位置字典

        Returns:
            {joint_name: position} 字典，如果数据无效则返回 None
        """
        data = self.read(max_age_ms)
        if data is None:
            return None
        return data.get("joints", {})

    def read_joint_array(self, max_age_ms: float = DEFAULT_MAX_AGE_MS) -> Optional[np.ndarray]:
        """
        读取关节位置数组 (有序)

        Returns:
            关节位置 numpy 数组，如果数据无效则返回 None
        """
        data = self.read(max_age_ms)
        if data is None:
            return None

        joint_list = data.get("joint_list", [])
        if not joint_list:
            return None

        return np.array(joint_list)

    def get_status_info(self) -> Dict[str, Any]:
        """获取状态文件信息 (用于调试)"""
        try:
            if not self.status_file.exists():
                return {"exists": False, "error": "文件不存在"}

            with open(self.status_file, 'r') as f:
                data = json.load(f)

            age_ms = (time.time() - data.get("timestamp", 0)) * 1000
            return {
                "exists": True,
                "age_ms": age_ms,
                "write_count": data.get("write_count", -1),
                "joint_count": len(data.get("joints", {})),
                "is_fresh": age_ms < DEFAULT_MAX_AGE_MS
            }
        except Exception as e:
            return {"exists": False, "error": str(e)}

    def wait_for_fresh_data(self, timeout_s: float = 5.0) -> bool:
        """
        等待新鲜数据

        Args:
            timeout_s: 超时时间 (秒)

        Returns:
            是否在超时前获得新鲜数据
        """
        start = time.time()
        while time.time() - start < timeout_s:
            data = self.read(max_age_ms=100)
            if data is not None:
                return True
            time.sleep(0.05)
        return False


# 关节名称到索引的映射 (与 SupreRobotFollower 配置一致)
JOINT_NAME_TO_INDEX = {
    'left_arm_joint_1': 0,
    'left_arm_joint_2': 1,
    'left_arm_joint_3': 2,
    'left_arm_joint_4': 3,
    'left_arm_joint_5': 4,
    'left_arm_joint_6': 5,
    'left_arm_joint_7': 6,  # 左夹爪
    'right_arm_joint_1': 7,
    'right_arm_joint_2': 8,
    'right_arm_joint_3': 9,
    'right_arm_joint_4': 10,
    'right_arm_joint_5': 11,
    'right_arm_joint_6': 12,
    'right_arm_joint_7': 13,  # 右夹爪
    'trunk_joint_1': 14,
    'trunk_joint_2': 15,
}

INDEX_TO_JOINT_NAME = {v: k for k, v in JOINT_NAME_TO_INDEX.items()}


def joints_dict_to_array(joints_dict: Dict[str, float]) -> np.ndarray:
    """将关节字典转换为有序数组"""
    array = np.zeros(16)
    for name, value in joints_dict.items():
        if name in JOINT_NAME_TO_INDEX:
            array[JOINT_NAME_TO_INDEX[name]] = value
    return array


def joints_array_to_dict(joints_array: np.ndarray) -> Dict[str, float]:
    """将有序数组转换为关节字典"""
    return {INDEX_TO_JOINT_NAME[i]: joints_array[i] for i in range(len(joints_array))}
