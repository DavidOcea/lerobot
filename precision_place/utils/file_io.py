"""
文件读写工具 (File I/O Utilities)

提供配置文件和数据文件的读写功能。
"""

import json
import yaml
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional


def load_yaml(filepath: str) -> Optional[Dict]:
    """
    加载YAML文件

    Args:
        filepath: 文件路径

    Returns:
        解析后的字典，失败返回None
    """
    filepath = Path(filepath)
    if not filepath.exists():
        return None

    try:
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"加载YAML失败: {e}")
        return None


def save_yaml(data: Dict, filepath: str) -> bool:
    """
    保存到YAML文件

    Args:
        data: 数据字典
        filepath: 文件路径

    Returns:
        是否成功
    """
    try:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        return True
    except Exception as e:
        print(f"保存YAML失败: {e}")
        return False


def load_json(filepath: str) -> Optional[Dict]:
    """
    加载JSON文件

    Args:
        filepath: 文件路径

    Returns:
        解析后的字典，失败返回None
    """
    filepath = Path(filepath)
    if not filepath.exists():
        return None

    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载JSON失败: {e}")
        return None


def save_json(data: Any, filepath: str, indent: int = 2) -> bool:
    """
    保存到JSON文件

    Args:
        data: 数据
        filepath: 文件路径
        indent: 缩进

    Returns:
        是否成功
    """
    try:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=indent, default=str)
        return True
    except Exception as e:
        print(f"保存JSON失败: {e}")
        return False


def load_numpy_array(filepath: str) -> Optional[np.ndarray]:
    """
    加载numpy数组

    Args:
        filepath: 文件路径 (.npy或.npz)

    Returns:
        numpy数组
    """
    filepath = Path(filepath)
    if not filepath.exists():
        return None

    try:
        return np.load(str(filepath))
    except Exception as e:
        print(f"加载numpy数组失败: {e}")
        return None


def save_numpy_array(array: np.ndarray, filepath: str) -> bool:
    """
    保存numpy数组

    Args:
        array: numpy数组
        filepath: 文件路径

    Returns:
        是否成功
    """
    try:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(filepath), array)
        return True
    except Exception as e:
        print(f"保存numpy数组失败: {e}")
        return False