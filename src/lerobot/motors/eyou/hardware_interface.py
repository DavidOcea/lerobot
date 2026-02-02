from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional


class HardwareInterface(ABC):
    """
    通用硬件接口，定义了与高层控制器交互的标准方法。
    """
    @abstractmethod
    def init(self, config: Dict[str, Any]) -> bool:
        """根据配置初始化硬件。"""
        pass    
    @abstractmethod
    def get_joint_count(self) -> int:
        """返回硬件管理的关节数量。"""
        pass

    @abstractmethod
    def read(self) -> List[Tuple[Optional[float], Optional[float]]]:
        """从硬件读取所有关节的当前位置。"""
        pass
    @abstractmethod
    def write(self, commands_positions: List[float]):
        """向硬件发送目标位置指令。"""
        pass

    @abstractmethod
    def activate(self) -> bool:
        """激活硬件（例如，使能电机）。"""
        pass

    @abstractmethod
    def deactivate(self):
        """停用硬件。"""
        pass