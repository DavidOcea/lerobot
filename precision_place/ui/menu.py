"""
菜单系统 (Menu System)

提供基础的菜单功能。
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Callable


class MenuItem:
    """菜单项"""

    def __init__(self, key: str, label: str, callback: Callable = None):
        """
        初始化菜单项

        Args:
            key: 按键
            label: 显示标签
            callback: 回调函数
        """
        self.key = key
        self.label = label
        self.callback = callback


class MenuBase(ABC):
    """菜单基类"""

    def __init__(self, title: str = "Menu"):
        """
        初始化菜单

        Args:
            title: 菜单标题
        """
        self.title = title
        self.items: List[MenuItem] = []

    def add_item(self, key: str, label: str, callback: Callable = None):
        """
        添加菜单项

        Args:
            key: 按键
            label: 显示标签
            callback: 回调函数
        """
        self.items.append(MenuItem(key, label, callback))

    def display(self):
        """显示菜单"""
        print(f"\n{'='*50}")
        print(f"{self.title}")
        print('='*50)

        for item in self.items:
            print(f"  {item.key}. {item.label}")

        print('='*50)

    def get_choice(self, prompt: str = "选项: ") -> str:
        """
        获取用户选择

        Args:
            prompt: 提示信息

        Returns:
            用户输入
        """
        return input(prompt).strip().upper()

    def run(self):
        """运行菜单"""
        while True:
            self.display()
            choice = self.get_choice()

            if choice == '0' or choice == 'Q':
                print("退出菜单")
                break

            # 查找匹配的菜单项
            for item in self.items:
                if item.key == choice:
                    if item.callback:
                        item.callback()
                    break
            else:
                print("无效选项，请重新选择")


class CalibrationMenu(MenuBase):
    """标定菜单"""

    def __init__(self, system):
        """
        初始化标定菜单

        Args:
            system: PrecisionPlaceSystem实例
        """
        super().__init__("标定选项")
        self.system = system

        self.add_item("H", "手眼标定 (ChArUco板，推荐)", self._hand_eye_calibration)
        self.add_item("R", "重投影验证", self._reprojection_verification)
        self.add_item("1", "像素-毫米标定 (传统)", self._pixel_mm_calibration)
        self.add_item("2", "XY关节灵敏度标定 (手动)", self._xy_sensitivity_manual)

    def _hand_eye_calibration(self):
        if hasattr(self.system, 'hand_eye_calibration'):
            self.system.hand_eye_calibration()

    def _reprojection_verification(self):
        if hasattr(self.system, 'reprojection_verification'):
            self.system.reprojection_verification()

    def _pixel_mm_calibration(self):
        print("像素-毫米标定...")

    def _xy_sensitivity_manual(self):
        print("XY关节灵敏度标定...")