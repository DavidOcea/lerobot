"""
机器人状态共享模块

用于示教程序和标定程序之间的跨进程通信。
"""

from precision_place.robot.status import (
    RobotStatusWriter,
    RobotStatusReader,
    joints_dict_to_array,
    JOINT_NAME_TO_INDEX,
    STATUS_FILE
)

__all__ = [
    'RobotStatusWriter',
    'RobotStatusReader',
    'joints_dict_to_array',
    'JOINT_NAME_TO_INDEX',
    'STATUS_FILE'
]