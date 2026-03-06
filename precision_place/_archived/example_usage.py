#!/usr/bin/env python3
"""
Example Usage - 使用示例

演示如何使用精准放置模块
"""

import sys
import time
import cv2
import numpy as np
from pathlib import Path

# 添加lerobot路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lerobot.robots.supre_robot_follower import SupreRobotFollower
from lerobot.robots.supre_robot_follower.supre_robot_follower_config import SupreRobotFollowerConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

from precision_place import (
    PrecisionPickPlaceController,
    WristCameraVisualServo,
    SlotDetector,
    JointController
)
from precision_place.calibration_tool import CalibrationTool


def setup_robot_and_cameras():
    """设置机器人和相机"""
    
    # 1. 创建机器人配置
    config = SupreRobotFollowerConfig()
    
    # 2. 创建机器人实例
    robot = SupreRobotFollower(config)
    
    # 3. 创建相机
    # 注意：需要根据实际情况修改相机索引
    cameras = {
        'head': OpenCVCamera(OpenCVCameraConfig(
            index_or_path=0,  # 修改为实际的相机索引
            fps=30,
            width=640,
            height=480
        )),
        'left_wrist': OpenCVCamera(OpenCVCameraConfig(
            index_or_path=1,  # 修改为实际的相机索引
            fps=30,
            width=640,
            height=480
        )),
        'right_wrist': OpenCVCamera(OpenCVCameraConfig(
            index_or_path=2,  # 修改为实际的相机索引
            fps=30,
            width=640,
            height=480
        ))
    }
    
    return robot, cameras


def example_1_calibration():
    """
    示例1: 标定流程
    
    首次使用时需要运行此流程
    """
    print("\n" + "="*60)
    print("示例1: 标定流程")
    print("="*60)
    
    robot, cameras = setup_robot_and_cameras()
    
    # 连接
    robot.connect()
    cameras['right_wrist'].connect()
    
    # 创建标定工具
    calibrator = CalibrationTool(robot, cameras['right_wrist'], arm="right")
    
    # 运行完整标定
    calibrator.run_full_calibration()
    
    # 断开连接
    robot.disconnect()
    cameras['right_wrist'].disconnect()


def example_2_manual_mode():
    """
    示例2: 手动模式抓放
    
    不使用ACT策略，完全手动控制
    """
    print("\n" + "="*60)
    print("示例2: 手动模式精准抓放")
    print("="*60)
    
    robot, cameras = setup_robot_and_cameras()
    
    # 创建控制器
    controller = PrecisionPickPlaceController(
        robot=robot,
        cameras=cameras,
        policy=None  # 不使用ACT策略
    )
    
    # 连接
    controller.connect()
    
    # 加载标定结果
    controller.right_servo.load_calibration()
    controller.right_servo.load_target_template()
    
    # 运行手动模式
    success = controller.run_manual_mode(tolerance_mm=2.0)
    
    # 打印统计
    controller.print_stats()
    
    # 断开连接
    controller.disconnect()


def example_3_with_act_policy():
    """
    示例3: 使用ACT策略 + 视觉伺服
    
    完整的混合方案
    """
    print("\n" + "="*60)
    print("示例3: ACT策略 + 视觉伺服")
    print("="*60)
    
    robot, cameras = setup_robot_and_cameras()
    
    # TODO: 加载训练好的ACT策略
    policy = None  # load_act_policy("path/to/policy")
    
    # 创建控制器
    controller = PrecisionPickPlaceController(
        robot=robot,
        cameras=cameras,
        policy=policy
    )
    
    # 连接
    controller.connect()
    
    # 加载标定结果
    controller.right_servo.load_calibration()
    controller.right_servo.load_target_template()
    
    # 运行完整流程
    success = controller.run_full_sequence(
        tolerance_mm=2.0,
        use_policy=True
    )
    
    # 打印统计
    controller.print_stats()
    
    # 断开连接
    controller.disconnect()


def example_4_visual_servo_only():
    """
    示例4: 仅使用视觉伺服
    
    手动移动到放置位置附近，然后执行视觉伺服
    """
    print("\n" + "="*60)
    print("示例4: 仅视觉伺服")
    print("="*60)
    
    robot, cameras = setup_robot_and_cameras()
    
    # 连接
    robot.connect()
    cameras['right_wrist'].connect()
    
    # 创建视觉伺服控制器
    servo = WristCameraVisualServo(
        robot=robot,
        camera=cameras['right_wrist'],
        arm="right"
    )
    
    # 加载标定结果和模板
    servo.load_calibration()
    servo.load_target_template()
    
    print("\n请手动将机器人移动到目标位置附近（卡槽上方5-10cm）")
    input("到位后按 Enter 开始视觉伺服...")
    
    # 执行视觉伺服
    success = servo.servo_to_target(tolerance_mm=2.0)
    
    if success:
        print("\n视觉伺服成功！")
        print("现在可以手动执行放置操作")
    
    # 断开连接
    robot.disconnect()
    cameras['right_wrist'].disconnect()


def example_5_continuous_mode():
    """
    示例5: 连续运行模式
    
    循环执行抓放任务
    """
    print("\n" + "="*60)
    print("示例5: 连续运行模式")
    print("="*60)
    
    robot, cameras = setup_robot_and_cameras()
    
    # 创建控制器
    controller = PrecisionPickPlaceController(
        robot=robot,
        cameras=cameras,
        policy=None
    )
    
    # 连接
    controller.connect()
    
    # 加载标定结果
    controller.right_servo.load_calibration()
    controller.right_servo.load_target_template()
    
    # 连续运行
    num_runs = 10
    for i in range(num_runs):
        print(f"\n{'#'*60}")
        print(f"# 运行 {i+1}/{num_runs}")
        print(f"{'#'*60}")
        
        success = controller.run_manual_mode(tolerance_mm=2.0)
        
        # 等待下一次
        if i < num_runs - 1:
            print("\n请准备好下一次抓取")
            input("准备好后按 Enter 继续...")
    
    # 打印统计
    controller.print_stats()
    
    # 断开连接
    controller.disconnect()


def main():
    """主函数"""
    print("\n" + "="*60)
    print("精准放置模块 - 使用示例")
    print("="*60)
    
    print("\n请选择要运行的示例:")
    print("1. 标定流程（首次使用）")
    print("2. 手动模式抓放")
    print("3. ACT策略 + 视觉伺服（需要训练好的模型）")
    print("4. 仅视觉伺服")
    print("5. 连续运行模式")
    print("0. 退出")
    
    choice = input("\n请输入选项: ").strip()
    
    if choice == "1":
        example_1_calibration()
    elif choice == "2":
        example_2_manual_mode()
    elif choice == "3":
        example_3_with_act_policy()
    elif choice == "4":
        example_4_visual_servo_only()
    elif choice == "5":
        example_5_continuous_mode()
    elif choice == "0":
        print("退出")
    else:
        print("无效选项")


if __name__ == "__main__":
    main()
