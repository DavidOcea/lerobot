#!/usr/bin/env python3
"""一次性抱闸验证脚本 — 确认「松抱闸」地址/写法在真机上有效。

背景：解除使能后电学已断电（状态字 Ready to switch on），但机械转不动，
      因为两批电机都带失电抱闸，伺服失能即自动抱死。本脚本验证主动松抱闸
      能否恢复可手动拖动。

两批电机的抱闸地址不同：
  老电机 (joint_1~5 / trunk, node 1,2,11~15,21~25):
      抱闸控制 0x2014:01   写 1 = 松开, 写 0 = 抱死
      抱闸状态 0x2014:02   读 0=抱死, 1=20V松开, 2=7V保持
  新电机 PHU&RHU (joint_6/7, node 16,17,26,27):
      抱闸模式 0x2110:00   0=手动, 1=自动跟随使能(默认)
      抱闸输出 0x2111:00   写 1=闭合抱闸, 写 0=松开抱闸

用法 (在真机 Orin 上，与 record_joint_positions.py 用同一 config):
    python verify_brake.py --config configs/agv_class_ADC2.yaml --joint 5

安全红线：松抱闸后重力关节会下坠，必须先扶住手臂再按回车测试。
"""

import argparse
import sys
import time


def _statusword_state(sw: int) -> str:
    low = sw & 0x000F
    if low == 0x0F:
        return "Fault reaction active"
    if low == 0x08:
        return "Fault"
    state = sw & 0x006F
    return {
        0x0000: "Not ready to switch on",
        0x0040: "Switch on disabled",
        0x0021: "Ready to switch on",
        0x0023: "Switched on",
        0x0027: "Operation enabled",
        0x0007: "Quick stop active",
    }.get(state, f"unknown(0x{sw:04X})")


def _find_motor(robot, obs_idx):
    """按观察索引找到对应电机对象；夹爪索引无电机返回 None。"""
    hw = robot._hardware_manager
    motor_hw = None
    for inst in hw._hardware_instances:
        if hasattr(inst, 'motor_nodes_'):
            motor_hw = inst
            break
    if motor_hw is None:
        raise RuntimeError("找不到 EyouMotorHardware 实例")

    obs_names = robot.observation_joint_names
    motor_idx = 0
    for i, name in enumerate(obs_names):
        if "joint_7" in name or "gripper" in name.lower():
            continue
        if i == obs_idx:
            return motor_hw.motor_nodes_[motor_idx], name
        motor_idx += 1
    raise RuntimeError(f"索引 {obs_idx} 无对应电机")


def _detect_batch(motor) -> str:
    """读 0x2014:02，成功=老电机，抛异常(码7/SDO读超时)=新电机。"""
    try:
        b = int(motor.read_u8(0x2014, 2))
        print(f"  [探测] 0x2014:02 可读, 值={b} → 老电机 (抱闸在 0x2014)")
        return "old"
    except Exception as e:
        print(f"  [探测] 0x2014:02 读失败 ({e}) → 新电机 PHU/RHU (抱闸在 0x2110/0x2111)")
        return "new"


def _release_brake(motor, batch: str):
    if batch == "old":
        motor.write_u8(0x2014, 1, 1)
        ctrl = int(motor.read_u8(0x2014, 1))
        st = int(motor.read_u8(0x2014, 2))
        print(f"  [松抱闸] 0x2014:01=1 写入 → 读回 ctrl={ctrl} state={st}")
    else:
        motor.write_u8(0x2110, 0, 0)   # 切手动模式
        motor.write_u8(0x2111, 0, 0)   # 松开抱闸
        mode = int(motor.read_u8(0x2110, 0))
        out = int(motor.read_u8(0x2111, 0))
        print(f"  [松抱闸] 0x2110=0 + 0x2111=0 写入 → 读回 mode={mode} output={out}")


def _engage_brake(motor, batch: str):
    if batch == "old":
        motor.write_u8(0x2014, 1, 0)
        ctrl = int(motor.read_u8(0x2014, 1))
        print(f"  [抱死] 0x2014:01=0 写入 → 读回 ctrl={ctrl}")
    else:
        motor.write_u8(0x2111, 0, 1)   # 闭合抱闸
        motor.write_u8(0x2110, 0, 1)   # 恢复自动跟随使能(默认)
        mode = int(motor.read_u8(0x2110, 0))
        out = int(motor.read_u8(0x2111, 0))
        print(f"  [抱死] 0x2111=1 + 0x2110=1 写入 → 读回 mode={mode} output={out}")


def main():
    parser = argparse.ArgumentParser(description="验证松抱闸地址/写法")
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("--joint", required=True, type=int, help="关节观察索引 (0-13)")
    args = parser.parse_args()

    from lerobot.tasks.config import load_config_from_yaml
    from lerobot.robots import make_robot_from_config

    print(f"Loading config: {args.config}")
    cfg = load_config_from_yaml(args.config)
    if hasattr(cfg.robot_config, 'cameras'):
        cfg.robot_config.cameras = {}
    robot = make_robot_from_config(cfg.robot_config)
    robot.connect()
    print("Robot connected.\n")

    motor, name = _find_motor(robot, args.joint)
    print(f"目标关节: {name} (观察索引 {args.joint})")

    batch = _detect_batch(motor)

    # 1. 解除使能（此时抱闸会自动抱死）
    ret = motor.disable()
    if not ret:
        motor.clear_fault()
        ret = motor.disable()
    sw = int(motor.get_status_word())
    print(f"  解除使能: {'成功' if ret else '失败'} → 状态字 0x{sw:04X}/{_statusword_state(sw)}")
    time.sleep(0.2)

    # 2. 松抱闸
    print("\n=== 松开抱闸 (请先扶住手臂!) ===")
    input("  扶稳手臂后按回车继续...")
    _release_brake(motor, batch)

    # 3. 物理测试
    print("\n>>> 现在试试能否手动转动该关节。<<<")
    input("  测试完毕按回车，自动重新抱死...")

    # 4. 恢复抱死
    _engage_brake(motor, batch)
    print("\n验证完成。抱闸已恢复抱死状态。")
    print("若刚才能转动 → 地址/写法确认无误，可以据此修改 record_joint_positions.py。")
    print("若仍转不动 → 贴本脚本输出回来，继续排查。")


if __name__ == "__main__":
    main()
