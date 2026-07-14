#!/usr/bin/env python3
"""动作编排辅助工具 — 手动摆位 + 记录关节角度。

连接机器人后, 可以手动解除电机使能, 人工拖动机械臂到目标位置,
再重新使能, 然后按 S 记录当前位置的完整关节角度。

输出格式可直接粘贴到 YAML 的 named_positions 或 steps 中。

用法:
    python record_joint_positions.py --config configs/agv_class_ADC2.yaml
    python record_joint_positions.py --config ... --output my_positions.yaml

控制:
    D <motor_id>    解除指定电机使能 (输入索引: 0-13, 或 all)
                    解除后可以手动拖动该关节
    E <motor_id>    重新使能 (输入索引: 0-13, 或 all)
    S               记录当前位置 (输入位置名称)
    L               列出所有关节当前角度
    H               重新显示帮助
    Q               退出
"""

import argparse
import sys
import time
from pathlib import Path


def _parse_motor_input(s: str, num_joints: int) -> list[int]:
    """Parse 'all' or '0,2,5' into list of motor indices."""
    s = s.strip().lower()
    if s == "all":
        return list(range(num_joints))
    indices = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            idx = int(part)
            if 0 <= idx < num_joints:
                indices.append(idx)
            else:
                print(f"  ⚠ 索引 {idx} 超出范围 (0-{num_joints-1}), 已跳过")
        except ValueError:
            print(f"  ⚠ '{part}' 不是有效数字, 已跳过")
    return indices


def main():
    parser = argparse.ArgumentParser(description="Manual joint position recorder")
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("--output", default=None, help="Output YAML file (append mode)")
    parser.add_argument("--no-camera", action="store_true", help="Skip camera init")
    args = parser.parse_args()

    from lerobot.tasks.config import load_config_from_yaml
    from lerobot.robots import make_robot_from_config

    print(f"Loading config: {args.config}")
    cfg = load_config_from_yaml(args.config)
    if args.no_camera and hasattr(cfg.robot_config, 'cameras'):
        cfg.robot_config.cameras = {}
    robot = make_robot_from_config(cfg.robot_config)
    robot.connect()
    print("Robot connected.\n")

    joint_names = robot.observation_joint_names
    num_joints = len(joint_names)

    _print_help(joint_names)

    # Track disabled motors
    disabled = set()

    while True:
        try:
            cmd_line = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break

        if not cmd_line:
            continue

        parts = cmd_line.split(maxsplit=1)
        cmd = parts[0].lower()

        if cmd == 'q':
            # Re-enable all before quit
            if disabled:
                print("  重新使能所有电机...")
                _set_motors(robot, list(disabled), True)
            break

        elif cmd == 'h':
            _print_help(joint_names)

        elif cmd == 'l':
            _show_positions(robot, joint_names, disabled)

        elif cmd == 'd':
            if len(parts) < 2:
                print("  用法: D <索引> 或 D all")
                continue
            indices = _parse_motor_input(parts[1], num_joints)
            if not indices:
                continue
            ok, fail = _set_motors(robot, indices, False)
            disabled.update(ok)
            if ok:
                names = [joint_names[i] for i in ok]
                print(f"  已解除使能: {', '.join(names)}")
            print(f"  当前解除总数: {len(disabled)}/{num_joints}")

        elif cmd == 'e':
            if len(parts) < 2:
                print("  用法: E <索引> 或 E all")
                continue
            indices = _parse_motor_input(parts[1], num_joints)
            if not indices:
                continue
            ok, fail = _set_motors(robot, indices, True)
            disabled.difference_update(ok)
            if ok:
                names = [joint_names[i] for i in ok]
                print(f"  已重新使能: {', '.join(names)}")
            print(f"  当前解除总数: {len(disabled)}/{num_joints}")

        elif cmd == 's':
            if disabled:
                print(f"  ⚠ 仍有 {len(disabled)} 个电机未使能: "
                      f"{', '.join(joint_names[i] for i in sorted(disabled))}")
                ans = input("  是否继续记录? (y/n): ").strip().lower()
                if ans != 'y':
                    continue

            pos_name = input("  位置名称 (如 place_B_step1): ").strip()
            if not pos_name:
                print("  名称不能为空, 已取消。")
                continue

            positions = _get_positions(robot, joint_names)
            # Show what was recorded
            print(f"\n  ┌─ {pos_name}")
            for j in joint_names:
                print(f"  │  {j}: {positions[j]:.1f}")
            print(f"  └{'─'*40}")

            # Write to output file
            _save_position(pos_name, positions, args.output)

        else:
            print(f"  未知命令: '{cmd}'.  输入 H 查看帮助。")


def _print_help(joint_names):
    print(f"\n{'='*60}")
    print(f"  关节列表:")
    for i, name in enumerate(joint_names):
        print(f"    [{i:>2}] {name}")
    print(f"\n  命令:")
    print(f"    D <索引>      解除使能 (如: D 5  或  D 0,2,5  或  D all)")
    print(f"    E <索引>      重新使能")
    print(f"    L              列出所有关节当前角度")
    print(f"    S              记录当前位置 → 输出 YAML")
    print(f"    H              显示此帮助")
    print(f"    Q              退出 (自动重新使能所有电机)")
    print(f"{'='*60}")


def _show_positions(robot, joint_names, disabled):
    positions = _get_positions(robot, joint_names)
    print(f"\n  当前关节角度 (°):")
    for name in joint_names:
        mark = " ⚡(解除)" if joint_names.index(name) in disabled else ""
        print(f"    {name:<24} {positions[name]:>8.2f}°{mark}")


def _get_positions(robot, joint_names):
    """Return {joint_name: angle_deg}."""
    pos = robot.get_current_position()
    return {j: round(pos.get(j, 0), 1) for j in joint_names}


def _set_motors(robot, indices, enable):
    """Enable or disable specific motors by index.
    Returns (ok_indices, failed_indices)."""
    hw = robot._hardware_manager
    if hw is None:
        print("  ✗ 硬件管理器未初始化")
        return [], indices

    # Find the EyouMotorHardware instance (arm_motors) in the manager
    motor_hw = None
    for inst in hw._hardware_instances:
        if hasattr(inst, 'motor_nodes_'):
            motor_hw = inst
            break
    if motor_hw is None:
        print("  ✗ 找不到电机硬件实例 (EyouMotorHardware)")
        return

    # Build observation_index → motor_index map.
    # Gripper (joint_7) is in observation_joint_names but NOT in motor_nodes_.
    # All indices after gripper need to be shifted down by 1.
    obs_names = robot.observation_joint_names
    motor_idx = 0
    obs_to_motor = {}
    for obs_i, name in enumerate(obs_names):
        if "joint_7" in name or "gripper" in name.lower():
            obs_to_motor[obs_i] = -1  # no motor backing
        else:
            obs_to_motor[obs_i] = motor_idx
            motor_idx += 1

    ok_idx = []
    fail_idx = []
    for idx in indices:
        mi = obs_to_motor.get(idx, -1)
        if mi < 0:
            print(f"  ⚠ {obs_names[idx]} 无独立电机 (可能为夹爪), 跳过")
            continue
        try:
            motor = motor_hw.motor_nodes_[mi]
            if enable:
                motor.clear_fault()
                motor.configure_csp_mode(0, False)
                motor.start_auto_feedback(0, 255, 20)
                print(f"  ✓ {obs_names[idx]} 已重新使能 (CSP)")
                ok_idx.append(idx)
            else:
                # CSP-enabled motors may reject direct disable().
                # Clear fault first, then try again.
                ret = motor.disable()
                if not ret:
                    motor.clear_fault()
                    ret = motor.disable()
                if ret:
                    print(f"  ✓ {obs_names[idx]} 已解除使能")
                    ok_idx.append(idx)
                else:
                    print(f"  ✗ {obs_names[idx]} 解除失败 (电机拒绝指令)")
                    fail_idx.append(idx)
        except Exception as e:
            print(f"  ✗ {obs_names[idx]} 操作失败: {e}")
            fail_idx.append(idx)

    if ok_idx:
        action = "使能" if enable else "解除"
        names = ", ".join(obs_names[i] for i in ok_idx)
        print(f"  {action}成功: {names}")
    if fail_idx:
        print(f"  失败: {', '.join(obs_names[i] for i in fail_idx)}")
    return ok_idx, fail_idx


def _save_position(pos_name, positions, output_path):
    """Write to YAML file or print to stdout."""
    lines = []
    lines.append(f"  {pos_name}:")
    for j, val in sorted(positions.items()):
        lines.append(f"    {j}: {val}")
    yaml_str = "\n".join(lines)

    if output_path:
        # Check if position already exists and ask
        if Path(output_path).exists():
            content = Path(output_path).read_text()
            if pos_name + ":" in content:
                ans = input(f"  ⚠ '{pos_name}' 已存在于 {output_path}, 是否覆盖? (y/n): ").strip().lower()
                if ans != 'y':
                    print("  已跳过。")
                    return
                # Remove old entry
                import re
                pattern = re.compile(rf"  {re.escape(pos_name)}:.*?(?=\n  \w+:|$)", re.DOTALL)
                content = pattern.sub("", content)
                Path(output_path).write_text(content.rstrip() + "\n")

        with open(output_path, 'a') as f:
            f.write(yaml_str + "\n\n")
        print(f"  → 已写入 {output_path}")
    else:
        print(f"\n  ── 复制以下内容到 YAML ──")
        print(yaml_str)
        print(f"  ────────────────────────────")


if __name__ == "__main__":
    main()
