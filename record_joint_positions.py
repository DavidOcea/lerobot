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
    print(f"                  解除后会自动松抱闸以便拖动；重力关节请先扶稳")
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


def _statusword_state(sw: int) -> str:
    """解码 CiA 402 状态字 (0x6041) 为可读的驱动状态。

    诊断用：disable() 返回 True 只代表 SDO 写控制字成功，不代表电机真的
    退出使能。真正是否断电要看状态字低字节。关键值：
      0x27 = Operation enabled (仍在使能/抱闸)
      0x21 = Ready to switch on (已软关断)
      0x40 = Switch on disabled (真·断电)
    """
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


# 抱闸批次缓存。同一电机的批次在整个会话内不变，只探测一次，避免每次
# D/E 都对新电机触发一次 "Generic Read failed with code 7" 的 SDO 读超时噪声。
_BRAKE_BATCH_CACHE: dict = {}


def _brake_batch(motor) -> str:
    """探测电机批次并缓存。

    两批电机的抱闸对象地址不同：
      老电机 (joint_1~5 / trunk, node 1,2,11~15,21~25)：
          抱闸控制 0x2014:01 (写 1=松开 / 0=抱死)，状态 0x2014:02
      新电机 PHU&RHU (joint_6/7, node 16,17,26,27)：
          抱闸模式 0x2110:00 (0=手动 / 1=自动跟随使能[默认])
          抱闸输出 0x2111:00 (写 1=闭合 / 0=松开)
    判别依据：读 0x2014:02 成功=老电机；抛异常(SDO 读超时/码7)=新电机。
    """
    key = id(motor)
    if key not in _BRAKE_BATCH_CACHE:
        try:
            motor.read_u8(0x2014, 2)
            _BRAKE_BATCH_CACHE[key] = "old"
        except Exception:
            _BRAKE_BATCH_CACHE[key] = "new"
    return _BRAKE_BATCH_CACHE[key]


def _release_brake(motor, batch: str) -> str:
    """失能后主动松抱闸，返回人类可读结果。

    两批电机抱闸控制模型不同：
      新电机：0x2110=0 切"手动模式"后固件撒手不管，写 0x2111=0 立即生效。
      老电机：只写控制字 0x2014:01=1。0x2014:02 是只读状态字（读=抱闸状态
              0/1/2），写它会被驱动忽略并 SDO 超时（code 8 = NoRespondW）。
    老电机写法与 verify_brake.py 完全一致 —— 0420_2.log 实测单写 0x2014:01=1
    （状态字 2→1）即可让关节可拖动。不再做"等固件写回 0"或长时轮询：那些
    基于未证实假设，且实证关节反而转不动。
    """
    if batch == "new":
        ok = bool(motor.write_u8(0x2110, 0, 0))              # 切手动模式
        ok = bool(motor.write_u8(0x2111, 0, 0)) and ok       # 松开
        return "已松开" if ok else "写入失败"

    # 老电机：单写控制字 0x2014:01=1，读回一次确认（与 verify_brake.py 同）。
    motor.write_u8(0x2014, 1, 1)
    try:
        c = int(motor.read_u8(0x2014, 1))
    except Exception:
        c = -1
    try:
        s = int(motor.read_u8(0x2014, 2))
    except Exception:
        s = -1
    if c == 1:
        return f"已松开(ctrl={c}, state={s})"
    return f"仍抱死(ctrl={c}, state={s})"


def _restore_brake(motor, batch: str) -> None:
    """重新使能前恢复抱闸到自动跟随(安全兜底)。

    新电机 0x2110/0x2111 是 Backup=YES，停留在"手动松抱闸"状态会掉电残留，
    使失电抱闸永久失效，必须写回 0x2110=1(自动跟随使能)。
    """
    if batch == "old":
        motor.write_u8(0x2014, 1, 0)          # 交还驱动自动管理(闭合抱闸)
    else:
        motor.write_u8(0x2111, 0, 1)          # 先闭合抱闸
        motor.write_u8(0x2110, 0, 1)          # 恢复自动跟随使能(默认)


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

    # Build observation_index → motor_index map by NAME, using the hardware
    # instance's own joint_names_ (the joints that actually have a motor).
    # 不硬编码 "joint_7=夹爪"：当前 18 关节配置里 joint_7 是真实电机(node 27)。
    # 用名字匹配，夹爪(若存在)自然落到 -1，joint_8 也不会再错位到 node 27。
    obs_names = robot.observation_joint_names
    motor_names = getattr(motor_hw, 'joint_names_', None) or list(obs_names)
    obs_to_motor = {}
    for obs_i, name in enumerate(obs_names):
        if name in motor_names:
            obs_to_motor[obs_i] = motor_names.index(name)
        else:
            obs_to_motor[obs_i] = -1  # no motor backing (gripper etc.)

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
                # 重新使能前先恢复抱闸到自动跟随。新电机 0x2110 是 Backup=YES，
                # 若停留在手动松抱闸状态，掉电后失电抱闸会永久失效。
                _restore_brake(motor, _brake_batch(motor))
                motor.clear_fault()
                motor.configure_csp_mode(0, False)
                motor.start_auto_feedback(0, 255, 20)
                print(f"  ✓ {obs_names[idx]} 已重新使能 (CSP)")
                ok_idx.append(idx)
            else:
                # 失能 = 写控制字 0x06 (Shutdown) → 状态 2 "Ready to switch on"，
                # 功率级关断。但两批电机都带失电抱闸，失能即自动抱死 → "转不动"。
                # 所以失能后必须主动松抱闸，才能手动拖动。
                ret = motor.disable()
                if not ret:
                    motor.clear_fault()
                    ret = motor.disable()
                # disable() 只写控制字 0x06，不验证状态机是否真退出使能。实测
                # 出现过 0x0237/Operation enabled（写成功但仍在使能），此时伺服
                # 仍在闭环保位，抱闸怎么松都转不动。故补验状态字并重试。
                for _ in range(3):
                    try:
                        sw = int(motor.get_status_word())
                    except Exception:
                        break
                    if (sw & 0x006F) != 0x0027:  # 非 Operation enabled = 已退出使能
                        break
                    time.sleep(0.1)
                    motor.disable()
                try:
                    sw = int(motor.get_status_word())
                    state = f"0x{sw:04X}/{_statusword_state(sw)}"
                except Exception as e:
                    state = f"readback-failed({e})"
                batch = _brake_batch(motor)
                if ret:
                    note = _release_brake(motor, batch)
                    tag = "老电机(0x2014)" if batch == "old" else "新电机(0x2110/0x2111)"
                    print(f"  ✓ {obs_names[idx]} 已解除使能 → {state} | 抱闸 {note} [{tag}]")
                    if "抱死" in note or "写入失败" in note:
                        print(f"    ⚠ 抱闸未真正松开，关节仍不可拖动")
                    else:
                        print(f"    ⚠ 关节现可拖动 — 重力关节请扶稳")
                    ok_idx.append(idx)
                else:
                    print(f"  ✗ {obs_names[idx]} 解除失败 (电机拒绝指令) → {state}")
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
