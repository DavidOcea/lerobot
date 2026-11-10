#!/usr/bin/env python3
"""数据标定的单臂关节镜像工具。

原理
----
机器人左右臂关节角近似满足  left_i + right_i = offset_i  （关于 offset_i/2 的反射对称），
而不是精确的 left_i = -right_i。offset_i 是"两臂机械零位偏差"（编码器零位/回零不对称），
是每台机器人的常量，可迁移到任意位姿（含单臂）。

镜像公式（反射，自身即逆）:
    mirror(x) = offset_i - x          (i = 1..7)
    mirror(x) = x                     (i = 8, 夹爪同号不取反)

标定数据源
----------
agv_class_AB1B2_boxcheck.yaml 中"对称度得分 <3°"的 50 个位姿
（对称度 = mean(|left_i + right_i|), i=1..7）。

标定后残差 std（越小越好，等价于镜像误差的典型幅度）:
    j1 0.88°  j2 2.67°  j3 0.15°  j4 2.57°  j5 0.88°  j6 1.70°  j7 2.86°
对比朴素取反 left=-right 的残差 std: 4~20°。

局限
----
反射只补偿"零位偏差"，不含 DH 非线性。要 <1° 仍需粗 DH
（见 memory: rough-dh-physical-measurement）。

用法（换机器人时走完整流水线，同台机器人可直接 --mirror-named）
----
    # 1) 标定：从新配置的对称位姿重算 offset，写入 JSON
    python scripts/mirror_arm_pose.py --calibrate /path/to/config.yaml --offset-file offsets.json

    # 2) 校验：确认新 offset 残差达标（读 offset 文件）
    python scripts/mirror_arm_pose.py --verify /path/to/config.yaml --offset-file offsets.json

    # 3) 镜像：把左臂 named_positions 镜像成右臂，追加 <name>__mirrored
    python scripts/mirror_arm_pose.py --mirror-named /path/to/config.yaml \
        --source-side left --suffix __mirrored --out /tmp/mirrored.yaml \
        --offset-file offsets.json

    # 只打印当前生效的标定参数（来源：offset 文件或内置默认）
    python scripts/mirror_arm_pose.py --show-params --offset-file offsets.json
"""

import argparse
import copy
import json
import os
import statistics
import sys

import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OFFSET_FILE = os.path.join(SCRIPT_DIR, "mirror_offsets.json")

# 内置标定（robot1, 18 关节），仅当没有 calibrate 产物时作为默认值
BUILTIN_OFFSET = {
    1: -0.824,
    2: -0.500,
    3: -0.450,
    4: -0.640,
    5: +0.934,
    6: +2.980,
    7: +0.438,
    # joint_8 夹爪：同号，不参与反射
}

GRIPPER_JOINT = 8
ARM_JOINTS = (1, 2, 3, 4, 5, 6, 7, 8)
SYMMETRY_THRESHOLD = 3.0  # 对称度得分 < 此值才视为"两臂对称"位姿

# 当前生效的 offset（main() 里会按 --offset-file / --calibrate 更新）
MIRROR_OFFSET = dict(BUILTIN_OFFSET)


def load_offsets(offset_file):
    """从 JSON 文件读标定 offset；缺失/损坏/不完整则回退内置默认。"""
    if offset_file and os.path.exists(offset_file):
        try:
            with open(offset_file) as f:
                data = json.load(f)
            off = {int(k): float(v) for k, v in data.items()}
            if set(off) == set(range(1, 8)):
                return off
            print(f"警告: {offset_file} 内容不完整，回退内置 offset", file=sys.stderr)
        except (ValueError, OSError) as e:
            print(f"警告: 读取 {offset_file} 失败({e})，回退内置 offset", file=sys.stderr)
    return dict(BUILTIN_OFFSET)


def mirror_joint_value(value, joint_index):
    """单关节镜像值。joint_8(夹爪)同号返回原值。"""
    if joint_index == GRIPPER_JOINT:
        return value
    return MIRROR_OFFSET[joint_index] - value


def mirror_single_arm(values):
    """镜像一条手臂的 8 个关节角。

    values: dict {1..8: 角度} 或 list [j1..j8]
    返回同构的镜像结果。反射关于 offset/2 对称，所以 mirror(mirror(x)) == x。
    """
    if isinstance(values, dict):
        return {i: mirror_joint_value(values[i], i) for i in ARM_JOINTS}
    return [mirror_joint_value(v, i) for i, v in enumerate(values, start=1)]


def _joint_index_from_key(key):
    """从 'left_arm_joint_3' 之类的键名解析关节号 1..8。"""
    try:
        return int(key.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return None


def mirror_named_pose(pose, source_side):
    """把一条 named_position 里 source_side 手臂镜像到另一条手臂。

    source_side='left'  => 读 left_arm_joint_*，把 mirror(left_i)=offset-left_i
                           写到 right_arm_joint_* 键上。
    源手臂与 trunk 保持原值不变；镜像关系 left+right=offset 成立。
    """
    src_prefix = f"{source_side}_arm_joint_"
    dst_prefix = "right_arm_joint_" if source_side == "left" else "left_arm_joint_"
    out = copy.deepcopy(pose)
    for key, val in pose.items():
        if key.startswith(src_prefix):
            idx = _joint_index_from_key(key)
            if idx is not None:
                out[f"{dst_prefix}{idx}"] = mirror_joint_value(val, idx)
    return out


def arm_symmetry_score(pose):
    """两臂对称度 = mean(|left_i + right_i|), i=1..7。越小越对称。"""
    tot, n = 0.0, 0
    for i in range(1, 8):
        lk, rk = f"left_arm_joint_{i}", f"right_arm_joint_{i}"
        if lk in pose and rk in pose:
            tot += abs(pose[lk] + pose[rk])
            n += 1
    return tot / n if n else float("inf")


def derive_offsets(named):
    """从对称位姿推导 offset_i = mean(left_i + right_i)。无对称位姿返回 None。"""
    sym = [p for p in named.values() if arm_symmetry_score(p) < SYMMETRY_THRESHOLD]
    if not sym:
        return None
    off = {}
    for i in range(1, 8):
        vals = [
            p[f"left_arm_joint_{i}"] + p[f"right_arm_joint_{i}"]
            for p in sym
            if f"left_arm_joint_{i}" in p and f"right_arm_joint_{i}" in p
        ]
        off[i] = sum(vals) / len(vals)
    return off


def _accuracy(named, offsets):
    """返回 (对称位姿残差, 全位姿残差) 两个 dict {i: [..]}。

    镜像残差 = 真实左臂 - 预测左臂 = left - (offset - right) = left + right - offset。
    精度只能看对称位姿；全位姿残差大是正常的（双臂协同里两臂本就做不同动作）。
    """
    sym_r = {i: [] for i in range(1, 8)}
    all_r = {i: [] for i in range(1, 8)}
    for p in named.values():
        for i in range(1, 8):
            lk, rk = f"left_arm_joint_{i}", f"right_arm_joint_{i}"
            if lk in p and rk in p:
                all_r[i].append(p[lk] + p[rk] - offsets[i])
    for p in named.values():
        if arm_symmetry_score(p) < SYMMETRY_THRESHOLD:
            for i in range(1, 8):
                lk, rk = f"left_arm_joint_{i}", f"right_arm_joint_{i}"
                if lk in p and rk in p:
                    sym_r[i].append(p[lk] + p[rk] - offsets[i])
    return sym_r, all_r


def print_accuracy(named, offsets):
    sym_r, all_r = _accuracy(named, offsets)
    n_sym = sum(1 for p in named.values() if arm_symmetry_score(p) < SYMMETRY_THRESHOLD)
    print(f"=== 镜像精度（对称位姿 n={n_sym}，这才是真实误差）===")
    print("关节   残差 mean ± std")
    for i in range(1, 8):
        m = statistics.mean(sym_r[i])
        s = statistics.pstdev(sym_r[i])
        print(f"  j{i}    {m:+6.3f}° ± {s:5.2f}°")
    print(f"\n=== 全位姿 (n={len(named)}，仅供参考：大残差=双臂协同本身不对称) ===")
    for i in range(1, 8):
        s = statistics.pstdev(all_r[i])
        print(f"  j{i}    std {s:5.2f}°")


def verify_config(path, offsets):
    """读配置，用给定 offset 统计镜像残差。"""
    cfg = yaml.safe_load(open(path))
    named = cfg.get("named_positions", {})
    if not named:
        print("配置里没有 named_positions", file=sys.stderr)
        return
    print_accuracy(named, offsets)


def calibrate_config(path, offset_file):
    """从对称位姿重算 offset，写入 JSON，并打印精度。返回新 offset（失败返回 None）。"""
    cfg = yaml.safe_load(open(path))
    named = cfg.get("named_positions", {})
    off = derive_offsets(named)
    if off is None:
        print("标定失败：配置里没有足够对称的双臂位姿（对称度<3°）。", file=sys.stderr)
        print("镜像需要左右臂成对出现的位姿才能提取零位偏差。", file=sys.stderr)
        return None
    with open(offset_file, "w") as f:
        json.dump(off, f, indent=2)
    print(f"从对称位姿标定出 offset_i，已写入 {offset_file}：")
    for i in range(1, 8):
        print(f"  j{i}  {off[i]:+.3f}")
    print()
    print_accuracy(named, off)
    return off


def mirror_named_all(path, source_side, suffix, out_path):
    """把 source_side 手臂的所有 named_positions 镜像，追加 suffix 写入新 yaml。"""
    cfg = yaml.safe_load(open(path))
    named = cfg.get("named_positions", {})
    added = {}
    for name, pose in named.items():
        new_name = f"{name}{suffix}"
        added[new_name] = mirror_named_pose(pose, source_side)
    named.update(added)
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    print(f"镜像 {len(added)} 个位姿（{source_side} 臂 -> 另一臂），写入 {out_path}")


def main():
    ap = argparse.ArgumentParser(description="数据标定的单臂关节镜像工具")
    ap.add_argument("--calibrate", metavar="YAML", help="从对称位姿重算 offset 并写入 --offset-file")
    ap.add_argument("--verify", metavar="YAML", help="读配置校验镜像残差")
    ap.add_argument("--mirror-named", metavar="YAML", help="镜像 named_positions 写入新 yaml")
    ap.add_argument("--show-params", action="store_true", help="打印当前生效的每关节标定参数")
    ap.add_argument("--offset-file", default=DEFAULT_OFFSET_FILE,
                    help="标定 offset 的 JSON 文件路径（calibrate 写入，其余模式读取）")
    ap.add_argument("--source-side", choices=["left", "right"], default="left")
    ap.add_argument("--suffix", default="__mirrored")
    ap.add_argument("--out", metavar="YAML", default="/tmp/mirrored.yaml")
    args = ap.parse_args()

    global MIRROR_OFFSET

    # 标定模式：直接从数据重算，不读旧 offset
    if args.calibrate:
        off = calibrate_config(args.calibrate, args.offset_file)
        if off:
            MIRROR_OFFSET = off
        return

    # 其余模式：加载已存在的 offset 文件（覆盖内置默认）
    MIRROR_OFFSET = load_offsets(args.offset_file)

    if args.show_params:
        src = args.offset_file if os.path.exists(args.offset_file) else "内置默认"
        print(f"关节    offset_i (来源: {src})   镜像公式")
        for i in ARM_JOINTS:
            if i == GRIPPER_JOINT:
                print(f"  j{i}    (夹爪)        mirror = x  (同号)")
            else:
                print(f"  j{i}    {MIRROR_OFFSET[i]:+7.3f}°    mirror = {MIRROR_OFFSET[i]:+.3f} - x")
        return

    if args.verify:
        verify_config(args.verify, MIRROR_OFFSET)
        return

    if args.mirror_named:
        mirror_named_all(args.mirror_named, args.source_side, args.suffix, args.out)
        return

    ap.print_help()


if __name__ == "__main__":
    main()
