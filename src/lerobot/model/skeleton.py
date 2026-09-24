"""Screw-axis (Skeleton) 运动学描述 — supre 机器人模板。

背景 / 为什么不用 DH 而用 screw 轴
---------------------------------
DH 用每关节 4 参数 (alpha, a, d, theta) 描述，代价是强框架约定：
  - 相邻关节轴平行时"公法线"不唯一 -> 参数奇异、标定病态；
  - 第 i 框架建在第 i-1 框架上 -> 测量误差逐关节累积；
  - 几何要被硬拆成 alpha/a/d 三个量 -> 对"反测标定"极不友好。

screw/Skeleton 表示每关节只问一句："这根关节绕空间里哪条线转？"
  - axis: 单位旋转轴 (omega)
  - parent_t_this: 零位姿下"父系 -> 本元素系"的固定刚体变换
这等价于 Product-of-Exponentials (PoE)。每个轴在同一个基座系里独立测量，
平行轴不奇异、误差不累积。本文件刻意对齐 Intrinsic Core 的
`intrinsic_kinematics` Skeleton 格式（element 图 + parent_t_this + axis + limits）。

与 DH 的对应关系（便于你把"粗 DH"结论迁移过来）：
  - 杆长 a / 偏距 d   -> 由各 element 的 parent_t_this 里的 position 承载
  - 扭转角 alpha       -> 由 axis 方向承载（screw 里 alpha 根本不单独存在）
  - 零点偏置 theta0    -> 约定"零位姿"吸收：所有 parent_t_this/axis 都在某个
                          参考位姿（theta=0）下测量。先定零点再测几何。

待填值（CPA / 物理测量产出）：
  - 每个 element 的 parent_t_this（关节/连杆的固定几何变换）
  - 每个 joint 的 axis（旋转轴方向）
  - limits 已自动从 profile 读入（见 build_supre_skeleton）
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# --------------------------------------------------------------------------- #
# 基础数学：SE(3) 变换 + 旋转
# --------------------------------------------------------------------------- #

def quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """四元数 (x, y, z, w) -> 3x3 旋转矩阵。"""
    x, y, z, w = float(qx), float(qy), float(qz), float(qw)
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=float)


def axis_angle_to_rot4(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    """绕单位轴 axis 旋转 angle(rad) 的 4x4 齐次变换（Rodrigues）。"""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]],
                  [a[2], 0, -a[0]],
                  [-a[1], a[0], 0]], dtype=float)
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    T = np.eye(4)
    T[:3, :3] = R
    return T


@dataclass
class Transform:
    """SE(3)：位置 + 四元数姿态（x, y, z, qx, qy, qz, qw）。"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0  # 单位四元数

    def matrix(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = quat_to_rot(self.qx, self.qy, self.qz, self.qw)
        T[:3, 3] = [self.x, self.y, self.z]
        return T

    @classmethod
    def identity(cls) -> "Transform":
        return cls()


# --------------------------------------------------------------------------- #
# 数据模型（对齐 Intrinsic Skeleton）
# --------------------------------------------------------------------------- #

@dataclass
class Element:
    """Skeleton 图里的一个节点：link 或 joint（两者共用同一结构）。

    parent_t_this = 零位姿下"父元素系 -> 本元素系"的固定变换。
    joint 额外有 axis（旋转轴）和 limits。
    """
    id: int
    name: str
    kind: str                                # "link" | "joint"
    parent_t_this: Optional[Transform] = None  # None = 待测（CPA/物理测量填）
    parent: Optional[int] = None
    children: list[int] = field(default_factory=list)

    # joint 专属
    axis: Optional[tuple[float, float, float]] = None  # None = 待测
    limits: Optional[tuple[float, float]] = None       # (lower, upper) 单位：度

    # link 专属（模板阶段暂不需要动力学量，留接口）
    mass: Optional[float] = None
    center_of_gravity: Optional[Transform] = None
    inertia: Optional[np.ndarray] = None


@dataclass
class Skeleton:
    """一条串行链（基座 -> 末端）的运动学描述。"""
    name: str
    elements: list[Element] = field(default_factory=list)

    def joints(self) -> list[Element]:
        return [e for e in self.elements if e.kind == "joint"]

    def links(self) -> list[Element]:
        return [e for e in self.elements if e.kind == "link"]

    def dof(self) -> int:
        return len(self.joints())

    def missing_geometry(self) -> list[str]:
        """列出尚未测量的几何量（axis / parent_t_this 仍为 None 的）。"""
        missing = []
        for e in self.elements:
            if e.parent_t_this is None and e.parent is not None:
                missing.append(f"{e.name}.parent_t_this")
            if e.kind == "joint" and e.axis is None:
                missing.append(f"{e.name}.axis")
        return missing


# --------------------------------------------------------------------------- #
# 正运动学（PoE）：填完几何后即可验证
# --------------------------------------------------------------------------- #

def forward_kinematics(skeleton: Skeleton, joint_angles: dict[str, float]) -> np.ndarray:
    """基座系 -> 末端系的 4x4 变换。

    joint_angles: {joint_name: 角度(度)}，未给出的关节按 0 处理。

    PoE 约定（与 Intrinsic 一致）：沿 element 链顺序相乘，
    每个 element 先乘 parent_t_this（固定），joint 再乘绕本地 axis 的旋转。
    """
    T = np.eye(4)
    for e in skeleton.elements:
        if e.parent_t_this is None:
            if e.parent is None:
                continue  # base 元素（parent=None）默认单位变换
            raise ValueError(
                f"几何未填：{e.name}.parent_t_this 为 None。"
                f"请先用 CPA/物理测量补齐（missing_geometry() 列出全部待填项）。"
            )
        T = T @ e.parent_t_this.matrix()
        if e.kind == "joint":
            if e.axis is None:
                raise ValueError(f"几何未填：{e.name}.axis 为 None。")
            theta = np.deg2rad(joint_angles.get(e.name, 0.0))
            T = T @ axis_angle_to_rot4(e.axis, theta)
    return T


# --------------------------------------------------------------------------- #
# 从 profile 构建 supre 机器人 Skeleton（模板）
# --------------------------------------------------------------------------- #

def _is_gripper(j: dict) -> bool:
    """profile 里判断一个 joint 是否为夹爪（不进 FK 链）。"""
    return bool(j.get("is_gripper", False)) or "device" in j


def _is_rotational(j: dict) -> bool:
    """有 CAN node_id 且非夹爪 -> 参与运动学链的旋转关节。"""
    return "node_id" in j and not _is_gripper(j)


def _limits(j: dict) -> Optional[tuple[float, float]]:
    if "min" in j and "max" in j:
        return (float(j["min"]), float(j["max"]))
    return None


def _element(eid: int, name: str, kind: str, parent: Optional[int],
             j: Optional[dict] = None) -> Element:
    e = Element(id=eid, name=name, kind=kind, parent=parent)
    if kind == "joint" and j is not None:
        e.limits = _limits(j)
    return e


def build_supre_skeleton(profile_path: str | Path, arm: str,
                         include_trunk: bool = True) -> Skeleton:
    """从 supre profile YAML 构建一条臂的 Skeleton（几何留空待测）。

    arm: "left" | "right"。
    include_trunk: 是否把腰部 trunk_joint_1/2 作为链根（完整 base->flange 链）。
        注意 trunk 是双臂共享的，两条臂各自的 Skeleton 会重复包含 trunk。

    链顺序：base_link -> [trunk 链] -> {arm}_arm_joint_1..N -> flange_link
    其中夹爪（is_gripper / device 串口）被跳过，不参与 FK。
    """
    import yaml  # 延迟导入，保持模块可独立于 profile 依赖

    with open(profile_path) as f:
        profile = yaml.safe_load(f)

    joints = profile.get("joints", [])
    prefix = f"{arm}_arm_"
    arm_rot = [j for j in joints
               if j["name"].startswith(prefix) and _is_rotational(j)]
    trunk_rot = [j for j in joints
                 if j["name"].startswith("trunk_") and _is_rotational(j)]

    elements: list[Element] = []
    eid = 0
    base = _element(eid, "base_link", "link", parent=None)
    elements.append(base)

    parent_id = 0

    def _chain(chain_joints: list[dict], link_tag: str):
        nonlocal eid, parent_id
        for j in chain_joints:
            eid += 1
            jt = _element(eid, j["name"], "joint", parent=parent_id, j=j)
            elements.append(jt)
            elements[parent_id].children.append(eid)
            parent_id = eid
            eid += 1
            ln = _element(eid, f"{j['name']}_{link_tag}_link", "link", parent=parent_id)
            elements.append(ln)
            elements[parent_id].children.append(eid)
            parent_id = eid

    if include_trunk:
        _chain(trunk_rot, "trunk")
    _chain(arm_rot, "arm")

    eid += 1
    flange = _element(eid, f"{arm}_flange", "link", parent=parent_id)
    elements.append(flange)
    elements[parent_id].children.append(eid)

    name = f"supre_{arm}_arm" + ("_with_trunk" if include_trunk else "")
    return Skeleton(name=name, elements=elements)


def _print_skeleton(sk: Skeleton) -> None:
    print(f"Skeleton: {sk.name}  (DOF={sk.dof()})")
    for e in sk.elements:
        tag = "L" if e.kind == "link" else "J"
        lim = f"  limits={e.limits}" if e.kind == "joint" and e.limits else ""
        axis = f"  axis={e.axis}" if e.kind == "joint" else ""
        ptt = "待测" if e.parent_t_this is None and e.parent is not None else "单位"
        print(f"  [{tag}] id={e.id:<3} {e.name:<28} parent={e.parent}  parent_t_this={ptt}{lim}{axis}")
    missing = sk.missing_geometry()
    print(f"  待测几何 ({len(missing)} 项):")
    for m in missing:
        print(f"    - {m}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="构建 supre Skeleton 模板并打印待测几何")
    ap.add_argument("--profile", default="profiles/robot1_trunk.yaml")
    ap.add_argument("--arm", choices=["left", "right"], default="left")
    ap.add_argument("--no-trunk", action="store_true")
    args = ap.parse_args()

    for arm in ("left", "right"):
        sk = build_supre_skeleton(args.profile, arm, include_trunk=not args.no_trunk)
        _print_skeleton(sk)
        print()
