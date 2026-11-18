"""Screw-axis (Skeleton) 运动学描述 — supre 机器人模板（全局 PoE 版）。

背景 / 为什么不用 DH、也不用局部 PoE
-----------------------------------
DH：每关节 4 参数 (alpha, a, d, theta)，框架约束强：
  - 相邻关节轴平行时"公法线"不唯一 -> 参数奇异、标定病态；
  - 第 i 框架建在第 i-1 框架上 -> 测量误差逐关节累积；
  - 几何被硬拆成 alpha/a/d -> 对"反测标定"极不友好。

局部 PoE（本文件旧版）：轴表达在各连杆系、parent_t_this 逐段相乘；
  但中间连杆 frame 单 tracker 测不到、需反算，误差仍沿链累积 —— 本质还是 DH 毛病。

全局 PoE（本文件）：所有 screw 轴都表达在同一个基座系里：
    T(θ) = exp(S1·θ1) · exp(S2·θ2) · … · exp(Sn·θn) · M
  - Si = (ωi, qi)：关节 i 的旋转轴方向 ωi + 轴上一点 qi，均在基座系。
  - M：零位姿（θ=0）下末端法兰在基座系的位姿。
  圆拟合"直进直出"：单关节慢扫 → tracker 画圆 → 圆面法向 = ωi、圆心 = qi；
  各轴独立测量、平行轴不奇异、误差不累积。

数据来源（VIVE 反测标定）：
  - 每关节 axis(ω) / point(q)：单关节慢扫的 3D 圆拟合。
  - M：θ=0 时 T2(法兰 probe) 在 T0(基座) 里的位姿。
  - limits 已自动从 profile 读入（见 build_supre_skeleton）。

与"粗 DH"的对应（便于把已有 DH 结论迁移过来）：
  - 杆长 a / 偏距 d -> 由各 joint 的 point(q) 位置承载；
  - 扭转角 alpha   -> 由 axis(ω) 方向承载（screw 里 alpha 不单独存在）；
  - 零点偏置 theta0 -> 约定"零位姿"吸收：所有 axis/point/M 都在 θ=0 参考位姿下测。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# --------------------------------------------------------------------------- #
# 基础数学：四元数 / 旋转 / screw 指数
# --------------------------------------------------------------------------- #

def quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """四元数 (x, y, z, w) -> 3x3 旋转矩阵（自动归一化）。"""
    x, y, z, w = float(qx), float(qy), float(qz), float(qw)
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=float)


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 旋转矩阵 -> 四元数 (x, y, z, w)。Shepperd 方法，数值稳健。"""
    R = np.asarray(R, dtype=float)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def axis_angle_to_rot(axis, angle: float) -> np.ndarray:
    """绕单位轴 axis 旋转 angle(rad) 的 3x3 旋转矩阵（Rodrigues）。"""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]],
                  [a[2], 0, -a[0]],
                  [-a[1], a[0], 0]], dtype=float)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def screw_transform(axis, point, angle: float) -> np.ndarray:
    """绕基座系里的轴 (axis 单位方向, point 轴上一点) 转 angle(rad) 的 4x4 变换。

    R = exp([ω]× θ)；平移 = (I - R)·q。对应全局 PoE 里的 exp(S·θ)。
    """
    omega = np.asarray(axis, dtype=float)
    omega = omega / np.linalg.norm(omega)
    q = np.asarray(point, dtype=float)
    R = axis_angle_to_rot(omega, angle)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = (np.eye(3) - R) @ q
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
# 数据模型（全局 PoE）
# --------------------------------------------------------------------------- #

@dataclass
class Element:
    """Skeleton 图里的一个节点：link 或 joint。

    全局 PoE 里，joint 的几何都在基座系：
      axis  = ω（单位旋转轴方向）
      point = q（轴上一点）
    link 元素只有结构/打印意义，不参与 FK。
    """
    id: int
    name: str
    kind: str                                # "link" | "joint"
    parent: Optional[int] = None
    children: list[int] = field(default_factory=list)

    # joint 专属几何（基座系，None=待测）
    axis: Optional[np.ndarray] = None        # ω，3 维单位向量
    point: Optional[np.ndarray] = None       # q，轴上一点（3 维）
    limits: Optional[tuple[float, float]] = None  # (lower, upper) 单位：度

    # link 专属（模板阶段暂不需要动力学量，留接口）
    mass: Optional[float] = None
    center_of_gravity: Optional[Transform] = None
    inertia: Optional[np.ndarray] = None


@dataclass
class Skeleton:
    """一条串行链（基座 -> 末端）的全局 PoE 描述。"""
    name: str
    elements: list[Element] = field(default_factory=list)
    M: Optional[Transform] = None            # 零位姿末端（基座系），None=待测

    def joints(self) -> list[Element]:
        return [e for e in self.elements if e.kind == "joint"]

    def links(self) -> list[Element]:
        return [e for e in self.elements if e.kind == "link"]

    def dof(self) -> int:
        return len(self.joints())

    def missing_geometry(self) -> list[str]:
        """列出尚未测量的几何量（axis / point / M）。"""
        missing = []
        for e in self.joints():
            if e.axis is None:
                missing.append(f"{e.name}.axis")
            if e.point is None:
                missing.append(f"{e.name}.point")
        if self.M is None:
            missing.append("M (零位姿末端)")
        return missing


# --------------------------------------------------------------------------- #
# 正运动学（全局 PoE）
# --------------------------------------------------------------------------- #

def forward_kinematics(skeleton: Skeleton, joint_angles: dict[str, float]) -> np.ndarray:
    """基座系 -> 末端系的 4x4 变换：T(θ) = exp(S1θ1)·…·exp(Snθn)·M。

    joint_angles: {joint_name: 角度(度)}，未给出的关节按 0 处理。
    """
    if skeleton.M is None:
        raise ValueError("几何未填：M（零位姿末端）为 None。")
    T = skeleton.M.matrix()
    for e in reversed(skeleton.joints()):
        if e.axis is None or e.point is None:
            raise ValueError(f"几何未填：{e.name}.axis/point 为 None。")
        theta = np.deg2rad(joint_angles.get(e.name, 0.0))
        T = screw_transform(e.axis, e.point, theta) @ T
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
    return Skeleton(name=name, elements=elements)  # M 默认 None（待测）


def _print_skeleton(sk: Skeleton) -> None:
    print(f"Skeleton: {sk.name}  (DOF={sk.dof()})")
    for e in sk.elements:
        tag = "L" if e.kind == "link" else "J"
        lim = f"  limits={e.limits}" if e.kind == "joint" and e.limits else ""
        geo = ""
        if e.kind == "joint":
            a = "待测" if e.axis is None else np.array2string(np.asarray(e.axis), precision=3)
            p = "待测" if e.point is None else np.array2string(np.asarray(e.point), precision=3)
            geo = f"  axis={a}  point={p}"
        print(f"  [{tag}] id={e.id:<3} {e.name:<28} parent={e.parent}{geo}{lim}")
    m = "待测" if sk.M is None else f"({sk.M.x:.3f},{sk.M.y:.3f},{sk.M.z:.3f})"
    print(f"  M(零位姿末端) = {m}")
    missing = sk.missing_geometry()
    print(f"  待测几何 ({len(missing)} 项):")
    for x in missing:
        print(f"    - {x}")


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
