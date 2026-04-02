#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TCP标定模块 (TCP Calibrator)

使用四点法标定工具中心点(TCP)相对于法兰的偏移量。

原理：
  假设探针尖端固定在世界坐标系的某一点 P_tip，
  法兰中心的坐标为 P_flange_i，旋转矩阵为 R_i。
  探针相对于法兰的偏置向量为 t_offset。

  则对于每一个采集的姿态 i，都有：
  P_flange_i + R_i * t_offset = P_tip

  使用最小二乘法求解最优的 t_offset 和 P_tip。

操作步骤：
  1. 在工作台上固定一个尖锐靶点（如大头针）
  2. 安装探针到机械臂末端
  3. 用探针尖端精确对准靶点
  4. 采集当前姿态（保持针尖对准，变换姿态）
  5. 至少采集4个不同姿态
  6. 系统计算TCP偏移量和RMSE误差

验收标准：
  RMSE < 0.5mm = 合格
"""

import numpy as np
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple
from scipy.spatial.transform import Rotation as R


@dataclass
class TCPCalibrationResult:
    """TCP标定结果"""
    # TCP偏移量 (相对于法兰, 米)
    offset_x: float = 0.0
    offset_y: float = 0.0
    offset_z: float = 0.0
    # 靶点世界坐标 (米)
    target_x: float = 0.0
    target_y: float = 0.0
    target_z: float = 0.0
    # RMSE误差 (毫米)
    rmse_mm: float = 0.0
    # 采集姿态数量
    num_poses: int = 0
    # 是否有效
    valid: bool = False


class TCPCalibrator:
    """
    TCP四点法标定器

    用于标定工具中心点(TCP)相对于法兰的偏移量。

    使用方法：
        calibrator = TCPCalibrator()

        # 采集姿态 (至少4个)
        calibrator.capture_pose(flange_position, flange_rotation)
        calibrator.capture_pose(flange_position2, flange_rotation2)
        ...

        # 计算TCP偏移
        success, result = calibrator.solve()

        # 保存结果
        calibrator.save("tcp_offset.yaml")
    """

    def __init__(self):
        """初始化TCP标定器"""
        self.R_flanges: List[np.ndarray] = []  # 法兰旋转矩阵
        self.p_flanges: List[np.ndarray] = []  # 法兰位置向量
        self.result = TCPCalibrationResult()
        # 诊断信息
        self._first_rotation = None  # 第一个旋转矩阵（用于计算变化）

    def capture_pose(self,
                     flange_position: np.ndarray,
                     flange_rotation: np.ndarray,
                     rotation_format: str = "quaternion") -> bool:
        """
        采集一个姿态

        Args:
            flange_position: 法兰位置 [x, y, z] (米)
            flange_rotation: 法兰旋转
                - "quaternion": [qx, qy, qz, qw]
                - "euler": [roll, pitch, yaw] (弧度)
                - "matrix": 3x3旋转矩阵
            rotation_format: 旋转格式

        Returns:
            是否采集成功
        """
        # 转换旋转格式
        if rotation_format == "quaternion":
            r_matrix = R.from_quat(flange_rotation).as_matrix()
        elif rotation_format == "euler":
            r_matrix = R.from_euler('xyz', flange_rotation).as_matrix()
        elif rotation_format == "matrix":
            r_matrix = np.array(flange_rotation)
        else:
            raise ValueError(f"未知的旋转格式: {rotation_format}")

        # 保存数据
        self.R_flanges.append(r_matrix)
        self.p_flanges.append(np.array(flange_position).reshape(3, 1))

        # 诊断：计算旋转变化
        n = len(self.R_flanges)
        if n == 1:
            self._first_rotation = r_matrix.copy()
            rot_diff = 0.0
        else:
            rot_diff = np.linalg.norm(r_matrix - self._first_rotation, 'fro')

        print(f"  ✓ 捕获姿态 {n}: pos=({flange_position[0]:.4f}, {flange_position[1]:.4f}, {flange_position[2]:.4f})m")
        print(f"    旋转变化: {rot_diff:.4f} {'✓' if rot_diff > 0.3 else '⚠ 建议增大姿态变化'}")

        return True

    def get_capture_count(self) -> int:
        """获取已采集的姿态数量"""
        return len(self.R_flanges)

    def clear_captures(self):
        """清空已采集的数据"""
        self.R_flanges.clear()
        self.p_flanges.clear()
        self.result = TCPCalibrationResult()

    def solve(self) -> Tuple[bool, TCPCalibrationResult]:
        """
        执行四点法标定

        Returns:
            (success, result) 标定结果
        """
        n = len(self.R_flanges)

        if n < 4:
            print(f"✗ 采集数量不足 ({n}/4)，四点法标定至少需要4个不同姿态")
            return False, self.result

        print(f"\n开始四点法标定计算，使用 {n} 个姿态...")

        try:
            """
            数学推导：
            对于 i = 1..N:
            P_flange_i + R_i * t_offset = P_tip
            R_i * t_offset - P_tip = -P_flange_i

            构建超定方程组 A * X = B
            其中未知数 X = [t_offset_x, t_offset_y, t_offset_z, P_tip_x, P_tip_y, P_tip_z]^T

            对于第 i 个方程：
            [ R_i  |  -I ] * X = -P_flange_i
            """

            # 构建方程组
            A = np.zeros((3 * n, 6))
            B = np.zeros((3 * n, 1))
            I3 = np.eye(3)

            for i in range(n):
                A[3*i : 3*i+3, 0:3] = self.R_flanges[i]
                A[3*i : 3*i+3, 3:6] = -I3
                B[3*i : 3*i+3, 0] = -self.p_flanges[i].flatten()

            # 诊断：计算方程组条件数
            try:
                cond_num = np.linalg.cond(A)
                print(f"  方程组条件数: {cond_num:.1f}")
                if cond_num > 100:
                    print(f"  ⚠ 条件数过大，方程组接近奇异，需要更大的姿态差异")
                elif cond_num > 50:
                    print(f"  ⚠ 条件数偏高，建议增大姿态差异")
                else:
                    print(f"  ✓ 方程组条件良好")
            except:
                pass

            # 最小二乘求解
            X, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)

            t_offset = X[0:3].flatten()
            p_tip = X[3:6].flatten()

            # 计算RMSE误差
            errors = []
            for i in range(n):
                predicted_tip = self.p_flanges[i].flatten() + self.R_flanges[i] @ t_offset
                err_dist = np.linalg.norm(predicted_tip - p_tip)
                errors.append(err_dist)

            rmse_m = np.sqrt(np.mean(np.array(errors)**2))
            rmse_mm = rmse_m * 1000.0

            # 保存结果 (确保转换为Python原生类型)
            self.result = TCPCalibrationResult(
                offset_x=float(t_offset[0]),
                offset_y=float(t_offset[1]),
                offset_z=float(t_offset[2]),
                target_x=float(p_tip[0]),
                target_y=float(p_tip[1]),
                target_z=float(p_tip[2]),
                rmse_mm=float(rmse_mm),
                num_poses=int(n),
                valid=bool(rmse_mm < 0.5)
            )

            # 打印结果
            print("\n" + "="*50)
            print("TCP标定结果")
            print("="*50)
            print(f"TCP偏移量 (相对于法兰):")
            print(f"  X = {t_offset[0]*1000:.2f} mm")
            print(f"  Y = {t_offset[1]*1000:.2f} mm")
            print(f"  Z = {t_offset[2]*1000:.2f} mm")
            print(f"\n靶点世界坐标:")
            print(f"  X = {p_tip[0]:.4f} m")
            print(f"  Y = {p_tip[1]:.4f} m")
            print(f"  Z = {p_tip[2]:.4f} m")
            print(f"\nRMSE误差: {rmse_mm:.2f} mm")

            if rmse_mm > 0.5:
                print(f"\n⚠ 警告: RMSE误差 ({rmse_mm:.2f}mm) 大于0.5mm")
                print("  可能原因:")
                print("  1. 探针尖端没有精确对准靶点")
                print("  2. 姿态变化太小")
                print("  3. 探针固定不牢")
                print("  建议: 按Q退出后重新采集")
                return False, self.result
            else:
                print(f"\n✓ 标定精度达标!")
                return True, self.result

        except Exception as e:
            print(f"✗ 标定计算失败: {e}")
            return False, self.result

    def save(self, filepath: str) -> bool:
        """
        保存TCP标定结果到YAML文件

        Args:
            filepath: 文件路径

        Returns:
            是否保存成功
        """
        if not self.result.valid:
            print("警告: TCP标定结果无效，不建议保存")

        try:
            data = {
                'tcp_offset': {
                    'x': float(self.result.offset_x),
                    'y': float(self.result.offset_y),
                    'z': float(self.result.offset_z)
                },
                'target_position': {
                    'x': float(self.result.target_x),
                    'y': float(self.result.target_y),
                    'z': float(self.result.target_z)
                },
                'rmse_error_mm': float(self.result.rmse_mm),
                'num_poses': self.result.num_poses,
                'valid': self.result.valid
            }

            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

            with open(filepath, 'w') as f:
                yaml.safe_dump(data, f, default_flow_style=False)

            print(f"✓ TCP标定结果已保存: {filepath}")
            return True

        except Exception as e:
            print(f"✗ 保存失败: {e}")
            return False

    @staticmethod
    def load(filepath: str) -> Optional[TCPCalibrationResult]:
        """
        从YAML文件加载TCP标定结果

        Args:
            filepath: 文件路径

        Returns:
            标定结果，失败返回None
        """
        if not Path(filepath).exists():
            return None

        try:
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)

            result = TCPCalibrationResult()
            result.offset_x = data['tcp_offset']['x']
            result.offset_y = data['tcp_offset']['y']
            result.offset_z = data['tcp_offset']['z']
            result.target_x = data['target_position']['x']
            result.target_y = data['target_position']['y']
            result.target_z = data['target_position']['z']
            result.rmse_mm = data['rmse_error_mm']
            result.num_poses = data['num_poses']
            result.valid = data['valid']

            return result

        except Exception as e:
            print(f"加载TCP标定结果失败: {e}")
            return None

    def get_offset_array(self) -> np.ndarray:
        """获取TCP偏移量数组 [x, y, z] (米)"""
        return np.array([
            self.result.offset_x,
            self.result.offset_y,
            self.result.offset_z
        ])

    def set_offset(self, offset: np.ndarray):
        """设置TCP偏移量"""
        self.result.offset_x = float(offset[0])
        self.result.offset_y = float(offset[1])
        self.result.offset_z = float(offset[2])

    def compute_tcp_position(self, flange_position: np.ndarray,
                              flange_rotation: np.ndarray,
                              rotation_format: str = "quaternion") -> np.ndarray:
        """
        根据法兰位姿和TCP偏移计算TCP位置

        Args:
            flange_position: 法兰位置 [x, y, z] (米)
            flange_rotation: 法兰旋转
            flange_rotation: 法兰旋转

            rotation_format: 旋转格式

        Returns:
            TCP位置 [x, y, z] (米)
        """
        # 转换旋转格式
        if rotation_format == "quaternion":
            r_matrix = R.from_quat(flange_rotation).as_matrix()
        elif rotation_format == "euler":
            r_matrix = R.from_euler('xyz', flange_rotation).as_matrix()
        elif rotation_format == "matrix":
            r_matrix = np.array(flange_rotation)
        else:
            raise ValueError(f"未知的旋转格式: {rotation_format}")

        t_offset = self.get_offset_array()
        tcp_position = np.array(flange_position) + r_matrix @ t_offset

        return tcp_position


class TCPIterativeRefiner:
    """
    TCP迭代优化器

    使用相机辅助迭代优化TCP偏移量。
    通过测量探针实际移动距离与预期移动距离的偏差来修正TCP偏移。

    原理：
      假设TCP偏移误差为 Δt，执行移动指令后：
      实际移动 = 指令移动 + R × Δt

      通过测量实际移动偏差，可以反推并修正 Δt。
    """

    def __init__(self, tcp_calibrator: TCPCalibrator, pixel_to_mm_ratio: float = 0.5):
        """
        初始化迭代优化器

        Args:
            tcp_calibrator: TCP标定器实例
            pixel_to_mm_ratio: 像素到毫米的转换比例 (mm/pixel)
        """
        self.tcp_calibrator = tcp_calibrator
        self.pixel_to_mm_ratio = pixel_to_mm_ratio
        self.iteration_history = []

    def set_pixel_ratio_from_board(self, board_square_size_mm: float,
                                     square_pixels: float):
        """
        从标定板设置像素比例

        Args:
            board_square_size_mm: 标定板格子实际尺寸 (mm)
            square_pixels: 格子在图像中的像素尺寸
        """
        self.pixel_to_mm_ratio = board_square_size_mm / square_pixels
        print(f"像素比例已设置: {self.pixel_to_mm_ratio:.4f} mm/pixel")

    def measure_movement_error(self,
                               pixel_start: Tuple[float, float],
                               pixel_end: Tuple[float, float],
                               expected_movement_mm: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        测量移动误差

        Args:
            pixel_start: 起始像素坐标 (u, v)
            pixel_end: 结束像素坐标 (u, v)
            expected_movement_mm: 预期移动向量 [dx, dy, dz] (mm)

        Returns:
            (误差向量 [mm], 误差距离 mm)
        """
        # 计算实际移动 (像素)
        du = pixel_end[0] - pixel_start[0]
        dv = pixel_end[1] - pixel_start[1]

        # 转换为毫米 (仅XY平面)
        actual_dx = du * self.pixel_to_mm_ratio
        actual_dy = dv * self.pixel_to_mm_ratio

        # 计算误差
        error_x = actual_dx - expected_movement_mm[0]
        error_y = actual_dy - expected_movement_mm[1]

        error_vector = np.array([error_x, error_y, 0.0])
        error_distance = np.sqrt(error_x**2 + error_y**2)

        return error_vector, error_distance

    def refine_offset_2d(self,
                         error_xy_mm: np.ndarray,
                         flange_rotation: np.ndarray,
                         rotation_format: str = "quaternion",
                         learning_rate: float = 0.5) -> np.ndarray:
        """
        根据XY平面误差修正TCP偏移

        Args:
            error_xy_mm: XY平面误差 [ex, ey, 0] (mm)
            flange_rotation: 法兰旋转
            rotation_format: 旋转格式
            learning_rate: 学习率 (0-1)

        Returns:
            修正后的TCP偏移 (mm)
        """
        # 转换旋转格式
        if rotation_format == "quaternion":
            r_matrix = R.from_quat(flange_rotation).as_matrix()
        elif rotation_format == "euler":
            r_matrix = R.from_euler('xyz', flange_rotation).as_matrix()
        else:
            r_matrix = np.array(flange_rotation)

        # 计算修正量 (逆向投影)
        # 误差 = R × Δt，所以 Δt = R^T × error
        r_matrix_inv = r_matrix.T
        correction = r_matrix_inv @ (error_xy_mm / 1000.0)  # 转换为米

        # 应用学习率
        correction = correction * learning_rate

        # 修正TCP偏移
        current_offset = self.tcp_calibrator.get_offset_array()
        new_offset = current_offset - correction

        self.tcp_calibrator.set_offset(new_offset)

        return new_offset * 1000  # 返回mm

    def run_iteration(self,
                      pixel_start: Tuple[float, float],
                      pixel_end: Tuple[float, float],
                      expected_movement_mm: np.ndarray,
                      flange_rotation: np.ndarray,
                      rotation_format: str = "quaternion",
                      learning_rate: float = 0.5) -> dict:
        """
        执行一次迭代

        Args:
            pixel_start: 起始像素坐标
            pixel_end: 结束像素坐标
            expected_movement_mm: 预期移动 (mm)
            flange_rotation: 法兰旋转
            rotation_format: 旋转格式
            learning_rate: 学习率

        Returns:
            迭代结果字典
        """
        # 测量误差
        error_vector, error_distance = self.measure_movement_error(
            pixel_start, pixel_end, expected_movement_mm
        )

        # 修正偏移
        new_offset = self.refine_offset_2d(
            error_vector, flange_rotation, rotation_format, learning_rate
        )

        # 记录历史
        iteration_result = {
            'error_mm': error_distance,
            'error_vector_mm': error_vector,
            'new_offset_mm': new_offset,
            'pixel_start': pixel_start,
            'pixel_end': pixel_end
        }
        self.iteration_history.append(iteration_result)

        return iteration_result

    def get_statistics(self) -> dict:
        """获取迭代统计信息"""
        if not self.iteration_history:
            return {'iterations': 0}

        errors = [h['error_mm'] for h in self.iteration_history]
        return {
            'iterations': len(self.iteration_history),
            'initial_error_mm': errors[0],
            'final_error_mm': errors[-1],
            'min_error_mm': min(errors),
            'improvement': (errors[0] - errors[-1]) / errors[0] * 100 if errors[0] > 0 else 0
        }


if __name__ == "__main__":
    # 测试
    print("TCP标定模块测试")

    calibrator = TCPCalibrator()

    # 模拟数据测试
    np.random.seed(42)

    # 真实TCP偏移
    true_offset = np.array([0.05, 0.0, 0.2])

    # 真实靶点位置
    true_target = np.array([0.3, 0.2, 0.1])

    # 生成4个姿态
    for i in range(4):
        # 随机旋转
        euler = np.random.uniform(-0.5, 0.5, 3)
        R_i = R.from_euler('xyz', euler).as_matrix()

        # 计算法兰位置: P_flange = P_tip - R * t_offset
        p_flange = true_target - R_i @ true_offset

        # 添加小噪声
        p_flange += np.random.normal(0, 0.0001, 3)  # 0.1mm噪声

        # 转换为四元数
        quat = R.from_matrix(R_i).as_quat()

        calibrator.capture_pose(p_flange, quat, "quaternion")

    # 求解
    success, result = calibrator.solve()

    if success:
        print(f"\n测试通过!")
        print(f"真实偏移: {true_offset*1000} mm")
        print(f"计算偏移: [{result.offset_x*1000:.2f}, {result.offset_y*1000:.2f}, {result.offset_z*1000:.2f}] mm")