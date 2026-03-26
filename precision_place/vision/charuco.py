"""
ChArUco检测器 (ChArUco Detector)

提供ChArUco标定板的检测功能。
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List


class CharucoDetector:
    """ChArUco标定板检测器"""

    def __init__(self,
                 squares_x: int = 5,
                 squares_y: int = 7,
                 square_length: float = 0.03,
                 marker_length: float = 0.022,
                 dictionary_id: int = cv2.aruco.DICT_6X6_250):
        """
        初始化ChArUco检测器

        Args:
            squares_x: X方向格子数
            squares_y: Y方向格子数
            square_length: 格子边长 (米)
            marker_length: ArUco标记边长 (米)
            dictionary_id: ArUco字典ID
        """
        self.squares_x = squares_x
        self.squares_y = squares_y
        self.square_length = square_length
        self.marker_length = marker_length

        # 创建字典和板
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.board = cv2.aruco.CharucoBoard(
            (squares_x, squares_y),
            square_length,
            marker_length,
            self.dictionary
        )

        # 检测参数
        self.params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.params)

    def detect(self,
               image: np.ndarray,
               camera_matrix: np.ndarray = None,
               dist_coeffs: np.ndarray = None) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        检测ChArUco标定板

        Args:
            image: 输入图像
            camera_matrix: 相机内参矩阵 (用于位姿估计)
            dist_coeffs: 畸变系数

        Returns:
            (success, rvec, tvec, corners)
            - success: 是否检测成功
            - rvec: 旋转向量
            - tvec: 平移向量
            - corners: ChArUco角点
        """
        if image is None:
            return False, None, None, None

        # 转灰度
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 检测ArUco标记
        marker_corners, marker_ids, rejected = self.detector.detectMarkers(gray)

        if marker_ids is None or len(marker_ids) == 0:
            return False, None, None, None

        # 插值ChArUco角点
        ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, gray, self.board
        )

        if not ret or charuco_ids is None or len(charuco_ids) < 4:
            return False, None, None, None

        # 如果提供了相机内参，估计位姿
        if camera_matrix is not None and dist_coeffs is not None:
            success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                charuco_corners, charuco_ids, self.board,
                camera_matrix, dist_coeffs,
                np.empty(1), np.empty(1)
            )

            if not success:
                return False, None, None, charuco_corners

            return True, rvec, tvec, charuco_corners

        return True, None, None, charuco_corners

    def draw_detection(self,
                       image: np.ndarray,
                       rvec: np.ndarray,
                       tvec: np.ndarray,
                       camera_matrix: np.ndarray,
                       dist_coeffs: np.ndarray,
                       axis_length: float = 0.1) -> np.ndarray:
        """
        绘制检测结果

        Args:
            image: 输入图像
            rvec: 旋转向量
            tvec: 平移向量
            camera_matrix: 相机内参
            dist_coeffs: 畸变系数
            axis_length: 坐标轴长度 (米)

        Returns:
            绘制后的图像
        """
        result = image.copy()
        cv2.drawFrameAxes(result, camera_matrix, dist_coeffs, rvec, tvec, axis_length)
        return result

    @staticmethod
    def generate_board(squares_x: int = 5,
                       squares_y: int = 7,
                       square_length_px: int = 100,
                       marker_length_px: int = 80,
                       output_path: str = "charuco_board.png") -> bool:
        """
        生成ChArUco标定板图像

        Args:
            squares_x: X方向格子数
            squares_y: Y方向格子数
            square_length_px: 格子边长 (像素)
            marker_length_px: 标记边长 (像素)
            output_path: 输出路径

        Returns:
            是否成功
        """
        try:
            dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

            board = cv2.aruco.CharucoBoard(
                (squares_x, squares_y),
                square_length_px,
                marker_length_px,
                dictionary
            )

            img_size = (squares_x * square_length_px, squares_y * square_length_px)
            board_img = board.generateImage(img_size)

            cv2.imwrite(output_path, board_img)
            print(f"ChArUco标定板已生成: {output_path}")
            return True

        except Exception as e:
            print(f"生成标定板失败: {e}")
            return False