# lerobot/cameras/pybullet_camera.py
import numpy as np
import pybullet as p
from lerobot.cameras.camera import Camera
from lerobot.cameras.configs import CameraConfig, ColorMode
from dataclasses import dataclass

@CameraConfig.register_subclass("pybullet")
@dataclass
class PyBulletCameraConfig(CameraConfig):
    """配置PyBullet仿真相机的参数"""
    # PyBullet相机特有参数
    fov: float = 60.0  # 视野角度(度)
    near_val: float = 0.1  # 近平面
    far_val: float = 10.0  # 远平面
    position: tuple[float, float, float] = (0.5, 0, 0.8)  # 相机位置(x,y,z)
    orientation: tuple[float, float, float, float] = (0.5, -0.5, -0.5, 0.5)  # 四元数朝向
    width: int = 640
    height: int = 480
    fps: int = 30


class PyBulletCamera(Camera):
    """PyBullet仿真相机实现"""
    def __init__(self, config: PyBulletCameraConfig):
        super().__init__(config)
        self.config = config
        self.pybullet_client = None  # PyBullet客户端ID
        self.is_connect = False

    def connect(self, pybullet_client: int | None = None) -> None:
        """连接到PyBullet仿真环境"""
        if self.is_connected:
            return
        # 使用外部传入的PyBullet客户端或默认客户端
        self.pybullet_client = pybullet_client or p.connect(p.GUI) if p.getConnectionInfo(0).get("isConnected", 0) == 0 else 0
        self.is_connect = True

    @property
    def is_connected(self) -> bool:
        return self.is_connect
    
    @staticmethod
    def find_cameras():
        return []
    
    def read(self, color_mode: ColorMode | None = None) -> np.ndarray:
        """从PyBullet获取渲染图像"""
        if not self.is_connected:
            raise RuntimeError("PyBullet camera not connected")

        # 计算相机视图矩阵和投影矩阵
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.config.position,
            cameraTargetPosition=(0, 0, 0),  # 看向原点(可自定义)
            cameraUpVector=(0, 0, 1),  # 上方向为z轴
            physicsClientId=self.pybullet_client
        )

        projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.config.fov,
            aspect=self.config.width / self.config.height,
            nearVal=self.config.near_val,
            farVal=self.config.far_val,
            physicsClientId=self.pybullet_client
        )

        # 渲染图像
        _, _, rgb_img, _, _ = p.getCameraImage(
            width=self.config.width,
            height=self.config.height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,  # 使用硬件加速渲染
            physicsClientId=self.pybullet_client
        )

        # 转换为RGB格式并去除alpha通道
        rgb_img = rgb_img[:, :, :3].astype(np.uint8)
        return rgb_img

    def async_read(self, timeout_ms: float = 1000) -> np.ndarray:
        """异步读取（PyBullet中同步实现即可）"""
        return self.read()

    def disconnect(self) -> None:
        """断开连接（不关闭PyBullet客户端，避免影响其他组件）"""
        self.is_connect = False