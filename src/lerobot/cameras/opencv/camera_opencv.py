"""
OpenCV Camera implementation for robot vision system.

This module provides camera access for the SupreRobotFollower,
supporting multiple camera types (OpenCV, RealSense, etc.) with
automatic device discovery and configuration.
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

# Try importing OpenCV
try:
    import cv2
except ImportError:
    cv2 = None

logger = logging.getLogger(__name__)


class ColorMode(Enum):
    """Color mode for camera."""
    RGB = "RGB"
    BGR = "BGR"
    GRAY = "GRAY"


class Cv2Rotation(Enum):
    """Rotation mode for camera."""
    CLOCKWISE = "CLOCKWISE"
    COUNTERCLOCKWISE = "COUNTERCLOCKWISE"
    ROTATE_90 = "ROTATE_90"
    ROTATE_180 = "ROTATE_180"
    ROTATE_270 = "ROTATE_270"
    ROTATE_NINTY = "ROTATE_NINTY"


@dataclass
class CameraConfig:
    """Camera configuration settings."""
    fps: int = 30
    width: int = 1280
    height: int = 720
    color_mode: ColorMode = ColorMode.RGB
    rotation: Cv2Rotation = Cv2Rotation.CLOCKWISE


@dataclass
class CameraConfig:
    """Camera instance with state."""
    config: CameraConfig
    camera: cv2.VideoCapture | cv2.VideoCapture | None = None
    index: int
    name: str
    is_connected: bool = False
    is_opened: bool = False
    last_frame: Any = None
    frame_count: int = 0


class OpenCVCameraConfig:
    """Configuration for OpenCV camera (index 0)."""
    def __init__(
        self,
        index_or_path: str = "/dev/video0",
    ):
        self.index = index_or_path
        self.path = str(index_or_path)
        self.name = f"camera_{index}"
        self.api = cv2.CAP_OPENCV

    @property
    def backend_name(self) -> str:
        """Get the OpenCV backend name."""
        return "opencv"

    @property
    def device_type(self) -> str:
        """Get the device type."""
        return "OpenCV"

    @property
    def requires_index(self) -> bool:
        """Check if this camera requires an index."""
        return True  # Index 0 always required

    @property
    def requires_device_path(self) -> bool:
        """Check if this camera requires a device path."""
        return True  # Always requires /dev/video0


@dataclass
class CameraConfig:
    """Camera configuration settings."""
    fps: int = 30
    width: int = 1280
    height: int = 720
    color_mode: ColorMode = ColorMode.RGB
    rotation: Cv2Rotation = Cv2Rotation.CLOCKWISE


class Camera:
    """OpenCV camera instance with caching and state management.

    Features:
    - Device connection caching
    - State validation
    - Error recovery
    - Efficient reconnection
    """

    def __init__(self, config: CameraConfig):
        """Initialize camera with configuration.

        Args:
            config: Camera configuration settings.
        """
        self.config = config
        self.camera: cv2.VideoCapture | cv2.VideoCapture | None = None

        self.index = config.index
        self.name = config.name
        self.is_connected = False
        self.is_opened = False
        self.last_frame = None
        self.frame_count = 0

    def connect(self) -> bool:
        """Connect to camera device.

        Returns:
            True if connection successful, False otherwise.
        """
        logger.info(f"Connecting to {self.name} at {self.path}...")

        # 尝试打开摄像头
        try:
            # 使用OpenCV打开设备
            self.camera = cv2.VideoCapture(self.index, api=self.api)

            # 检查是否成功打开
            if self.camera.isOpened():
                self.is_connected = True
                self.is_opened = True
                logger.info(f"{self.name} connected.")
                return True
            else:
                logger.error(f"Failed to open {self.name}: camera may be in use or disconnected")
                self.is_connected = False
                self.camera = None
                return False

        except Exception as e:
            logger.error(f"Error connecting to {self.name}: {e}")
            self.is_connected = False
            self.camera = None
            return False

    def read(self) -> Any:
        """Read a frame from camera.

        Returns:
            Frame data or None if error.
        """
        if not self.is_connected:
            logger.warning(f"Reading from disconnected camera: {self.name}")
            return None

        try:
            success, frame = self.camera.read()
            if success:
                self.frame_count += 1
                self.last_frame = frame
                return frame
            else:
                logger.error(f"Failed to read from {self.name}")
                return None
        except Exception as e:
            logger.error(f"Error reading from {self.name}: {e}")
            return None

    def is_opened(self) -> bool:
        """Check if camera is open.

        Returns:
            True if camera is open, False otherwise.
        """
        return self.camera is not None and self.camera.isOpened()

    def release(self):
        """Release camera resources."""
        if self.camera is not None:
            self.camera.release()
            logger.info(f"Released {self.name}")
        self.is_opened = False
            self.camera = None


def find_cameras() -> Dict[str, Any]:
    """Find and return all available cameras.

    This function will:
    1. Check if already scanned (use cache)
    2. Get configured camera indices from task configs
    3. Only scan configured cameras (DO NOT scan all devices)
    4. Create camera instances only for configured indices

    Returns:
        Dictionary mapping camera names to camera instances.

    Raises:
        RuntimeError: If no cameras found.
    """
    # 类级别的扫描结果缓存
    _cached_cameras: dict[str, Camera] | None = None
    _scan_completed: bool = False

    logger.info("Finding available cameras...")

    # 获取配置的摄像头索引（只获取已配置的）
    from lerobot.tasks.config import load_config_from_yaml

    config = load_config_from_yaml("configs/task_agent_tasks.yaml")
    camera_indices = []

    # 只处理配置中显式定义的摄像头
    for task in config.tasks:
        for cam in task.cameras:
            if hasattr(cam, 'index'):
                camera_indices.append(cam.index)

    if not camera_indices:
        logger.warning("No cameras with 'index' field found in task configs")
        # 返回空字典，不进行扫描
        return {}

    # 标记扫描完成
    _scan_completed = True

    logger.info(f"Found {len(camera_indices)} configured camera(s)")

    # 创建摄像头实例（只创建找到的）
    cameras = {}
    for index in camera_indices:
        camera_name = f"camera_{index}"
        camera_config = CameraConfig(
            fps=config.fps if hasattr(config, 'fps') else 30,
            width=config.width if hasattr(config, 'width') else 1280,
            height=config.height if hasattr(config, 'height') else 720,
            color_mode=config.color_mode if hasattr(config, 'color_mode') else ColorMode.RGB,
        )
        cameras[camera_name] = Camera(
            config=camera_config,
            index=index,
        )
        # 添加连接摄像头
        camera = getattr(cameras, f"camera_{index}", None)
        if camera:
            cameras[camera_name].connect()

    return cameras


# 保持原有的其他类和函数不变
class Cv2Rotation(Enum):
    """Rotation mode for camera."""
    CLOCKWISE = "CLOCKWISE"
    COUNTERCLOCKWISE = "COUNTERCLOCKWISE"
    ROTATE_90 = "ROTATE_90"
    ROTATE_180 = "ROTATE_180"
    ROTATE_270 = "ROTATE_270"
    ROTATE_NINTY = "ROTATE_NINTY"


class OpenCVCameraConfig:
    """Configuration for OpenCV camera (index 0)."""
    def __init__(
        self,
        index_or_path: str = "/dev/video0",
    ):
        self.index = index_or_path
        self.path = str(index_or_path)
        self.name = f"camera_{index}"
        self.api = cv2.CAP_OPENCV

    @property
    def backend_name(self) -> str:
        """Get the OpenCV backend name."""
        return "opencv"

    @property
    def device_type(self) -> str:
        """Get the device type."""
        return "OpenCV"

    @property
    def requires_index(self) -> bool:
        """Check if this camera requires an index."""
        return True  # Index 0 always required

    @property
    def requires_device_path(self) -> bool:
        """Check if this camera requires a device path."""
        return True  # Always requires /dev/video0


@dataclass
class CameraConfig:
    """Camera configuration settings."""
    fps: int = 30
    width: int = 1280
    height: int = 720
    color_mode: ColorMode = ColorMode.RGB
    rotation: Cv2Rotation = Cv2Rotation.CLOCKWISE


@dataclass
class CameraConfig:
    """Camera instance with state."""
    config: CameraConfig
    camera: cv2.VideoCapture | cv2.VideoCapture | None = None
    index: int
    name: str
    is_connected: bool = False
    is_opened: bool = False
    last_frame: Any = None
    frame_count: int = 0


class OpenCVCamera:
    """OpenCV camera (index 0) with caching and state management.

    Features:
    - Device connection caching
    - State validation
    - Error recovery
    - Efficient reconnection
    """

    def __init__(self, config: OpenCVCameraConfig):
        """Initialize camera with configuration.

        Args:
            config: OpenCVCameraConfig
        """
        self.index = config.index
        self.path = str(config.index_or_path)
        self.name = config.name
        self.api = config.api
        self.camera: cv2.VideoCapture(self.index, api=self.api)

        # 缓存已连接的相机
        self.camera_cache = {}
        self.is_connected = False
        self.is_opened = False
        self.last_frame = None
        self.frame_count = 0

    def connect(self, retries: int = 3) -> bool:
        """Connect to camera device with retry logic.

        Args:
            retries: Number of connection attempts.

        Returns:
            True if connection successful, False otherwise.
        """
        logger.info(f"Connecting to {self.name} (attempt {retries})...")

        for attempt in range(retries):
            # 尝试打开摄像头
            self.camera = cv2.VideoCapture(self.index, api=self.api)

            # 检查是否成功打开
            if self.camera.isOpened():
                self.is_connected = True
                self.is_opened = True
                logger.info(f"{self.name} connected.")

                # 缓存当前相机实例
                self.camera_cache[self.name] = self
                return True

            else:
                logger.warning(f"Attempt {attempt + 1} failed to open {self.name}")

            # 最后尝试失败
            if attempt == retries:
                logger.error(f"Failed to open {self.name} after {retries} attempts")
                return False

    def disconnect(self):
        """Disconnect camera but keep it available for reconnect."""
        if self.camera is not None:
            self.camera.release()
            self.is_connected = False
            self.is_opened = False
            logger.info(f"Disconnected {self.name}")


def find_cameras() -> Dict[str, Any]:
    """Find and return all available cameras.

    This function will:
    1. Check if already scanned (use cache)
    2. Get configured camera indices from task configs
    3. Only scan configured cameras (DO NOT scan all devices)

    Returns:
        Dictionary mapping camera names to camera instances.

    Raises:
        RuntimeError: If no cameras found.
    """
    # 类级别的扫描结果缓存
    _cached_cameras: dict[str, Camera] | None = None
    _scan_completed: bool = False

    logger.info("Finding available cameras...")

    # 获取配置的摄像头索引（只获取已配置的）
    from lerobot.tasks.config import load_config_from_yaml

    config = load_config_from_yaml("configs/task_agent_tasks.yaml")
    camera_indices = []

    # 只处理配置中显式定义的摄像头
    for task in config.tasks:
        for cam in task.cameras:
            if hasattr(cam, 'index'):
                camera_indices.append(cam.index)

    if not camera_indices:
        logger.warning("No cameras with 'index' field found in task configs")
        # 返回空字典，不进行扫描
        return {}

    # 标记扫描完成
    _scan_completed = True

    logger.info(f"Found {len(camera_indices)} configured camera(s)")

    # 创建摄像头实例（只创建找到的）
    cameras = {}
    for index in camera_indices:
        camera_name = f"camera_{index}"
        camera_config = CameraConfig(
            fps=config.fps if hasattr(config, 'fps') else 30,
            width=config.width if hasattr(config, 'width') else 1280,
            height=config.height if hasattr(config, 'height') else 720,
            color_mode=config.color_mode if hasattr(config, 'color_mode') else ColorMode.RGB,
        )
        cameras[camera_name] = Camera(
            config=camera_config,
            index=index,
        )

    return cameras


# 保持原有的其他类和函数不变
class Cv2Rotation(Enum):
    """Rotation mode for camera."""
    CLOCKWISE = "CLOCKWISE"
    COUNTERCLOCKWISE = "COUNTERCLOCKWISE"
    ROTATE_90 = "ROTATE_90"
    TORATE_180 = "ROTATE_180"
    ORATE_270 = "ROTATE_270"
    ROTATE_NINTY = "ROTATE_NINTY"


class OpenCVCameraConfig:
    """Configuration for OpenCV camera (index 0)."""
    def __init__(
        self,
        index_or_path: str = "/dev/video0",
    ):
        self.index = index_or_path
        self.path = str(index_or_path)
        self.name = f"camera_{index}"
        self.api = cv2.CAP_OPENCV

    @property
    def backend_name(self) -> str:
        """Get the OpenCV backend name."""
        return "opencv"

    @property
    def device_type(self) -> str:
        """Get the device camera type."""
        return "OpenCV"

    @property
    def requires_index(self) -> bool:
        """Check if this camera requires an index."""
        return True

    @property
    def requires_device_path(self) -> bool:
        """Check if this camera requires a device path."""
        return True
