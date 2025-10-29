# lerobot/src/lerobot/robots/sim_robot/config_sim_robot.py
from dataclasses import dataclass, field
from lerobot.cameras.configs import CameraConfig, Cv2Rotation
# from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.pybullet.pybullet_camera import PyBulletCameraConfig
from ..config import RobotConfig

def sim_robot_cameras_config() -> dict[str, CameraConfig]:
    """仿真环境相机配置，匹配camera_cfg参数"""
    # import pdb; pdb.set_trace()

    return {
        "head_cam": PyBulletCameraConfig(
            # type="pybullet",  # 用于后续动态创建相机
            # id="head_cam",  # 仿真相机标识
            position=(0.6, 0, 0.9),  # 调整相机位置
            orientation=(0.5, -0.5, -0.5, 0.5),  # 调整朝向
            width=640,
            height=480
        ),
        "right_wrist_cam": PyBulletCameraConfig(
            position=(0.6, 0, 0.9),  # 调整相机位置
            orientation=(0.5, -0.5, -0.5, 0.5),  # 调整朝向
            width=640,
            height=480
        ),
        "left_wrist_cam": PyBulletCameraConfig(
            position=(0.6, 0, 0.9),  # 调整相机位置
            orientation=(0.5, -0.5, -0.5, 0.5),  # 调整朝向
            width=640,
            height=480
        ),
    }

@RobotConfig.register_subclass("sim_robot")
@dataclass
class SimRobotConfig(RobotConfig):
    # 仿真环境参数
    headless: bool = False
    is_manual: bool = False
    urdf_path: str = "/home/zzj/dc_space/lerobot-main-250612/sim_lerobot_example/sim_example/rf2502_new_3/urdf/rf2502_new_3.urdf"
    
    # 相机配置
    cameras: dict[str, CameraConfig] = field(default_factory=sim_robot_cameras_config)
    
    # 安全参数
    max_relative_target: float = 0.1  # 关节最大相对移动量
    disable_torque_on_disconnect: bool = True