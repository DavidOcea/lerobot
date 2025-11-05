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

def obj_conf() -> list:
    """目标物件的参数"""
    return [
        {"name" : "desk", # 位置和高度都有问题，需要调整
         "path" : "/home/smai/workspace/dc_dir/sim_lerobot/rf_object_workspace/Assemfinal_dest/urdf/Assemfinal_dest.urdf",
         "basePosition" : [0.0, -0.63, 0.0],
         "globalScaling": 0.00085,
         "useFixedBase" : True,
         "color": (1.0, 1.0, 1.0, 1.0)
         },
        #  {"name" : "workdesk", # 这个可以不加，目前没有调好
        #  "path" : "/home/smai/workspace/dc_dir/sim_lerobot/rf_object_workspace/Assemfinal_workdesk/urdf/Assemfinal_workdesk.urdf",
        #  "basePosition" : [0.0, -0.6, 0.05],
        #  "globalScaling": 100.5,
        #  "useFixedBase" : True,
        #  "color": (0.8, 0.8, 0.8, 1.0)
        #  },
        #  {"name" : "stick2", # 短件
        #  "path" : "/home/smai/workspace/dc_dir/sim_lerobot/rf_object_workspace/Assem6--finalone2-stick2/urdf/Assem6--finalone2-stick2.urdf",
        #  "basePosition" : [0.0, -0.5, 0.72],
        #  "globalScaling": 1.0,
        #  "useFixedBase" : False,
        #  "color": (0.0, 0.0, 0.0, 1.0)
        #  },
         {"name" : "items2", #长件
         "path" : "/home/smai/workspace/dc_dir/sim_lerobot/rf_object_workspace/Assemfinal_items2/urdf/Assemfinal_items2.urdf",
         "basePosition" : [0.2, -0.55, 0.8],
         "globalScaling": 1.0,
         "useFixedBase" : False,
         "color": (0.0, 0.0, 0.0, 1.0)
         },
    ]

@RobotConfig.register_subclass("sim_robot")
@dataclass
class SimRobotConfig(RobotConfig):
    # 仿真环境参数
    headless: bool = False
    urdf_path: str = "/home/smai/workspace/dc_dir/sim_lerobot/rf2502_new_3/urdf/rf2502_new_3.urdf"
    obj_cfg: list = field(default_factory=obj_conf)
    
    # 相机配置
    cameras: dict[str, CameraConfig] = field(default_factory=sim_robot_cameras_config)
    
    # 安全参数
    max_relative_target: float = None #10.0 #0.1 改角度为了和机器对应 弧度 # 关节最大相对移动量
    disable_torque_on_disconnect: bool = True