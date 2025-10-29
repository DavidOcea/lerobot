# lerobot/src/lerobot/robots/sim_robot/simulator.py
import pybullet as p
import pybullet_data
import numpy as np
import math
from typing import List, Dict, Any, Tuple

# 保持用户提供的camera_cfg配置
camera_cfg = {
    "head_cam": {
        "width": 640,
        "height": 480,
        "fov": 70,
        "link_name": "body_roll",
        "offset_pos": [0.0, 0.25, 0.18],
        "offset_orn": p.getQuaternionFromEuler([0, math.radians(-10), math.radians(-90)])
    },
    "right_wrist_cam": {
        "width": 640,
        "height": 480,
        "fov": 60,
        "link_name": "gripper_flex_right",
        "offset_pos": [-0.0, -0.03, 0.06],
        "offset_orn": p.getQuaternionFromEuler([0, math.radians(30), math.radians(-90)])
    },
    "left_wrist_cam": {
        "width": 640,
        "height": 480,
        "fov": 60,
        "link_name": "gripper_flex_left",
        "offset_pos": [0.0, -0.03, 0.06],
        "offset_orn": p.getQuaternionFromEuler([0, math.radians(30), math.radians(-90)])
    }
}

class Simulator:
    """保持用户实现的仿真逻辑，仅调整接口适配"""
    def __init__(self, headless: bool = False, is_manual: bool = False):
        self.physics_client = p.connect(p.DIRECT if headless else p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setPhysicsEngineParameter(fixedTimeStep=0.01, numSubSteps=10)
        
        self.plane_id = p.loadURDF("plane.urdf")
        
        # 关节名称与ID映射（保持用户定义）
        self.joint_name2id = {
            "waist_flex":1, "body_roll":3, "shoulder_roll_right":4,
            "shoulder_roll_left":12, "shoulder_lift_right":5, "shoulder_lift_left":13,
            "elbow_roll_right":6, "elbow_roll_left":14, "elbow_flex_right":7, "elbow_flex_left":15,
            "wrist_roll_right":8, "wrist_roll_left":16, "gripper_flex_right":9, "gripper_flex_left":17,
            "gripper_right_1":10, "gripper_right_2":11, "gripper_left_1":18, "gripper_left_2":19
        }

        # 活动关节（保持用户定义）
        self.flexible_name = [
            "shoulder_roll_right", "shoulder_lift_right", "elbow_roll_right", 
            "elbow_flex_right", "wrist_roll_right", "gripper_flex_right",
            "gripper_right_1", "gripper_right_2", "shoulder_roll_left",
            "shoulder_lift_left", "elbow_roll_left", "elbow_flex_left",
            "wrist_roll_left", "gripper_flex_left", "gripper_left_1", "gripper_left_2"
        ]
        self.flexible_joint = {name: self.joint_name2id[name] for name in self.flexible_name}
        self.action_dim = len(self.flexible_name)

        # 关节范围参数（保持用户定义）
        self.action_low = np.array([-2.9671, -3.1416, -1.1345, -0.1745, -1.1345, -1.5707, 0.0, 0.0, 
                                   -2.9671, -3.1416, -1.1345, -0.1745, -1.1345, -1.5708, 0.0, 0.0])
        self.action_high = np.array([2.9671, 0.1745, 1.1345, 1.9199, 1.13345, 1.5707, 0.025, 0.025, 
                                    2.9671, 0.1745, 1.1345, 1.9198, 1.1345, 1.5708, 0.025, 0.025])
        self.action_scale = (self.action_high - self.action_low) / 2.0
        self.action_bias = (self.action_high + self.action_low) / 2.0

        # 加载机器人模型（使用配置中的路径）
        from .config_sim_robot import SimRobotConfig
        self.robot_id = p.loadURDF(
            SimRobotConfig().urdf_path,
            basePosition=[0, 0, 0.05],
            useFixedBase=True
        )

        # 加载场景物体（保持用户定义）
        self.objects = self._load_objects()
        self.is_manual = is_manual
        self.cube_hight = 0.5  # 从_load_objects中提取为类属性

    # 保持用户实现的其他方法：_load_objects, get_observation, step, reset, get_camera_images等
    def _load_objects(self) -> Dict:
        objects = {}
        objects["cube"] = p.loadURDF(
            "cube.urdf",
            basePosition=[0.0, -0.5, 0.5],
            globalScaling=0.08
        )
        # objects["desk"] = p.loadURDF(
        #     "/home/smai/workspace/dc_dir/sim_lerobot/rf_object_workspace/Assemfinal_dest/urdf/Assemfinal_dest.urdf",
        #     basePosition=[0.0, -1.8, 0.05],
        #     globalScaling=0.5
        # )
        # objects["workdesk"] = p.loadURDF(
        #     "/home/smai/workspace/dc_dir/sim_lerobot/rf_object_workspace/Assemfinal_workdesk/urdf/Assemfinal_workdesk.urdf",
        #     basePosition=[0.0, -1.6, 0.67],
        # )
        # objects["stick2"] = p.loadURDF(
        #     "/home/smai/workspace/dc_dir/sim_lerobot/rf_object_workspace/Assem6--finalone2-stick2/urdf/Assem6--finalone2-stick2.urdf",
        #     basePosition=[0.0, -1.6, 0.68],
        # )
        # objects["items2"] = p.loadURDF(
        #     "/home/smai/workspace/dc_dir/sim_lerobot/rf_object_workspace/Assemfinal_items2/urdf/Assemfinal_items2.urdf",
        #     basePosition=[0.0, -1.6, 0.68],
        # )
        # 桌子
        objects["desk"] = p.loadURDF(
            "/home/zzj/dc_space/lerobot-main-250612/sim_lerobot_example/sim_example/workspace/Assemfinal_dest/urdf/Assemfinal_dest.urdf",
            basePosition=[0.0, -1.8, 0.05],
            globalScaling=0.5,
            useFixedBase=True
        )
        # 工作台
        objects["workdesk"] = p.loadURDF(
            "/home/zzj/dc_space/lerobot-main-250612/sim_lerobot_example/sim_example/workspace/Assemfinal_workdesk/urdf/Assemfinal_workdesk.urdf",
            basePosition=[0.0, -1.6, 0.67],
            useFixedBase=True
        )
        # 杆子
        objects["stick2"] = p.loadURDF(
            "/home/zzj/dc_space/lerobot-main-250612/sim_lerobot_example/sim_example/workspace/Assem6--finalone2-stick2/urdf/Assem6--finalone2-stick2.urdf",
            basePosition=[0.0, -1.6, 0.68],
        )
        # 零件
        objects["items2"] = p.loadURDF(
            "/home/zzj/dc_space/lerobot-main-250612/sim_lerobot_example/sim_example/workspace/Assemfinal_items2/urdf/Assemfinal_items2.urdf",
            basePosition=[0.0, -1.6, 0.68],
        )
        return objects

    def get_observation(self) -> np.ndarray:
        joint_pos = []
        joint_vel = []
        for name in self.flexible_name:
            idx = self.flexible_joint[name]
            pos, vel, _, _ = p.getJointState(self.robot_id, idx)
            joint_pos.append(pos)
            joint_vel.append(vel)
        
        cube_pos, _ = p.getBasePositionAndOrientation(self.objects["cube"])
        idx_list = [self.flexible_joint[name] for name in self.flexible_name 
                   if "gripper_left" in name or "gripper_right" in name]
        ee_pos_list = []
        for ee_link_idx in idx_list:
            ee_pos = p.getLinkState(self.robot_id, ee_link_idx)[0]
            ee_pos_list.extend(ee_pos)
        return np.concatenate([joint_pos, joint_vel, ee_pos_list, cube_pos])

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        action = action * self.action_scale + self.action_bias
        ia = 0
        for idx in self.flexible_joint.values():
            p.setJointMotorControl2(
                self.robot_id,
                idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=action[ia],
                force=500
            )
            ia += 1
        
        p.stepSimulation()
        next_obs = self.get_observation()
        
        ee_pos = next_obs[-15:-3]
        ee_pos_list = [ee_pos[i:i+3] for i in range(0, len(ee_pos), 3)]
        cube_pos = next_obs[-3:]
        
        distance_reward = 0.0
        distances = 0.0
        for epos in ee_pos_list:
            distance = np.linalg.norm(epos - cube_pos)
            distance_reward += max(0.0, 1.0 - distance)
            distances += distance
        distance_reward /= 4
        distance = distances / 4

        gripper_pos = action[-1]
        grasp_reward = 0.0
        if gripper_pos < 0.015 and distance < 0.025:
            grasp_reward = 1.0
        
        lift_reward = 0.0
        if grasp_reward > 0 and cube_pos[2] - self.cube_hight > 0.05:
            lift_reward = 2.0
        
        total_reward = distance_reward + grasp_reward + lift_reward
        done = cube_pos[2] - self.cube_hight > 0.15
        
        return next_obs, total_reward, done, {"distance": distance}

    def reset(self) -> np.ndarray:
        p.resetBasePositionAndOrientation(
            self.objects["cube"], 
            [0.0, -0.5, 0.5],
            [0, 0, 0, 1]
        )
        
        for name in self.flexible_name:
            idx = self.flexible_joint[name]
            p.resetJointState(self.robot_id, idx, targetValue=0)
            p.setJointMotorControl2(
                self.robot_id, idx, controlMode=p.POSITION_CONTROL, targetPosition=0
            )
        return self.get_observation()

    def get_camera_images(self) -> Dict[str, np.ndarray]:
        images = {}
        for cam_name, cfg in camera_cfg.items():
            link_id = self.joint_name2id[cfg["link_name"]]
            link_state = p.getLinkState(self.robot_id, link_id, computeForwardKinematics=True)
            link_pos = link_state[0]
            link_orn = link_state[1]
            
            cam_pos, cam_orn = p.multiplyTransforms(
                link_pos, link_orn,
                cfg["offset_pos"], cfg["offset_orn"]
            )
            
            target_distance = 0.5
            forward_dir = p.rotateVector(cam_orn, [1, 0, 0])
            target_pos = [
                cam_pos[0] + forward_dir[0] * target_distance,
                cam_pos[1] + forward_dir[1] * target_distance,
                cam_pos[2] + forward_dir[2] * target_distance
            ]
            
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=cam_pos,
                cameraTargetPosition=target_pos,
                cameraUpVector=p.rotateVector(cam_orn, [0, 0, 1])
            )
            
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=cfg["fov"],
                aspect=cfg["width"] / cfg["height"],
                nearVal=0.01,
                farVal=5.0
            )
            
            _, _, rgb, _, _ = p.getCameraImage(
                width=cfg["width"],
                height=cfg["height"],
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix
            )
            
            rgb_array = np.array(rgb, dtype=np.uint8).reshape(
                cfg["height"], cfg["width"], 4
            )[:, :, :3]
            images[cam_name] = rgb_array
        return images
    
    def get_joint_states(self) -> Tuple[np.ndarray, np.ndarray]:
        positions = []
        velocities = []
        for name in self.flexible_name:
            idx = self.flexible_joint[name]
            pos, vel, _, _ = p.getJointState(self.robot_id, idx)
            positions.append(pos)
            velocities.append(vel)
        return np.array(positions), np.array(velocities)

    def close(self):
        p.disconnect(self.physics_client)