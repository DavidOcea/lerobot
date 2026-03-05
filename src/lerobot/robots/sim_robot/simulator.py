# lerobot/src/lerobot/robots/sim_robot/simulator.py
import pybullet as p
import pybullet_data
import numpy as np
import math
from typing import List, Dict, Any, Tuple
from .config_sim_robot import SimRobotConfig

# 保持用户提供的camera_cfg配置
camera_cfg = {
    "head_cam": {
        "width": 640,
        "height": 480,
        "fov": 70,
        "link_name": "body_roll",
        # 空中全局视角
        "offset_pos": [0.0, 0.65, 0.18], # 相机调高40公分 "offset_pos": [0.0, 0.45, 0.10], # 相机调高15公分 向后移动5公分
        "offset_orn": p.getQuaternionFromEuler([0, math.radians(-10), math.radians(-90)])
        #  脸部视角
        # "offset_pos": [0.0, 0.25, 0.18],
        # "offset_orn": p.getQuaternionFromEuler([0, math.radians(-10), math.radians(-90)])
    },
    "right_wrist_cam": {
        "width": 640,
        "height": 480,
        "fov": 60,
        "link_name": "gripper_flex_right",
        "offset_pos": [-0.0, -0.03, 0.06],
        "offset_orn": p.getQuaternionFromEuler([0, math.radians(30), math.radians(-90)])
    },
    "right_wrist_cam2": {
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
    def __init__(self, headless: bool = False):
        self.physics_client = p.connect(p.DIRECT if headless else p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        # p.setPhysicsEngineParameter(fixedTimeStep=0.01, numSubSteps=10)
        # 配置物理参数
        p.setPhysicsEngineParameter(
            fixedTimeStep=0.01,
            numSubSteps=10,
            erp=0.5,
            enableFileCaching=0
        )
        
        self.plane_id = p.loadURDF("plane.urdf")
        p.changeDynamics(self.plane_id, -1, lateralFriction=2.0)
        
        # 关节名称与ID映射（保持用户定义） 这个是对应仿真机器人关节序号
        self.joint_name2id = {
            "waist_flex":1, "body_roll":3, "shoulder_roll_right":4,
            "shoulder_roll_left":12, "shoulder_lift_right":5, "shoulder_lift_left":13,
            "elbow_roll_right":6, "elbow_roll_left":14, "elbow_flex_right":7, "elbow_flex_left":15,
            "wrist_roll_right":8, "wrist_roll_left":16, "gripper_flex_right":9, "gripper_flex_left":17,
            "gripper_right_1":10, "gripper_right_2":11, "gripper_left_1":18, "gripper_left_2":19,
            "gripper_left": 18, "gripper_right":10
        }


        # 活动关节（保持用户定义）
        # 为了对应真机修改
        self.flexible_name = [
            "shoulder_roll_left", "shoulder_lift_left", "elbow_roll_left", 
            "elbow_flex_left", "wrist_roll_left", "gripper_flex_left", "gripper_left",
            "shoulder_roll_right", "shoulder_lift_right", "elbow_roll_right", 
            "elbow_flex_right", "wrist_roll_right", "gripper_flex_right", "gripper_right",
            "body_roll", "waist_flex"
        ]

        # idx2joint 这个是对应真机的关节序号
        self.joint2idx = {
            "shoulder_roll_left":0, "shoulder_lift_left":1, "elbow_roll_left":2, 
            "elbow_flex_left":3, "wrist_roll_left":4, "gripper_flex_left":5, "gripper_left":6,
            "shoulder_roll_right":7, "shoulder_lift_right":8, "elbow_roll_right":9, 
            "elbow_flex_right":10, "wrist_roll_right":11, "gripper_flex_right":12, "gripper_right":13,
            "body_roll":14, "waist_flex":15
            }

        # joint_direction 关节动作方向
        self.joint_direction = [
            -1,1,-1,-1,1,-1,1,
            -1,-1,1,1,1,-1,1,
            -1,1
            ]
        
        # 这个是对应仿真机器人关节序号
        self.flexible_joint = {name: self.joint_name2id[name] for name in self.flexible_name}
        self.action_dim = len(self.flexible_name)

        # 关节范围参数（保持用户定义）
        # self.action_low = np.array([-2.9671, -3.1416, -1.1345, -0.1745, -1.1345, -1.5707, 0.0, 0.0, 
        #                            -2.9671, -3.1416, -1.1345, -0.1745, -1.1345, -1.5708, 0.0, 0.0])
        # self.action_high = np.array([2.9671, 0.1745, 1.1345, 1.9199, 1.13345, 1.5707, 0.025, 0.025, 
        #                             2.9671, 0.1745, 1.1345, 1.9198, 1.1345, 1.5708, 0.025, 0.025])
        
        # 为了对应真机修改
        self.action_low = np.array([
            -2.9671, -3.1416, -2.6179, -0.1745, -2.6179, -1.5707, 0.0,
            -2.9671, -3.1416, -2.6179, -0.1745, -2.6179, -1.5708, 0.0,
            -0.7853, 0
            ])
        self.action_high = np.array([
            2.9671, 0.1745, 2.6179, 1.9199, 2.6179, 1.5707, 0.025,
            2.9671, 0.1745, 2.6179, 1.9198, 2.6179, 1.5708, 0.025,
            0.7853, 0.7853
            ])
        self.action_scale = (self.action_high - self.action_low) / 2.0
        self.action_bias = (self.action_high + self.action_low) / 2.0

        # 加载机器人模型（使用配置中的路径）
        self.robot_id = p.loadURDF(
            SimRobotConfig().urdf_path,
            basePosition=[0, 0, 0.05],
            useFixedBase=True
        )

        # 加载场景物体（保持用户定义）
        self.objects = self._load_objects()
        # self.is_manual = is_manual
        # self.cube_hight = 0.5  # 从_load_objects中提取为类属性

    # 保持用户实现的其他方法：_load_objects, get_observation, step, reset, get_camera_images等
    def _load_objects(self) -> Dict:
        objects = {}
        # objects["cube"] = p.loadURDF(
        #     "cube.urdf",
        #     basePosition=[0.0, -0.5, 0.5],
        #     globalScaling=0.08
        # )
        # 桌子
        # objects["desk"] = p.loadURDF(
        #      "/home/smai/workspace/dc_dir/sim_lerobot/rf_object_workspace/Assemfinal_dest/urdf/Assemfinal_dest.urdf",
        #     basePosition=[0.0, -1.8, 0.05],
        #     globalScaling=0.5,
        #     useFixedBase=True
        # )
        # # 工作台
        # objects["workdesk"] = p.loadURDF(
        #     "/home/smai/workspace/dc_dir/sim_lerobot/rf_object_workspace/Assemfinal_workdesk/urdf/Assemfinal_workdesk.urdf",
        #     basePosition=[0.0, -1.6, 0.67],
        #     useFixedBase=True
        # )
        # # 杆子
        # objects["stick2"] = p.loadURDF(
        #     "/home/smai/workspace/dc_dir/sim_lerobot/rf_object_workspace/Assem6--finalone2-stick2/urdf/Assem6--finalone2-stick2.urdf",
        #     basePosition=[0.0, -1.6, 0.68],
        # )
        # # 零件
        # objects["items2"] = p.loadURDF(
        #     "/home/smai/workspace/dc_dir/sim_lerobot/rf_object_workspace/Assemfinal_items2/urdf/Assemfinal_items2.urdf",
        #     basePosition=[0.0, -1.6, 0.68],
        # )
        if len(SimRobotConfig().obj_cfg) == 0:
            return objects
        for cfg in SimRobotConfig().obj_cfg:
            # import pdb; pdb.set_trace()
            name = cfg['name']
            path = cfg['path']
            basePosition = cfg['basePosition']
            useFixedBase = cfg['useFixedBase']
            globalScaling = cfg['globalScaling']

            objects[name] = p.loadURDF(
                path,
                basePosition=basePosition,
                useFixedBase=useFixedBase,
                globalScaling=globalScaling
            )

            # 配置物理属性
            p.changeDynamics(
                objects[name], -1,  # 物体唯一ID # 链接索引（-1表示基座）
                mass=1.0,      # 新质量
                lateralFriction=1.0, #横向摩擦系数
                angularDamping=2.0, # 旋转摩擦系数
                linearDamping=0.5  # 弹性恢复系数
            )
            # 配置颜色
            self._set_object_color(objects[name], cfg['color'])

            # 短工件沿z轴旋转180度
            if name == "items2" or name == "desk":
                pos, orn = p.getBasePositionAndOrientation(objects[name])
                # 将四元数转换为欧拉角
                rpy = list(p.getEulerFromQuaternion(orn))
                # 仅修改Z轴旋转角度
                # import pdb; pdb.set_trace()
                rpy[2] += math.radians(180)  # Z轴是欧拉角中的第三个元素
                # 将欧拉角转换回四元数
                new_orn = p.getQuaternionFromEuler(rpy)
                # 更新工件姿态
                p.resetBasePositionAndOrientation(objects[name], pos, new_orn)


        return objects

    def _set_object_color(self, object_id, color):
        """
        设置物体颜色
        color: 四元组 (R, G, B, A)，值范围0-1
        """
        # 确保颜色值在有效范围内
        r, g, b, a = color
        r = max(0.0, min(1.0, r))
        g = max(0.0, min(1.0, g))
        b = max(0.0, min(1.0, b))
        a = max(0.0, min(1.0, a))
        
        # 获取物体的所有视觉形状
        # num_visual_shapes = p.getNumVisualShapes(object_id)
        
        # 为每个视觉形状设置颜色
        # for i in range(num_visual_shapes):
        p.changeVisualShape(
            object_id,
            linkIndex=-1,
            rgbaColor=[r, g, b, a]
        )
    
    def get_observation(self) -> np.ndarray:
        joint_pos = []
        joint_vel = []
        for name in self.flexible_name:
            idx = self.flexible_joint[name]
            pos, vel, _, _ = p.getJointState(self.robot_id, idx)
            # joint_pos.append(pos)
            # joint_vel.append(vel)
            # 加上电机转动方向
            ia = self.joint2idx[name]
            joint_pos.append(pos*self.joint_direction[ia])
            joint_vel.append(vel*self.joint_direction[ia])
        
        # cube_pos, _ = p.getBasePositionAndOrientation(self.objects["cube"])
        # idx_list = [self.flexible_joint[name] for name in self.flexible_name 
        #            if "gripper_left" in name or "gripper_right" in name]
        # ee_pos_list = []
        # for ee_link_idx in idx_list:
        #     ee_pos = p.getLinkState(self.robot_id, ee_link_idx)[0]
        #     ee_pos_list.extend(ee_pos)
        # return np.concatenate([joint_pos, joint_vel, ee_pos_list, cube_pos])
        return np.concatenate([joint_pos, joint_vel])

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        # action = action * self.action_scale + self.action_bias
        # ia = 0
        # import pdb; pdb.set_trace()
        for joint, idx in self.flexible_joint.items():
            ia = self.joint2idx[joint]
            # action_pos = math.radians(action[ia]) * self.joint_direction[ia]
            action_pos = action[ia] * self.joint_direction[ia]

            # 超限位处理
            if action_pos < self.action_low[ia]:
                print(joint+" ia: "+ str(ia) +" action="+ str(action_pos)+" 超最低限位 new_action="+str(self.action_low[ia]))
                action_pos = self.action_low[ia]
            if action_pos > self.action_high[ia]:
                print(joint+" ia: "+ str(ia) +" action="+ str(action_pos)+" 超最高限位 new_action="+str(self.action_low[ia]))
                action_pos = self.action_high[ia]

            p.setJointMotorControl2(
                self.robot_id,
                idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=action_pos,
                force=500
            )
            # ia += 1
            # print(joint+" : "+str(idx)+" : "+str(action_pos))
        
        p.stepSimulation()
        next_obs = self.get_observation()
        
        # ee_pos = next_obs[-15:-3]
        # ee_pos_list = [ee_pos[i:i+3] for i in range(0, len(ee_pos), 3)]
        # cube_pos = next_obs[-3:]
        
        # distance_reward = 0.0
        # distances = 0.0
        # for epos in ee_pos_list:
        #     distance = np.linalg.norm(epos - cube_pos)
        #     distance_reward += max(0.0, 1.0 - distance)
        #     distances += distance
        # distance_reward /= 4
        # distance = distances / 4

        # gripper_pos = action[-1]
        # grasp_reward = 0.0
        # if gripper_pos < 0.015 and distance < 0.025:
        #     grasp_reward = 1.0
        
        # lift_reward = 0.0
        # if grasp_reward > 0 and cube_pos[2] - self.cube_hight > 0.05:
        #     lift_reward = 2.0
        
        # total_reward = distance_reward + grasp_reward + lift_reward
        # done = cube_pos[2] - self.cube_hight > 0.15
        
        # return next_obs, total_reward, done, {"distance": distance}
        return next_obs

    # def reset(self) -> np.ndarray:
    #     p.resetBasePositionAndOrientation(
    #         self.objects["cube"], 
    #         [0.0, -0.5, 0.5],
    #         [0, 0, 0, 1]
    #     )
        
    #     for name in self.flexible_name:
    #         idx = self.flexible_joint[name]
    #         p.resetJointState(self.robot_id, idx, targetValue=0)
    #         p.setJointMotorControl2(
    #             self.robot_id, idx, controlMode=p.POSITION_CONTROL, targetPosition=0
    #         )
    #     return self.get_observation()

    def get_camera_images(self) -> Dict[str, np.ndarray]:
        images = {}
        for cam_name, cfg in camera_cfg.items():
            if cam_name not in SimRobotConfig().cameras:
                continue
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
            # positions.append(pos)
            # velocities.append(vel)
            # 加上电机转动方向
            ia = self.joint2idx[name]
            positions.append(pos*self.joint_direction[ia])
            velocities.append(vel*self.joint_direction[ia])
        return np.array(positions), np.array(velocities)

    def close(self):
        p.disconnect(self.physics_client)