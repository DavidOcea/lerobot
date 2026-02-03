# teleop_aligner.py
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration
import threading
from action_msgs.msg import GoalStatus

class TeleopAligner(Node):
    def __init__(self, leader_joint_names,follower_joint_names):
        super().__init__('teleop_aligner')
        self.leader_joint_names = leader_joint_names
        self.follower_joint_names = follower_joint_names
        self.leader_current_joints = None
        self.follower_current_joints = None
        self.lock = threading.Lock()

        # 1. 创建Action Client以发送对齐轨迹
        self.right_arm_traj_action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/supre_robot_follower/right_arm_trajectory_controller/follow_joint_trajectory' # 确保这是你正确的Action名称
        )

        self.left_arm_traj_action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/supre_robot_follower/left_arm_trajectory_controller/follow_joint_trajectory' # 确保这是你正确的Action名称
        )
        # 2. 创建Subscriber以获取follower的当前位置
        # 注意：使用JointState作为消息类型，因为dynamic_joint_states发布的是这个类型
        self.leader_joint_state_sub = self.create_subscription(
            JointState,
            '/supre_robot_leader/joint_states', # 确保这是你正确的关节状态话题
            self.leader_joint_state_callback,
            10
        )
        self.follower_joint_state_sub = self.create_subscription(
            JointState,
            '/supre_robot_follower/joint_states', # 确保这是你正确的关节状态话题
            self.follower_joint_state_callback,
            10
        )
        self.get_logger().info("对齐器节点已启动。")


    def follower_joint_state_callback(self, msg: JointState):
        with self.lock:
            # 只在第一次收到消息时存储关节位置
            if self.follower_current_joints is None:
                # 将收到的关节状态按我们期望的顺序排序
                ordered_positions = [0.0] * len(self.follower_joint_names)
                for i, name in enumerate(self.follower_joint_names):
                    try:
                        idx = msg.name.index(name)
                        ordered_positions[i] = msg.position[idx]
                    except ValueError:
                        self.get_logger().error(f"在/supre_robot_leader/joint_states 中找不到关节'{name}'！")
                        return # 如果缺少关节，则不更新
                
                self.follower_current_joints = ordered_positions
                self.get_logger().info(f"成功获取到Follower的初始位置: {self.follower_current_joints}")
    def leader_joint_state_callback(self, msg: JointState):
        with self.lock:
            # 只在第一次收到消息时存储关节位置
            if self.leader_current_joints is None:
                # 将收到的关节状态按我们期望的顺序排序
                ordered_positions = [0.0] * len(self.leader_joint_names)
                for i, name in enumerate(self.leader_joint_names):
                    try:
                        idx = msg.name.index(name)
                        ordered_positions[i] = msg.position[idx]
                    except ValueError:
                        self.get_logger().error(f"在/supre_robot_leader/joint_states 中找不到关节'{name}'！")
                        return # 如果缺少关节，则不更新
                
                self.leader_current_joints = ordered_positions
                self.get_logger().info(f"成功获取到Leader的初始位置: {self.leader_current_joints}")

    def align(self, leader_initial_joints, left_arm_joint_num,align_time_sec=5.0):
        self.get_logger().info("正在等待Follower和Leader的初始关节状态...")
        # 等待回调函数获取到初始位置
        while self.follower_current_joints is None or self.leader_current_joints is None: 
            rclpy.spin_once(self, timeout_sec=0.1)
            if not rclpy.ok(): return False

        self.get_logger().info("正在等待Action Server连接...")
        if not self.left_arm_traj_action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Action Server连接超时！无法执行对齐。")
            return False
        if not self.right_arm_traj_action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Action Server连接超时！无法执行对齐。")
            return False
        result = self.align_one_arm(self.follower_joint_names[0:left_arm_joint_num],
                                    self.leader_current_joints[0:left_arm_joint_num],
                                    self.follower_current_joints[0:left_arm_joint_num],
                                    self.left_arm_traj_action_client,
                                    align_time_sec)
        if not result:
            self.get_logger().error("左臂对齐失败！")
            return False
        result = self.align_one_arm(self.follower_joint_names[left_arm_joint_num:],
                                    self.leader_current_joints[left_arm_joint_num:],
                                    self.follower_current_joints[left_arm_joint_num:],
                                    self.right_arm_traj_action_client,
                                    align_time_sec)
        if not result:
            self.get_logger().warn("右臂对齐失败！")
            return False
        return True
    def align_one_arm(self, joint_names,leader_joints, follower_joints,traj_action_client,align_time_sec=5.0) -> bool:
        goal_msg = FollowJointTrajectory.Goal()
        trajectory = JointTrajectory()
        trajectory.joint_names = joint_names

        # 创建一个包含两个点的轨迹：起点(当前位置)和终点(leader位置)
        # Point 1: 起始点 (当前位置,时间为0)
        start_point = JointTrajectoryPoint()
        start_point.positions = follower_joints
        start_point.time_from_start = Duration(sec=0, nanosec=0)

        # Point 2: 终点 (leader位置,在指定时间到达)
        end_point = JointTrajectoryPoint()
        end_point.positions = leader_joints
        end_point.time_from_start = Duration(sec=int(align_time_sec), nanosec=0)

        trajectory.points.append(start_point)
        trajectory.points.append(end_point)
        goal_msg.trajectory = trajectory

        self.get_logger().info(f"发送对齐目标，将在 {align_time_sec} 秒内完成...")
        future = traj_action_client.send_goal_async(goal_msg)
        
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error("对齐目标被拒绝！")
            return False

        get_result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, get_result_future)
        
        # 从 future 中获取 _GetResult_Response 对象
        result_response = get_result_future.result()
        
        # 【关键修复点】
        if result_response.status == GoalStatus.STATUS_SUCCEEDED:
            # 访问嵌套的 .result 对象来获取 error_code
            final_result = result_response.result
            if final_result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
                self.get_logger().info("轨迹执行成功！")
                return True
            else:
                # 虽然Action状态是SUCCEEDED，但控制器内部可能报告了错误
                self.get_logger().warn(f"轨迹执行完成，但有错误码: {final_result.error_code} - {final_result.error_string}")
                return False # 或者根据你的逻辑返回True
        else:
            self.get_logger().error(f"轨迹执行失败，最终状态: {result_response.status}")
            return False

def main(args=None):
    rclpy.init(args=args)
    leader_joint_name_prefix = "leader_"
    follower_joint_name_prefix = "follower_"
    observation_joint_names= [
        'left_arm_joint_1',
        'left_arm_joint_2',
        'left_arm_joint_3',
        'left_arm_joint_4',
        'left_arm_joint_5',
        'left_arm_joint_6',
        'left_arm_joint_7',
        'right_arm_joint_1',
        'right_arm_joint_2',
        'right_arm_joint_3',
        'right_arm_joint_4',
        'right_arm_joint_5',
        'right_arm_joint_6',
        'right_arm_joint_7',
    ]

    leader_joint_names = [f"{leader_joint_name_prefix}{name}" for name in observation_joint_names]
    follower_joint_names = [f"{follower_joint_name_prefix}{name}" for name in observation_joint_names]
    aligner = TeleopAligner(leader_joint_names,follower_joint_names)

    # --- 这里是你的遥操作主逻辑 ---
    try:
        # 1. 模拟从Leader设备获取初始位置
        # 在真实应用中，你需要从你的leader机器人或设备读取
        leader_start_pos = [0.1, -0.5, 0.2, 0.8, 0.3, 0.0] 
        
        # 2. 执行对齐
        if aligner.align(leader_start_pos, 7, align_time_sec=6.0):
            # 3. 对齐成功后，在这里启动你的实时遥操作循环
            aligner.get_logger().info("========= 进入实时遥操作模式 =========")
            # a. 创建一个高频发布器 (例如，使用JointGroupPositionController的Topic)
            # b. 在一个 while rclpy.ok() 循环中:
            #      - 读取leader的实时位置
            #      - 发布给follower
            #      - time.sleep(0.02) # 控制频率
            pass

    except KeyboardInterrupt:
        pass
    finally:
        aligner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()