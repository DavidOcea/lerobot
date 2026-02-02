# controller_switcher.py
import rclpy
from rclpy.node import Node
from controller_manager_msgs.srv import SwitchController
from lerobot.teleop_aligner import TeleopAligner
class ControllerSwitcher(Node):
    def __init__(self):
        # 创建一个独立的、临时的节点来调用服务
        super().__init__('controller_switcher_client')
        self.cli = self.create_client(SwitchController, '/supre_robot_follower/controller_manager/switch_controller')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('切换控制器服务不可用, 正在等待...')
        self.req = SwitchController.Request()

    def switch(self, activate_controllers, deactivate_controllers, strictness=SwitchController.Request.STRICT):
        self.req.activate_controllers = activate_controllers
        self.req.deactivate_controllers = deactivate_controllers
        self.req.strictness = strictness
        
        future = self.cli.call_async(self.req)
        self.get_logger().info(f"请求切换: 激活 {activate_controllers}, 停止 {deactivate_controllers}")
        
        # 等待服务调用完成
        rclpy.spin_until_future_complete(self, future)
        
        if future.result().ok:
            self.get_logger().info("控制器切换成功！")
            return True
        else:
            self.get_logger().error("控制器切换失败！")
            return False

# --- 如何在你的主程序中使用 ---
# from teleop_aligner import TeleopAligner
# from controller_switcher import ControllerSwitcher

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
    switcher = ControllerSwitcher()
    
    align_controllers = ['left_arm_trajectory_controller', 'right_arm_trajectory_controller']
    teleop_controllers = ['left_arm_controller','right_arm_controller']
    # --- 遥操作主逻辑 ---
    print("========= 进入对齐模式 =========")
    try:
        # 1. 切换到对齐模式 (激活轨迹控制器)
        if not switcher.switch(
            deactivate_controllers= teleop_controllers,
            activate_controllers= [],
        ):
            aligner.get_logger().error("无法停止arm controller！")
            raise RuntimeError("无法停止arm controller！")

        if not switcher.switch(
            activate_controllers= align_controllers,
            deactivate_controllers=[],
        ):
            aligner.get_logger().error("无法启动trajectory controller！")
            raise RuntimeError("无法启动trajectory controller！")
        
        # 2. 执行对齐
        leader_start_pos = [0.1, -0.5, 0.2, 0.8, 0.3, 0.0] 
        if aligner.align(leader_start_pos, 7, align_time_sec=6.0):
            aligner.get_logger().info("========= 对齐成功 =========")
        else:
            aligner.get_logger().error("对齐失败！")

        # 5. 启动你的实时遥操作程序     
        # 3. 对齐成功，切换到实时遥操作模式
        if not switcher.switch(
            activate_controllers=[],
            deactivate_controllers=align_controllers
        ):  
             aligner.get_logger().error("无法停止trajectory controller！")

        if not switcher.switch(
            activate_controllers=teleop_controllers,
            deactivate_controllers=[],
        ):  
             aligner.get_logger().error("无法启动arm controller！")
        # 4. 在这里启动你的高频遥操作Publisher和循环
        aligner.get_logger().info("========= 进入实时遥操作模式 =========")
        print("========= 进入实时遥操作模式 =========")
        # teleop_publisher_node = ArmTeleopNode() # 启动你的高频发布节点
        # rclpy.spin(teleop_publisher_node)
        #pass

    except (KeyboardInterrupt, RuntimeError) as e:
        print(e)
        aligner.get_logger().error(f"程序终止: {e}")
    finally:
        # 清理
        print("程序退出")
        aligner.get_logger().info("程序退出，正在停止所有控制器...")
        #switcher.switch(activate_controllers=[], deactivate_controllers=['right_arm_trajectory_controller', 'right_arm_position_controller'])
        aligner.destroy_node()
        switcher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()