import time
import math
from supre_robot_hardware_manager import SupreRobotHardwareManager # 导入我们的管理器
import logging

logging.basicConfig(level=logging.INFO)

def main():
    # 1. 初始化管理器
    robot_manager = SupreRobotHardwareManager(config_path="supre_robot_config.yaml")
    if not robot_manager.init():
        print("Failed to initialize robot hardware manager. Exiting.")
        return

    try:
        # 2. 激活硬件
        if not robot_manager.activate():
            print("Failed to activate robot hardware. Exiting.")
            return

        # 3. 主控制循环
        print("\n--- Starting Main Control Loop (Press Ctrl+C to exit) ---")
        
        control_frequency = 100
        control_period = 1.0 / control_frequency
        start_loop_time = time.time()
        
        # 目标指令向量，与 `joint_order` 严格对应
        target_positions = list(robot_manager.commands)

        while True:
            loop_start = time.perf_counter()
            
            # a. Read: 从管理器读取整个机器人的状态
            current_positions = robot_manager.read()[0]
            
            # 打印状态（简化版，只打印部分关节）
            print(f"Time: {time.time() - start_loop_time:5.2f}s | "
                  f"Left Arm J1 Pos: {current_positions[0]:6.2f} | "
                  f"Left Gripper Pos: {current_positions[6]:6.2f} | "
                  f"Right Arm J1 Pos: {current_positions[7]:6.2f} | "
                  f"Right Gripper Pos: {current_positions[13]:6.2f}", end='\r')


            # b. Controller Logic: 计算新的目标位置
            #    这里只是一个简单的正弦波示例，让手臂动起来
            elapsed_time = time.time() - start_loop_time
            amplitude = 30.0
            frequency = 0.2
            
            # 让左右臂的6个关节做正弦运动
            for i in [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]:
                phase = i * (math.pi / 4)
                target_positions[i] = abs(amplitude * math.sin(2 * math.pi * frequency * elapsed_time + phase))
            
            # 让夹爪在 0.2 和 0.8 之间开合
            gripper_pos = 0.5 + 0.3 * math.sin(2 * math.pi * 0.5 * elapsed_time)
            target_positions[6] = gripper_pos  # Left gripper
            target_positions[13] = gripper_pos # Right gripper

            # c. Write: 将完整的指令向量写入管理器
            robot_manager.write(target_positions)
            
            loop_end = time.perf_counter()
            sleep_time = control_period - (loop_end - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nCtrl+C pressed. Shutting down.")
    except Exception as e:
        logging.exception("An error occurred in the control loop")
    finally:
        # 4. 确保在退出时停用硬件
        robot_manager.deactivate()

if __name__ == "__main__":
    main()