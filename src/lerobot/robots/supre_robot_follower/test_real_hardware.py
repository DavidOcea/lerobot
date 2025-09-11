import time
import math
from pathlib import Path
import yaml
import numpy as np

# 导入机器人实现
from supre_robot_follower import SupreRobotFollower
from supre_robot_follower_config import SupreRobotFollowerConfig

# --- 测试参数 ---
CONFIG_FILE_PATH = Path(__file__)/ "supre_robot_config.yaml"
POSITION_TOLERANCE_DEG = 1.5  # 真实硬件的容差可以适当放宽
TRAJECTORY_DURATION_S = 6.0  # 轨迹执行的默认时长

def run_test_suite():
    """
    执行与真实硬件交互的集成测试套件 (重构后版本)。
    """
    print("="*50)
    print("=  STARTING REFACTORED H/W INTEGRATION TEST  =")
    print("="*50)
    
    test_passed_summary = {}
    robot = None
    initial_positions = {}
    
    try:
        # -----------------------------------------------------------------
        # 1. 初始化和连接测试
        # -----------------------------------------------------------------
        print("\n--- [TEST 1] Initialization and Connection ---")
        config = SupreRobotFollowerConfig(config_path=str(CONFIG_FILE_PATH))
        robot = SupreRobotFollower(config)
        robot.connect(calibrate=False)
        initial_positions = robot.get_current_position()
        print(f"PASS: Robot connected. Initial positions: {initial_positions}")
        test_passed_summary["Connection"] = True

        # -----------------------------------------------------------------
        # 2. send_action 测试 (瞬时移动)
        # -----------------------------------------------------------------
        print(f"\n--- [TEST 2] send_action (Instantaneous Move) ---")
        # 移动到初始位置 + 5度的位置
        target_action = {f"{name}.pos": pos + 1.0 for name, pos in initial_positions.items()}
        print(f"INFO: Sending instantaneous target: {target_action}")
        
        # send_action 是非阻塞的，所以我们需要等待
        robot.send_action(target_action)
        time.sleep(TRAJECTORY_DURATION_S) # 给足够的时间移动
        
        current_pos_map = robot.get_current_position()
        is_success = True
        for name in robot.observation_joint_names:
            target = target_action[f"{name}.pos"]
            actual = current_pos_map[name]
            if abs(target - actual) > POSITION_TOLERANCE_DEG:
                print(f"  FAIL: Joint '{name}' did not reach target. (Target: {target:.2f}, Actual: {actual:.2f})")
                is_success = False
        
        if is_success:
            print("PASS: send_action moved robot to target.")
        test_passed_summary["send_action"] = is_success

        # -----------------------------------------------------------------
        # 3. execute_trajectory 测试 (平滑移动)
        # -----------------------------------------------------------------
        print(f"\n--- [TEST 3] execute_trajectory (Smooth Move) ---")
        # 从当前位置平滑移动到初始位置 - 5度的位置
        goal_action = {f"{name}.pos": initial_positions[name] + 5.0 for name in robot.observation_joint_names}
        
        print(f"INFO: Executing {TRAJECTORY_DURATION_S}s trajectory to: {goal_action}")
        start_time = time.time()
        robot.execute_trajectory(goal_action, duration=TRAJECTORY_DURATION_S)
        end_time = time.time()
        
        print(f"INFO: Trajectory execution took {end_time - start_time:.2f}s.")
        
        current_pos_map = robot.get_current_position()
        is_success = True
        for name in robot.observation_joint_names:
            target = goal_action[f"{name}.pos"]
            actual = current_pos_map[name]
            if abs(target - actual) > POSITION_TOLERANCE_DEG:
                print(f"  FAIL: Joint '{name}' did not reach trajectory goal. (Target: {target:.2f}, Actual: {actual:.2f})")
                is_success = False

        if is_success:
            print("PASS: execute_trajectory moved robot to goal smoothly.")
        test_passed_summary["execute_trajectory"] = is_success

    except Exception as e:
        import traceback
        print(f"\nCRITICAL FAILURE during tests: {e}")
        traceback.print_exc()
        # 将所有未完成的测试标记为失败
        for test_name in ["Connection", "send_action", "execute_trajectory"]:
            if test_name not in test_passed_summary:
                test_passed_summary[test_name] = False

    finally:
        # -----------------------------------------------------------------
        # 5. 清理：回到初始位置并断开连接
        # -----------------------------------------------------------------
        if robot and robot.is_connected:
            print("\n--- [CLEANUP] Returning to initial position and disconnecting ---")
            try:
                # 使用平滑轨迹返回初始位置
                home_action = {f"{name}.pos": pos for name, pos in initial_positions.items()}
                robot.execute_trajectory(home_action, duration=TRAJECTORY_DURATION_S)
                print("INFO: Robot returned to initial position.")
            except Exception as e:
                print(f"WARNING: Could not return to initial position: {e}")
            finally:    
                robot.disconnect()
                print("INFO: Robot disconnected.")

    # -----------------------------------------------------------------
    # 6. 打印最终测试报告
    # -----------------------------------------------------------------
    print("\n" + "="*50)
    print("=      HARDWARE TEST SUITE REPORT      =")
    print("="*50)
    all_passed = True
    for test_name, result in test_passed_summary.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"- {test_name:<25}: {status}")
        if not result:
            all_passed = False
    
    print("-" * 50)
    if all_passed:
        print("🎉 All tests passed successfully!")
    else:
        print("🔥 One or more tests failed. Please review the logs.")
    print("="*50)


if __name__ == "__main__":
    print("WARNING: This script will command a real robot to move.")
    print("Ensure the robot has clear space and you are ready to use the emergency stop.")
    try:
        if input("Type 'yes' to continue: ").lower() == 'yes':
            run_test_suite()
        else:
            print("Test cancelled by user.")
    except (KeyboardInterrupt, EOFError):
        print("\nTest cancelled by user.")