"""
Calibration Tool - 标定工具

用于标定：
1. 像素到毫米的转换比例
2. 关节灵敏度
3. 保存/加载目标模板
"""

import cv2
import numpy as np
import json
import time
from pathlib import Path


class CalibrationTool:
    """标定工具"""
    
    def __init__(self, robot, camera, arm: str = "right"):
        """
        Args:
            robot: SupreRobotFollower 实例
            camera: 手腕相机实例
            arm: 手臂标识
        """
        self.robot = robot
        self.camera = camera
        self.arm = arm
        self.results = {}
        
    def run_full_calibration(self):
        """运行完整标定流程"""
        print("\n" + "="*60)
        print(f"开始完整标定流程 - {self.arm}臂")
        print("="*60)
        
        # 1. 采集目标模板
        print("\n[步骤1] 采集目标模板")
        self.calibrate_target_template()
        
        # 2. 标定像素-毫米比例
        print("\n[步骤2] 标定像素-毫米比例")
        self.calibrate_pixel_to_mm()
        
        # 3. 标定关节灵敏度
        print("\n[步骤3] 标定关节灵敏度")
        self.calibrate_joint_sensitivity()
        
        # 4. 保存所有结果
        self.save_all_results()
        
        print("\n" + "="*60)
        print("标定完成!")
        print("="*60)
    
    def calibrate_target_template(self):
        """采集目标模板"""
        print("\n--- 目标模板采集 ---")
        print("请将机器人移动到目标放置位置（卡槽上方）")
        input("到位后按 Enter 采集模板...")
        
        # 采集图像
        image = self.camera.read()
        
        # 显示图像让用户确认
        cv2.imshow("Target Template", image)
        print("请确认图像中的卡槽清晰可见")
        print("按 'y' 确认，按其他键重新采集")
        
        key = cv2.waitKey(0)
        cv2.destroyWindow("Target Template")
        
        if key == ord('y'):
            # 保存模板
            template_path = Path(__file__).parent / f"template_{self.arm}.png"
            cv2.imwrite(str(template_path), image)
            self.results['template_path'] = str(template_path)
            print(f"模板已保存: {template_path}")
        else:
            print("重新采集...")
            self.calibrate_target_template()
    
    def calibrate_pixel_to_mm(self, move_distance_mm: float = 5.0):
        """
        标定像素到毫米的转换比例
        
        方法：
        1. 采集当前位置图像
        2. 手动移动机器人指定距离
        3. 采集新位置图像
        4. 计算像素偏移
        """
        print("\n--- 像素-毫米比例标定 ---")
        print(f"移动距离: {move_distance_mm}mm")
        
        # 采集第一张图像
        print("\n1. 采集初始位置图像...")
        img1 = self.camera.read()
        
        # 显示图像
        cv2.imshow("Position 1", img1)
        cv2.waitKey(500)
        cv2.destroyWindow("Position 1")
        
        # 提示移动
        print(f"\n2. 请手动将机器人沿X方向移动 {move_distance_mm}mm")
        print("   (建议使用示教器进行精确移动)")
        input("   移动完成后按 Enter...")
        
        # 采集第二张图像
        print("\n3. 采集移动后位置图像...")
        img2 = self.camera.read()
        
        # 显示图像
        cv2.imshow("Position 2", img2)
        cv2.waitKey(500)
        cv2.destroyWindow("Position 2")
        
        # 计算像素偏移
        pixel_offset = self._compute_pixel_offset(img1, img2)
        
        # 计算比例
        ratio = move_distance_mm / pixel_offset
        
        print(f"\n标定结果:")
        print(f"  像素偏移: {pixel_offset:.1f} pixels")
        print(f"  实际移动: {move_distance_mm} mm")
        print(f"  转换比例: {ratio:.4f} mm/pixel")
        print(f"           = {1/ratio:.1f} pixel/mm")
        
        self.results['pixel_to_mm_ratio'] = ratio
        
        # 保存图像用于验证
        calib_dir = Path(__file__).parent / "calibration_images"
        calib_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(calib_dir / f"calib_{self.arm}_pos1.jpg"), img1)
        cv2.imwrite(str(calib_dir / f"calib_{self.arm}_pos2.jpg"), img2)
        
        return ratio
    
    def calibrate_joint_sensitivity(self):
        """
        标定关节灵敏度
        
        方法：微调单个关节，观察末端位移
        """
        print("\n--- 关节灵敏度标定 ---")
        print("此步骤需要测量关节角度变化与末端位移的关系")
        print("建议使用外部测量工具（如尺子、激光测距仪）")
        
        sensitivities = {}
        
        # 只标定主要影响XY位置的关节
        key_joints = {
            'joint_1': '基座旋转（影响X）',
            'joint_2': '肩部俯仰（影响Y）'
        }
        
        for joint_name, description in key_joints.items():
            print(f"\n标定 {joint_name} ({description}):")
            
            # 采集初始位置
            print("  1. 请确保末端在合适位置")
            input("     准备好后按 Enter...")
            
            obs1 = self.robot.get_observation()
            joints1 = np.array(obs1.get('observation.state', []))
            img1 = self.camera.read()
            
            # 微调关节
            print(f"  2. 请微调 {joint_name} 约1度")
            print("     (观察末端位移)")
            input("     调整完成后按 Enter...")
            
            obs2 = self.robot.get_observation()
            joints2 = np.array(obs2.get('observation.state', []))
            img2 = self.camera.read()
            
            # 计算变化量
            if self.arm == "right":
                joint_idx = 7 + int(joint_name.split('_')[1]) - 1
            else:
                joint_idx = int(joint_name.split('_')[1]) - 1
            
            angle_change = abs(joints2[joint_idx] - joints1[joint_idx])
            pixel_offset = self._compute_pixel_offset(img1, img2)
            
            # 假设已知像素-毫米比例
            ratio = self.results.get('pixel_to_mm_ratio', 0.5)
            mm_offset = pixel_offset * ratio
            
            sensitivity = angle_change / mm_offset if mm_offset > 0 else 0.1
            sensitivities[joint_name] = sensitivity
            
            print(f"  结果:")
            print(f"    角度变化: {angle_change:.2f} 度")
            print(f"    末端位移: {mm_offset:.2f} mm")
            print(f"    灵敏度: {sensitivity:.4f} 度/mm")
        
        # 对于其他关节，使用估计值
        default_sensitivities = {
            'joint_3': 0.30,
            'joint_4': 0.50,
            'joint_5': 0.80,
            'joint_6': 1.00
        }
        
        for joint_name, default_val in default_sensitivities.items():
            if joint_name not in sensitivities:
                sensitivities[joint_name] = default_val
        
        self.results['joint_sensitivity'] = sensitivities
        
        print(f"\n关节灵敏度汇总:")
        for name, val in sensitivities.items():
            print(f"  {name}: {val:.4f} 度/mm")
        
        return sensitivities
    
    def _compute_pixel_offset(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """计算两张图像间的像素偏移"""
        # 转灰度
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # 光流法
        corners = cv2.goodFeaturesToTrack(g1, maxCorners=100, 
                                          qualityLevel=0.01, minDistance=10)
        
        if corners is not None and len(corners) >= 10:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(g1, g2, corners, None)
            
            if p1 is not None:
                good_old = corners[st == 1]
                good_new = p1[st == 1]
                
                if len(good_old) >= 5:
                    offsets = good_new - good_old
                    mean_offset = np.mean(offsets, axis=0)
                    return np.abs(mean_offset[0])
        
        # 备用方法：模板匹配
        h, w = g1.shape
        template = g1[h//3:2*h//3, w//3:2*w//3]
        result = cv2.matchTemplate(g2, template, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(result)
        
        th, tw = template.shape
        center1 = (w//2, h//2)
        center2 = (max_loc[0] + tw//2, max_loc[1] + th//2)
        
        return np.abs(center2[0] - center1[0])
    
    def save_all_results(self):
        """保存所有标定结果"""
        save_path = Path(__file__).parent / f"calibration_{self.arm}.json"
        
        self.results['arm'] = self.arm
        self.results['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        
        with open(save_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n标定结果已保存: {save_path}")
    
    def load_results(self) -> dict:
        """加载标定结果"""
        load_path = Path(__file__).parent / f"calibration_{self.arm}.json"
        
        if load_path.exists():
            with open(load_path, 'r') as f:
                self.results = json.load(f)
            print(f"已加载标定结果: {load_path}")
            return self.results
        
        print(f"标定文件不存在: {load_path}")
        return {}


def quick_calibrate(robot, camera, arm: str = "right"):
    """快速标定（只标定像素-毫米比例）"""
    tool = CalibrationTool(robot, camera, arm)
    tool.calibrate_target_template()
    tool.calibrate_pixel_to_mm()
    tool.save_all_results()
    return tool.results
