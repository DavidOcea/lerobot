#!/usr/bin/env python3
"""
Camera Test Script - 相机测试脚本

用于确定相机索引与物理位置的对应关系
"""

import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_all_cameras():
    """测试所有相机并显示图像"""
    
    # 可能的相机索引
    possible_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    available_cameras = []
    
    print("="*60)
    print("相机检测工具")
    print("="*60)
    print("\n正在检测可用相机...\n")
    
    for idx in possible_indices:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                available_cameras.append(idx)
                print(f"  相机 {idx}: 可用 (分辨率: {frame.shape[1]}x{frame.shape[0]})")
            cap.release()
    
    if not available_cameras:
        print("未检测到任何相机!")
        return
    
    print(f"\n共检测到 {len(available_cameras)} 个相机: {available_cameras}")
    print("\n" + "="*60)
    print("按任意键切换相机，按 'q' 退出")
    print("="*60 + "\n")
    
    # 逐个显示每个相机
    camera_labels = {
        0: "相机0 - 请识别是哪个位置",
        2: "相机2 - 请识别是哪个位置", 
        4: "相机4 - 请识别是哪个位置",
        6: "相机6 - 请识别是哪个位置",
        8: "相机8 - 请识别是哪个位置"
    }
    
    for idx in available_cameras:
        cap = cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print(f"\n当前显示: 相机 {idx}")
        print("请观察图像并确定这是哪个位置的相机:")
        print("  - 头部相机 (head)")
        print("  - 左手腕相机1 (left_wrist_1)")
        print("  - 左手腕相机2 (left_wrist_2)")
        print("  - 右手腕相机1 (right_wrist_1)")
        print("  - 右手腕相机2 (right_wrist_2)")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 添加标签
            label = camera_labels.get(idx, f"相机{idx}")
            cv2.putText(frame, label, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Index: {idx}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, "Press any key for next camera, 'q' to quit", (10, 450),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow(f"Camera {idx}", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key != 255:  # 任意键
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("测试完成!")
    print("请记录每个索引对应的相机位置")
    print("="*60)


def record_camera_mapping():
    """记录相机映射关系"""
    
    print("\n" + "="*60)
    print("相机映射记录")
    print("="*60)
    
    mapping = {}
    
    positions = [
        ("头部相机", "head"),
        ("左手腕相机1", "left_wrist_1"),
        ("左手腕相机2", "left_wrist_2"),
        ("右手腕相机1", "right_wrist_1"),
        ("右手腕相机2", "right_wrist_2")
    ]
    
    for name, key in positions:
        idx = input(f"\n请输入 {name} 的索引: ").strip()
        if idx.isdigit():
            mapping[key] = int(idx)
    
    print("\n" + "-"*60)
    print("相机映射结果:")
    print("-"*60)
    for key, idx in mapping.items():
        print(f"  '{key}': {idx},")
    
    print("\n请将以上映射复制到你的配置文件中")
    
    return mapping


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="相机测试工具")
    parser.add_argument("--record", action="store_true", help="记录相机映射")
    
    args = parser.parse_args()
    
    if args.record:
        record_camera_mapping()
    else:
        test_all_cameras()
