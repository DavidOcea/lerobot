import cv2
import argparse

def main():
    # 1. 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description="使用 OpenCV 从指定的摄像头捕获并显示视频流。"
    )
    
    # 2. 添加 'index' 参数
    parser.add_argument(
        'index', 
        type=int, 
        nargs='?', 
        default=0,
        help="要使用的摄像头索引 (例如, 0, 1, 2, ...)。默认为 0。"
    )
    
    args = parser.parse_args()
    camera_index = args.index

    print(f"尝试打开摄像头索引: {camera_index}")

    # 3. 使用整数索引直接打开摄像头
    # OpenCV 将使用系统默认的后端 (如 V4L2)
    cap = cv2.VideoCapture(camera_index)

    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print(f"错误: 无法打开摄像头索引 {camera_index}。")
        print("请检查:")
        print(f"  - 摄像头是否正确连接。")
        print(f"  - 索引 {camera_index} 是否正确。")
        print("  - 摄像头是否被其他程序占用。")
        exit()

    try:
        window_title = f'USB Camera (Index: {camera_index})'
        print("摄像头已成功打开。按 'q' 键退出。")

        while True:
            # 读取一帧
            ret, frame = cap.read()

            # 如果 ret 为 False，表示读取失败 (例如，摄像头被拔出)
            if not ret:
                print("错误: 无法读取帧。视频流可能已结束。")
                break

            # 在此处可以添加你的图像处理代码
            # ...

            # 显示帧
            cv2.imshow(window_title, frame)

            # 按 'q' 键退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("检测到 'q' 键按下，正在退出...")
                break
    finally:
        # 确保资源被释放
        print("正在释放资源...")
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()