from lerobot.cameras.nv_opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.nv_opencv.configuration_opencv import OpenCVCameraConfig, ColorMode

def create_orin_hw_accel_pipeline(
    device_path="/dev/video0",
    capture_width=640,
    capture_height=480,
    framerate=30,
    output_format="BGRx", # OpenCV 在 appsrc 中通常期望 BGR 格式
):
    """
    一个辅助函数，用于为 Jetson Orin 生成硬件加速的 GStreamer 捕获管道。
    """
    return (
        f"v4l2src device={device_path} ! "
        # 1. 定义从摄像头捕获的原始格式、分辨率和帧率
        f"video/x-raw, width={capture_width}, height={capture_height}, framerate={framerate}/1 ! "
        
        # 2. 使用 Orin 的硬件视频转换器 (VIC - Video Image Compositor)
        #    它高效地处理颜色空间转换和缩放，并把数据保持在NVIDIA内存中。
        "nvvidconv ! "
        
        # 3. 定义输出给 OpenCV 的格式
        #    memory:NVMM 尽可能地避免了昂贵的 CPU 内存拷贝
        f"video/x-raw(memory:NVMM), format={output_format} ! "
        
        # 4. 将 GStreamer 的输出连接到 OpenCV 的 "app sink"
        "appsink drop=true"
    )

# --- 主程序 ---

# 1. 创建硬件加速的 GStreamer 管道
orin_pipeline = create_orin_hw_accel_pipeline(
    device_path="/dev/video0", # 确认这是你的摄像头设备
    capture_width=640,
    capture_height=480,
    framerate=30
)
print("Using GStreamer pipeline:\n", orin_pipeline)

# 2. 创建配置对象，这次传入 gstreamer_pipeline
#    注意：width, height, fps 仍然可以传入，用于验证管道是否按预期工作
config_hw = OpenCVCameraConfig(
    gstreamer_pipeline=orin_pipeline,
    width=640,
    height=480,
    fps=30,
    color_mode=ColorMode.BGR # GStreamer 管道输出 BGR，所以这里设为 BGR
)

# 3. 创建并使用相机实例
camera = OpenCVCamera(config_hw)
try:
    # connect() 方法现在会自动检测到 gstreamer_pipeline 并使用它
    camera.connect()
    
    # 后续的 read() 调用方式完全不变
    for _ in range(100):
        frame = camera.read()
        # 在这里用 `top` 或 `jtop` 命令监控，你会发现 CPU 占用率远低于非硬件加速模式
        cv2.imshow("Hardware Accelerated Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    camera.disconnect()
    cv2.destroyAllWindows()