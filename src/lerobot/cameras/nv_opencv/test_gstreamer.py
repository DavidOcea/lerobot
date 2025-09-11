from lerobot.cameras.nv_opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.nv_opencv.configuration_opencv import OpenCVCameraConfig, ColorMode

def create_orin_mjpeg_pipeline(
    device_path="/dev/video0",
    capture_width=640,
    capture_height=480,
    framerate=30,
    output_format="BGRx",
):
    """
    为输出 MJPEG 的 USB 摄像头生成一个硬件加速的 GStreamer 管道。
    """
    return (
        f"v4l2src device={device_path} ! "
        # 1. 明确请求 MJPEG 格式的压缩流
        f"image/jpeg, width={capture_width}, height={capture_height}, framerate={framerate}/1 ! "
        
        # 2. 使用 Orin 的硬件 JPEG 解码器 (NVJPEG)
        #    如果 gst-inspect-1.0 nvjpegdec 找不到，可以尝试用 jpegdec
        "nvjpegdec ! "
        
        # 3. 使用硬件视频转换器将解码后的帧转为 BGRx
        "nvvidconv ! "
        
        # 4. 定义输出给 OpenCV 的格式 (移除 memory:NVMM 以提高兼容性)
        f"video/x-raw, format={output_format} ! "
        
        # 5. 连接到 appsink
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
    color_mode=ColorMode.BGR, # GStreamer 管道输出 BGR，所以这里设为 BGR
    index_or_path=0,
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