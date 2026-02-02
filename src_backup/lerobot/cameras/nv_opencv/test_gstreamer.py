from lerobot.cameras.nv_opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.nv_opencv.configuration_opencv import OpenCVCameraConfig, ColorMode
import cv2
def create_orin_mjpeg_pipeline(
    device_path="/dev/video2",
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
        f"image/jpeg, width={capture_width}, height={capture_height}, framerate={framerate}/1 ! "
        
        # <<< 核心修改：在解码前加入 jpegparse >>>
        "jpegparse ! "
        
        "nvjpegdec ! "
        "nvvidconv ! "
        f"video/x-raw, format=BGRx ! "
        "appsink drop=true"
    )

def create_orin_hybrid_pipeline(
    device_path="/dev/video0",
    capture_width=640,
    capture_height=480,
    framerate=30,
    output_format="BGR", # OpenCV appsink 更喜欢 BGR
):
    """
    最终的、推荐的硬件加速管道。
    它使用 nvjpegdec 进行硬件解码，然后用 nvvidconv 将结果安全地
    转换回 CPU 内存，以实现最佳的性能和兼容性。
    """
    return (
        f"v4l2src device={device_path} ! "
        f"image/jpeg, width={capture_width}, height={capture_height}, framerate={framerate}/1 ! "
        "jpegparse ! "
        
        # 1. 使用硬件解码器，它会输出到 NVMM 内存
        "nvjpegdec ! "
        
        # 2. 使用硬件转换器，它会从 NVMM 内存读取，并输出到 CPU 内存
        "nvvidconv ! "
        
        # 3. 指定输出格式为 BGR，并确保它在 CPU 内存中（默认行为）
        f"video/x-raw, format={output_format} ! "
        
        # 4. 连接到 appsink
        "appsink drop=true"
    )
# --- 主程序 ---
def create_software_jpeg_pipeline():
    """Uses a software JPEG decoder to isolate the nvjpegdec plugin."""
    return (
        "v4l2src device=/dev/video0 ! "
        "image/jpeg, width=640, height=480, framerate=30/1 ! "
        "jpegparse ! "
        
        # <<< USE THE SOFTWARE DECODER >>>
        "jpegdec ! "
        
        # nvvidconv is still useful for color conversion, even if not hardware accelerated
        "videoconvert ! " # Use generic videoconvert, not nvvidconv
        "video/x-raw, format=BGR ! "
        "appsink drop=true"
    )

def create_kitchen_sink_pipeline(
    device_path="/dev/video0",
    capture_width=640,
    capture_height=480,
    framerate=30,
    output_format="BGR",
):
    """
    The final attempt pipeline, combining all known robustness tricks.
    """
    return (
        # Use dmabuf mode for better memory sharing potential
        f"v4l2src device={device_path} io-mode=2 ! " 
        f"image/jpeg, width={capture_width}, height={capture_height}, framerate={framerate}/1 ! "
        # A queue to buffer the raw camera output
        "queue ! "
        "jpegparse ! "
        "nvjpegdec ! "
        # A second queue after hardware decoding
        "queue ! "
        "nvvidconv ! "
        f"video/x-raw, format={output_format} ! "
        "appsink drop=true"
    )
# 1. 创建硬件加速的 GStreamer 管道
#orin_pipeline = create_kitchen_sink_pipeline(
#    device_path="/dev/video0", # 确认这是你的摄像头设备
#    capture_width=640,
#    capture_height=480,
#    framerate=30,
#    output_format="RGB",
#)
#orin_pipeline = create_software_jpeg_pipeline()
orin_pipeline =(
     "v4l2src device=/dev/video0 ! "
     "image/jpeg, width=640, height=480, framerate=30/1 ! "
     "jpegparse ! nvjpegdec ! nvvidconv ! "
     "video/x-raw, format=BGR ! appsink drop=true"
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