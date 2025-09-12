import cv2

# 定义 GStreamer 管道字符串，从摄像头捕获并转换为 BGR 格式，最后发送到 appsink
gst_str = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,width=640,height=480,framerate=30/1,format=YUYV ! "
    "nvvidconv ! video/x-raw,format=BGRx ! "  # 使用 nvvidconv 进行硬件加速的色彩空间转换（BGRx）
    "videoconvert ! video/x-raw,format=BGR ! "  # 转换为 OpenCV 常用的 BGR 格式
    "appsink drop=true"  # 将视频流发送到 appsink，drop=true 表示在缓冲区满时丢弃旧帧
)

cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)  # 以 GStreamer 模式打开视频捕获对象

if not cap.isOpened():
    print("错误: 无法打开摄像头.")
    exit()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("错误: 无法读取帧.")
            break

        # 在此处添加你的图像处理代码
        # 例如：灰度化、边缘检测、目标检测等
        # processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('USB Camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()