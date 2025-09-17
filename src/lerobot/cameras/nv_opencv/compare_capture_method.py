import cv2
import time
import argparse
import psutil
import os
import numpy as np

# --- GStreamer Pipeline Definition (from your question) ---
def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0,
    device_id=0,
):
    """
    Constructs the GStreamer pipeline string for hardware-accelerated capture.
    """
    return (
        "v4l2src device=/dev/video%d ! "
        "video/x-raw, width=%d, height=%d, framerate=%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=%d, height=%d, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! appsink drop=true"
        % (
            device_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def run_test(args):
    """
    Main function to run the performance and resource test.
    """
    cap = None
    
    # --- Step 1: Initialize Video Capture ---
    print(f"--- Starting Test for Method: {args.method.upper()} ---")
    print(f"Resolution: {args.width}x{args.height} @ {args.framerate} FPS")
    
    if args.method == 'opencv':
        # Use standard OpenCV backend (V4L2)
        print("Initializing standard cv2.VideoCapture...")
        cap = cv2.VideoCapture(args.device_id, cv2.CAP_V4L2)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
            cap.set(cv2.CAP_PROP_FPS, args.framerate)
    
    elif args.method == 'gstreamer':
        # Use GStreamer backend
        pipeline = gstreamer_pipeline(
            capture_width=args.width,
            capture_height=args.height,
            display_width=args.width,
            display_height=args.height,
            framerate=args.framerate,
            device_id=args.device_id,
        )
        print("Using GStreamer Pipeline:")
        print(f"  {pipeline}")
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap or not cap.isOpened():
        print("!!! Error: Failed to open camera.")
        print("!!! Please check camera device ID and supported formats/resolutions.")
        return

    # --- Step 2: Run Measurement Loop ---
    print(f"\nRunning test for {args.frames} frames...")
    
    # Get current process for resource monitoring
    process = psutil.Process(os.getpid())
    
    frame_count = 0
    start_time = time.time()
    
    cpu_usage_list = []
    mem_usage_list = []

    # A short warm-up period
    for _ in range(30):
        cap.read()

    while frame_count < args.frames:
        ret, frame = cap.read()
        if not ret:
            print("\nWarning: Failed to grab frame. Test ending early.")
            break

        # Collect resource usage stats
        # Note: cpu_percent() is blocking for a tiny interval
        # Normalizing by CPU count gives a 0-100% total system load value
        cpu_percent = process.cpu_percent() / psutil.cpu_count()
        # RSS: Resident Set Size - the non-swapped physical memory a process has used.
        mem_mb = process.memory_info().rss / (1024 * 1024)
        
        cpu_usage_list.append(cpu_percent)
        mem_usage_list.append(mem_mb)
        
        frame_count += 1
        
        # Optional: Display the window to see it's working
        if not args.no_display:
            cv2.imshow(f"Test: {args.method}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    end_time = time.time()
    
    # --- Step 3: Calculate and Report Results ---
    total_time = end_time - start_time
    
    # Avoid division by zero if the test was too short or failed
    if total_time > 0 and len(cpu_usage_list) > 0:
        avg_fps = frame_count / total_time
        avg_cpu = np.mean(cpu_usage_list)
        avg_mem = np.mean(mem_usage_list)
        max_mem = np.max(mem_usage_list)

        print("\n--- Test Results ---")
        print(f"Method:              {args.method.upper()}")
        print(f"Frames processed:    {frame_count}")
        print(f"Total time:          {total_time:.2f} seconds")
        print("---------------------------------")
        print(f"Average FPS:         {avg_fps:.2f}")
        print(f"Average CPU Usage:   {avg_cpu:.2f}% (normalized total)")
        print(f"Average Memory:      {avg_mem:.2f} MB")
        print(f"Max Memory:          {max_mem:.2f} MB")
        print("---------------------------------")
    else:
        print("Test failed or was too short to gather meaningful data.")

    # --- Step 4: Cleanup ---
    print("Test finished. Releasing resources.")
    cap.release()
    cv2.destroyAllWindows()

# python compare_capture_methods.py --method gstreamer --width 640 --height 480 --framerate 30 --device-id 0
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare OpenCV video capture methods.")
    parser.add_argument('--method', type=str, required=True, choices=['opencv', 'gstreamer'],
                        help="The capture method to test.")
    parser.add_argument('--width', type=int, default=1280, help="Capture width.")
    parser.add_argument('--height', type=int, default=720, help="Capture height.")
    parser.add_argument('--framerate', type=int, default=30, help="Capture framerate.")
    parser.add_argument('--device-id', type=int, default=0, help="Camera device ID (e.g., 0 for /dev/video0).")
    parser.add_argument('--frames', type=int, default=500, help="Number of frames to capture for the test.")
    parser.add_argument('--no-display', action='store_true', help="Do not display the video window during the test.")
    
    args = parser.parse_args()
    
    run_test(args)