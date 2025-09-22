import unittest
import tempfile
import subprocess
import logging
from pathlib import Path
from PIL import Image, ImageDraw

# 假设您的 encode_video_frames 函数存放在名为 video_utils.py 的文件中
# 请根据实际情况修改下面的导入语句
from lerobot.datasets.video_utils import encode_video_frames_gst
from lerobot.datasets.lerobot_dataset import LeRobotDataset 
import shutil
import numpy as np
import os
import time
# 配置日志，方便在测试失败时查看 GStreamer 命令
logging.basicConfig(level=logging.INFO)

class TestVideoEncoding(unittest.TestCase):
    
    def setUp(self):
        """在每个测试方法运行前被调用"""
        # 1. 创建一个安全的临时目录
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_root = Path(self.temp_dir.name)
        
        # 2. 在临时目录中创建存放图片的子目录
        self.imgs_dir = self.test_root / "test_frames"
        self.imgs_dir.mkdir()
        
        # 3. 定义输出视频文件的路径
        self.video_path = self.test_root / "output.mp4"

    def tearDown(self):
        """在每个测试方法运行后被调用，用于清理"""
        self.temp_dir.cleanup()

    def _generate_test_frames(self, num_frames=150, width=640, height=480):
        """一个辅助函数，用于生成测试用的 PNG 图片"""
        for i in range(num_frames):
            # 创建一个黑色背景的图片
            img = Image.new('RGB', (width, height), color='black')
            draw = ImageDraw.Draw(img)
            
            # 在图片上绘制帧编号，方便肉眼验证
            text = f"Frame {i+1}"
            draw.text((10, 10), text, fill='white')
            
            # 保存为符合函数要求的格式
            filename = self.imgs_dir / f"frame_{i:06d}.png"
            img.save(filename)
        return num_frames, width, height

    def test_successful_encoding(self):
        """测试基本功能：成功将图片序列编码为视频"""
        # --- 准备 ---
        num_frames, width, height = self._generate_test_frames()
        fps = 5
        gop_size = 5

        # --- 执行 ---
        encode_video_frames_gst(
            imgs_dir=self.imgs_dir,
            video_path=self.video_path,
            fps=fps,
            g=gop_size,
            overwrite=True # 在测试中总是覆盖
        )

        # --- 验证 ---
        # 1. 基本验证：文件是否存在且不为空
        self.assertTrue(self.video_path.exists(), "视频文件未被创建")
        self.assertGreater(self.video_path.stat().st_size, 0, "视频文件大小为0")

        # 2. 高级验证：使用 ffprobe 检查视频元数据
        try:
            # 新代码，更健壮的 ffprobe 解析
            cmd = [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                # 在 show_entries 中指定我们要查询的字段
                "-show_entries", "stream=width,height,codec_name,r_frame_rate,avg_frame_rate",
                # 使用 key=value 的格式输出，这样顺序就不重要了
                "-of", "default=noprint_wrappers=1", 
                str(self.video_path)
            ]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # 将 key=value 格式的输出解析到一个字典中
            video_info = {}
            for line in result.stdout.strip().split('\n'):
                if "=" in line:
                    key, value = line.split('=', 1)
                    video_info[key] = value
            
            # 从字典中按键名获取值，而不是按顺序
            self.assertEqual(int(video_info.get("width")), width, "视频宽度不匹配")
            self.assertEqual(int(video_info.get("height")), height, "视频高度不匹配")
            # 注意：您的GStreamer命令使用h264编码器，所以这里我们检查h264
            self.assertEqual(video_info.get("codec_name"), "h264", "视频编码格式不是h264")
            self.assertIn(str(fps), video_info.get("r_frame_rate"), "视频帧率不匹配")
            self.assertIn(str(fps), video_info.get("avg_frame_rate"), "视频平均帧率不匹配")

            self._play_video_interactively(self.video_path, "单视频测试 (output.mp4)")
        except FileNotFoundError:
            self.skipTest("ffprobe 未安装，跳过视频元数据验证。")
        except subprocess.CalledProcessError as e:
            self.fail(f"ffprobe 执行失败: {e.stderr}")

    def test_no_input_images(self):
        """测试当输入目录为空时，是否按预期抛出异常"""
        # imgs_dir 是空的，因为我们没有调用 _generate_test_frames
        with self.assertRaises(FileNotFoundError, msg="当没有图片时应抛出 FileNotFoundError"):
            encode_video_frames_gst(
                imgs_dir=self.imgs_dir,
                video_path=self.video_path,
                fps=10
            )

    def test_overwrite_protection(self):
        """测试在 overwrite=False 的情况下，如果文件已存在是否会抛出异常"""
        # --- 准备 ---
        # 先创建一个空的占位文件
        self.video_path.touch()
        self._generate_test_frames(num_frames=1) # 至少要有一张图片

        # --- 执行与验证 ---
        with self.assertRaises(FileExistsError, msg="当文件存在且 overwrite=False 时应抛出 FileExistsError"):
            encode_video_frames_gst(
                imgs_dir=self.imgs_dir,
                video_path=self.video_path,
                fps=10,
                overwrite=False # 这是默认值，但显式写出更清晰
            )
        
        # 确保文件未被修改
        self.assertEqual(self.video_path.stat().st_size, 0, "受保护的文件不应被修改")

    def test_real_dataset_parallel_encoding(self):
        """
        测试 LeRobotDataset.save_episode() 是否能正确触发并行视频编码。
        这是一个集成测试，不使用任何模拟类。
        """
        # --- 准备 ---
        # 1. 定义数据集的属性
        repo_id = "test/test_dataset"
        # 使用self.test_root作为数据集的根目录，测试结束后会自动清理
        dataset_root = self.test_root / repo_id 
        
        video_keys = ["observation.image_main", "observation.image_wrist"]
        fps = 10
        features = {
            key: {"dtype": "video", "shape": (480, 640, 3)} for key in video_keys
        }
        features["state"] = {"dtype": "float32", "shape": (2,)}

        # 2. 使用 LeRobotDataset.create 创建一个真实的数据集实例
        try:
            dataset = LeRobotDataset.create(
                repo_id=repo_id,
                root=dataset_root,
                features=features,
                fps=fps,
            )
            # 为 encode_episode_videos_gst 设置并行工作线程数
            dataset.num_parallel_workers = 2
        except FileExistsError:
            # 如果之前的测试意外失败，目录可能已存在
            shutil.rmtree(dataset_root)
            dataset = LeRobotDataset.create(
                repo_id=repo_id,
                root=dataset_root,
                features=features,
                fps=fps,
            )
            dataset.num_parallel_workers = 2


        # 3. 模拟录制一个回合的数据
        num_frames_to_record = 30
        for i in range(num_frames_to_record):
            # 创建一个虚拟帧
            frame_data = {
                # 图像数据需要是 numpy 数组或 PIL Image
                "observation.image_main": Image.new("RGB", (640, 480), "red"),
                "observation.image_wrist": Image.new("RGB", (640, 480), "blue"),
                "state": np.array([i * 0.1, i * -0.1], dtype=np.float32),
            }
            dataset.add_frame(frame_data, task="test_task", timestamp=(i / fps))

        # --- 执行 ---
        # 保存回合，这将触发内部对 encode_episode_videos_gst 的调用
        dataset.save_episode()

        # --- 验证 ---
        # 1. 验证所有视频文件是否都已创建
        episode_index = 0
        for key in video_keys:
            # LeRobotDataset 会根据元数据自动生成路径
            expected_video_path = dataset.root / dataset.meta.get_video_file_path(episode_index, key)
            
            self.assertTrue(expected_video_path.exists(), f"视频文件 {expected_video_path} 未创建")
            self.assertGreater(
                expected_video_path.stat().st_size, 0, f"视频文件 {expected_video_path} 大小为0"
            )
        
        # 2. 验证 Parquet 数据文件也已创建
        expected_data_path = dataset.root / dataset.meta.get_data_file_path(episode_index)
        self.assertTrue(expected_data_path.exists(), "Parquet 数据文件未创建")

        # 3. 验证临时图像目录已被删除
        temp_image_dir = dataset.root / "images"
        self.assertFalse(temp_image_dir.exists(), "临时的 images 目录在编码后应被删除")
        
        # 4. （可选）用 ffprobe 快速检查一个视频的元数据
        main_video_path = dataset.root / dataset.meta.get_video_file_path(episode_index, "observation.image_main")
        try:
            cmd = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "stream=width,height,codec_name,nb_frames,r_frame_rate",
                "-of", "default=noprint_wrappers=1", 
                str(main_video_path)
            ]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            video_info = dict(line.split('=') for line in result.stdout.strip().split('\n'))
            
            self.assertEqual(int(video_info['width']), 640)
            self.assertEqual(int(video_info['height']), 480)
            self.assertEqual(video_info['codec_name'], 'h264')
            self.assertEqual(int(video_info['nb_frames']), num_frames_to_record, "视频帧数不匹配")
            self.assertIn(str(fps), video_info['r_frame_rate'])

            for key in video_keys:
                video_path = dataset.root / dataset.meta.get_video_file_path(episode_index, key)
                self._play_video_interactively(video_path, f"并行测试视频 ({key})")
        except FileNotFoundError:
            self.skipTest("ffprobe 未安装，跳过视频元数据验证。")
        except (subprocess.CalledProcessError, KeyError, ValueError) as e:
            self.fail(f"对 observation.image_main 的 ffprobe 验证失败: {e}")

    def _play_video_interactively(self, video_path: Path, video_name: str):
        """
        如果设置了 INTERACTIVE_TEST=1 环境变量，则自动播放视频并等待其关闭。
        """
        # 检查环境变量 INTERACTIVE_TEST 是否被设置为 '1' 或 'true'
        is_interactive = os.environ.get("INTERACTIVE_TEST", "0").lower() in ["1", "true", "yes"]

        if not is_interactive:
            return # 如果不是交互模式，则直接返回

        try:
            # 检查 ffplay 是否存在
            subprocess.run(["ffplay", "-version"], capture_output=True, check=True, text=True)
            
            logging.info(f"\n--- 交互式视频预览 ---")
            logging.info(f"正在播放 '{video_name}'。请检查视频内容。关闭播放器后测试将继续...")
            
            # 构建并执行 ffplay 命令
            # -autoexit: 播放结束后自动退出
            # -x, -y: 设置窗口大小
            # 我们不捕获输出，而是让它直接显示在终端上
            cmd = [
                "ffplay",
                "-autoexit",
                "-window_title", f"Preview: {video_name}",
                "-x", "640",
                "-y", "480",
                str(video_path)
            ]
            
            # 使用 subprocess.run() 并设置 check=True。
            # 这会阻塞代码，直到 ffplay 进程结束。
            # 如果用户强制关闭（非正常退出），可能会抛出异常。
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            logging.info(f"播放结束，测试继续...")

        except FileNotFoundError:
            logging.warning("\n'ffplay' 未安装，跳过交互式视频预览。")
        except subprocess.CalledProcessError as e:
            # 如果 ffplay 异常退出，打印警告但不要让测试失败
            logging.warning(f"\nffplay 播放时发生错误 (退出码: {e.returncode})，测试继续。")

# --- 将这个新的性能测试方法添加到 TestVideoEncoding 类中 ---

    @unittest.skipUnless(
        os.environ.get("RUN_PERFORMANCE_TESTS", "0").lower() in ["1", "true", "yes"],
        "Skipping performance test. Set RUN_PERFORMANCE_TESTS=1 to enable."
    )
    def test_parallel_encoding_performance(self):
        """
        对比单线程和双线程并行编码的耗时。
        """
        logging.info("\n\n" + "="*50)
        logging.info("  开始并行编码性能基准测试  ")
        logging.info("="*50)
        
        # --- 准备阶段 ---
        # 准备更大量的数据以获得有意义的计时结果
        repo_id_base = "performance_test/dataset"
        # 编码4个视频流，这样双核编码器可以充分利用
        video_keys = ["cam_front", "cam_wrist", "cam_left", "cam_right"]
        fps = 30
        num_frames_per_video = 200 # 增加帧数
        
        features = {
            key: {"dtype": "video", "shape": (240, 320, 3)} for key in video_keys
        }

        # 准备一个包含多个并行度设置的列表
        parallelism_levels = [1, 2]
        timings = {}

        for num_workers in parallelism_levels:
            logging.info(f"\n--- 正在测试并行度: {num_workers} ---")
            
            # 1. 为每个测试级别创建独立的数据集实例和数据
            repo_id = f"{repo_id_base}_w{num_workers}"
            dataset_root = self.test_root / repo_id
            
            try:
                dataset = LeRobotDataset.create(
                    repo_id=repo_id,
                    root=dataset_root,
                    features=features,
                    fps=fps,
                )
                dataset.num_parallel_workers = num_workers
            except FileExistsError:
                shutil.rmtree(dataset_root)
                dataset = LeRobotDataset.create(
                    repo_id=repo_id,
                    root=dataset_root,
                    features=features,
                    fps=fps,
                )
                dataset.num_parallel_workers = num_workers

            # 2. 生成图像帧 (这个过程不计时)
            logging.info("正在生成测试图像...")
            for i in range(num_frames_per_video):
                frame_data = {
                    key: Image.new("RGB", (320, 240), "gray") for key in video_keys
                }
                dataset.add_frame(frame_data, task="perf_task")
            
            # --- 执行与计时 ---
            logging.info("开始编码...")
            start_time = time.monotonic()
            
            # 调用 save_episode，这是我们要测量的核心操作
            dataset.save_episode()
            
            end_time = time.monotonic()
            
            # 记录耗时
            elapsed_time = end_time - start_time
            timings[num_workers] = elapsed_time
            logging.info(f"编码完成，耗时: {elapsed_time:.2f} 秒")

            # --- 快速验证 (确保编码成功) ---
            episode_index = 0
            for key in video_keys:
                video_path = dataset.root / dataset.meta.get_video_file_path(episode_index, key)
                self.assertTrue(video_path.exists())
                self.assertGreater(video_path.stat().st_size, 0)

        # --- 总结报告 ---
        logging.info("\n\n" + "="*50)
        logging.info("  性能基准测试总结  ")
        logging.info("="*50)
        logging.info(f"测试配置: {len(video_keys)} 个视频流, 每个视频 {num_frames_per_video} 帧, {fps} FPS")
        
        time_w1 = timings.get(1)
        time_w2 = timings.get(2)

        if time_w1 is not None:
            logging.info(f"并行度 = 1 (单线程): {time_w1:.2f} 秒")
        if time_w2 is not None:
            logging.info(f"并行度 = 2 (双线程): {time_w2:.2f} 秒")
        
        if time_w1 and time_w2:
            speedup = time_w1 / time_w2
            logging.info(f"\n性能提升 (Speedup): {speedup:.2f}x")
            # 断言：我们期望至少有一点性能提升
            self.assertGreater(speedup, 1.1, "并行度=2时应比并行度=1有显著的速度提升")

if __name__ == '__main__':
    # 替换 'your_module_name' 为您存放函数的Python文件名（不含.py后缀）
    # 例如，如果文件是 video_utils.py，就写 from video_utils import encode_video_frames
    try:
        from lerobot.datasets.video_utils import encode_video_frames_gst
    except ImportError:
        print("="*80)
        print("错误：请将 'your_module_name' 替换为包含 encode_video_frames 函数的文件名。")
        print("例如：如果您的文件是 'my_script.py'，请修改第12行和第165行的导入语句。")
        print("="*80)
        exit(1)
        
    unittest.main(argv=['first-arg-is-ignored'], exit=False)