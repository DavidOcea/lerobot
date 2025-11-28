import os
import json
import shutil
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
import logging
from tqdm import tqdm
import pandas as pd

import shutil
import posixpath
import copy

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetMerger:
    def __init__(self, output_dir: str, dataset_paths: List[str]):
        """
        初始化数据集合并器
        
        Args:
            output_dir: 合并后数据集的输出目录
            dataset_paths: 待合并的数据集路径列表
        """
        self.output_dir = Path(output_dir)
        self.dataset_paths = [Path(path) for path in dataset_paths]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查所有数据集路径是否存在
        for path in self.dataset_paths:
            if not path.exists():
                raise FileNotFoundError(f"数据集路径不存在: {path}")
        
        # 确保输出目录为空或不存在
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            logger.warning(f"输出目录 {self.output_dir} 不为空，可能会覆盖现有文件")
    
    def merge_metadata(self) -> None:
        """合并所有数据集的元数据"""
        logger.info("合并元数据...")
        
        # 创建输出元数据目录
        meta_dir = self.output_dir / "meta"
        meta_dir.mkdir(exist_ok=True)
        
        # 合并info.json
        info_out = None
        for i, path in enumerate(self.dataset_paths):
            info_path = path / "meta" / "info.json"
            if info_path.exists():
                with open(info_path, 'r') as f:
                    info = json.load(f)
                    if info_out is None:
                        info_out = info
                    else:
                        info_out['total_episodes'] += info['total_episodes']
                        info_out['total_frames'] += info['total_frames']
                        info_out['total_videos'] += info['total_videos']
            else:
                logger.warning(f"数据集 {path} 缺少info.json文件")
        
        info_out['splits']['train'] = str(info_out['splits']['train'].split(":")[0]) + ":" + str(info_out['total_episodes'])
        
        # 写入合并后的info.json
        with open(meta_dir / "info.json", 'w') as f:
            json.dump(info_out, f, indent=2)
        
        # cp task.json
        tast_path = path / "meta" / "tasks.jsonl"
        tast_out_path = meta_dir / "tasks.jsonl"
        logger.info("cp " + str(tast_path) + " to " + str(tast_out_path))
        shutil.copy2(str(tast_path), str(tast_out_path))
        
        # 合并stats.json（统计信息）
        self._merge_stats()
        
        # 合并episode_data_index
        self._merge_episode_data_index()
    
    def _merge_stats(self) -> None:
        """合并所有数据集的统计信息"""
        logger.info("合并统计信息...")
        
        all_stats = []
        idxs = 0
        stats_idx = 0
        
        for i, path in enumerate(self.dataset_paths):
            stats_path = path / "meta" / "episodes_stats.jsonl"
            if stats_path.exists():
                with open(stats_path, 'r') as f:
                    
                    for line in f:
                        # import pdb; pdb.set_trace()
                        stats = json.loads(line)
                        stats['episode_index'] = idxs
                        stats['stats']['episode_index']['min'] = [idxs]
                        stats['stats']['episode_index']['max'] = [idxs]
                        stats['stats']['episode_index']['mean'] = [idxs * 1.0]

                        stats['stats']['index']['min'] = [stats_idx]
                        stats['stats']['index']['max'] = [stats['stats']['index']['count'][0] + stats_idx - 1]
                        stats['stats']['index']['mean'] = [(stats['stats']['index']['min'][0] + stats['stats']['index']['max'][0]) / 2]

                        stats_idx += stats['stats']['index']['count'][0]


                        idxs += 1
                        all_stats.append(stats)
                        
            else:
                logger.warning(f"数据集 {path} 缺少episodes_stats.jsonl文件")
        
        # 写入合并后的stats.json
        with open(self.output_dir / "meta" / "episodes_stats.jsonl", 'w') as f:
            for stat in all_stats:
                json.dump(stat, f)
                f.write('\n')
            # json.dump(all_stats, f, indent=2)
    
    
    def _merge_episode_data_index(self) -> None:
        """合并所有数据集的episode_data_index"""
        logger.info("合并episode_data_index...")
        
        all_episode = []
        current_idx = 0
        
        for i, path in enumerate(self.dataset_paths):
            index_path = path / "meta" / "episodes.jsonl"
            if index_path.exists():
                with open(index_path, 'r') as f:
                    for line in f:
                        stats = json.loads(line)
                        stats['episode_index'] = current_idx
                        current_idx += 1
                        all_episode.append(stats)
            else:
                logger.warning(f"数据集 {path} 缺少episodes.jsonl文件")
        
        # 合并数组
        # 写入合并后的episode_data_index.json
        with open(self.output_dir / "meta" / "episodes.jsonl", 'w') as f:
            for episde in all_episode:
                json.dump(episde, f)
                f.write('\n')
            # json.dump(all_episode, f, indent=2)
    
    def merge_data_files(self) -> None:
        """合并数据文件（如parquet文件和视频文件）"""
        logger.info("合并数据文件...")
        
        # 创建输出数据目录
        self.data_dir = self.output_dir / "data"
        self.data_dir.mkdir(exist_ok=True)
        self.data_dir = self.data_dir/"chunk-000"
        self.data_dir.mkdir(exist_ok=True)
        
        # 合并parquet文件
        self._merge_parquet_files()
        
        # 合并视频文件
        self._merge_video_files()
    
    def _merge_parquet_files(self) -> None:
        """合并parquet格式的数据文件"""
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            logger.error("请安装pyarrow库: pip install pyarrow")
            return
        # import pdb; pdb.set_trace()
        # 查找所有数据集的parquet文件
        all_parquet_files = []
        for path in self.dataset_paths:
            # 排序 保证顺序
            parquet_files = sorted(list(path.glob("data/chunk-000/*.parquet")) , key=lambda x: int(os.path.splitext(x)[0].split("_")[-1]) )
            # file_names = sorted(os.listdir(str(path)+"/data/chunk-000"), key=lambda x: int(os.path.splitext(x)[0].split("_")[1]))
            if parquet_files:
                all_parquet_files.append(parquet_files)
            else:
                logger.warning(f"数据集 {path} 中未找到parquet文件")
        
        if not all_parquet_files:
            logger.error("未找到任何parquet文件")
            return
        
        file_idx = 0
        start_value = 0
        self.file_idx_len = len(os.path.splitext(all_parquet_files[0][0])[0].split('_')[-1])
        # 遍历所有文件夹内的所有文件
        for file_path_list in all_parquet_files:
            for file_path in file_path_list:
                # prif_path = os.path.splitext(file_path)[0]
                last_path = os.path.splitext(file_path)[1]
                dest_file_path = str(self.data_dir) + "/episode_" + str(file_idx).zfill(self.file_idx_len) + last_path
                
                # print("cp " + str(file_path) + " to " + dest_file_path)
                # log 现实是cp过去的，实际上是修改另存过去的 
                # logger.info("cp " + str(file_path) + " to " + dest_file_path)
                # shutil.copy2(file_path, dest_file_path)

                # 修改parquet index内容
                # import pdb; pdb.set_trace()
                # 读取Parquet文件
                df = pd.read_parquet(file_path)
                df['episode_index'] = file_idx
                length = len(df['index'])
                incremental_values = np.arange(start_value, start_value + length)
                df['index'] = incremental_values
                # 保存回 
                df.to_parquet(dest_file_path, engine='pyarrow')


                file_idx += 1
                start_value += length
                logger.info("create " + str(file_path) + " to " + dest_file_path)

       
    def _merge_video_files(self) -> None:
        """合并视频文件（可选，根据数据集配置）"""
        # 创建视频目录
        video_dir = self.output_dir / "videos"
        video_dir.mkdir(exist_ok=True)
        # import pdb; pdb.set_trace()
        # 复制所有视频文件并创建映射
        file_idx = 0
        temp_video_list = []
        for dataset_idx, path in enumerate(self.dataset_paths):
            video_dir = video_dir/"chunk-000"
            video_dir.mkdir(exist_ok=True)
            video_path = str(path) + "/videos/chunk-000"
            se_idx = 0
            for vpath_dir in os.listdir(video_path):
                video_read_dir = video_path + "/" + vpath_dir
                out_video_dir = self.output_dir / "videos"/"chunk-000"/vpath_dir
                out_video_dir.mkdir(exist_ok=True)
                try:
                    video_list = sorted(os.listdir(video_read_dir), key=lambda x: int(os.path.splitext(x)[0].split("_")[1]))
                except:
                    import pdb; pdb.set_trace()
                # if file_idx == 16:
                # import pdb; pdb.set_trace()
                temp_video_list.append([])
                for video in video_list:
                    out_video = video
                    if len(temp_video_list[se_idx]) > 0:
                        # if video in temp_video_list[se_idx]:
                        file_idx = int(sorted(temp_video_list[se_idx])[-1].split(".")[0].split("_")[1]) + 1
                        # file_idx = len(temp_video_list[se_idx])
                        out_video = "episode_" + str(file_idx).zfill(self.file_idx_len) + ".mp4"
                    file_path = video_read_dir + "/" + video
                    dest_file_path = str(out_video_dir) + "/" + out_video
                    logger.info("cp " + str(file_path) + " to " + dest_file_path)
                    shutil.copy2(file_path, dest_file_path)

                    temp_video_list[se_idx].append(out_video)
                
                se_idx += 1
    
    
    def merge(self) -> None:
        """执行完整的数据集合并流程"""
        logger.info(f"开始合并 {len(self.dataset_paths)} 个数据集到 {self.output_dir}")
        
        # 步骤1: 合并元数据
        self.merge_metadata()
        
        # 步骤2: 合并数据文件
        self.merge_data_files()
        
        logger.info("数据集合并完成!")

# 示例使用
if __name__ == "__main__":
    # 定义要合并的数据集路径 第一条数据必须是0开头的，可以适当的交换下面数据集顺序
    dataset_paths = [
        "dataset_1119_exec",
        "dataset_1127_good",
    ]
    
    # 定义输出目录
    output_dir = "dataset_1119t27eg"
    
    # 创建合并器实例并执行合并
    merger = DatasetMerger(output_dir, dataset_paths)
    merger.merge()