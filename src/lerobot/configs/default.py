#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from lerobot import (
    policies,  # noqa: F401
)
from lerobot.datasets.transforms import ImageTransformsConfig
from lerobot.datasets.video_utils import get_safe_default_codec
from PIL import Image

@dataclass
class DatasetConfig:
    # You may provide a list of datasets here. `train.py` creates them all and concatenates them. Note: only data
    # keys common between the datasets are kept. Each dataset gets and additional transform that inserts the
    # "dataset_index" into the returned item. The index mapping is made according to the order in which the
    # datasets are provided.
    repo_id: str
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: str | None = None
    episodes: list[int] | None = None
    image_transforms: ImageTransformsConfig = field(default_factory=ImageTransformsConfig)
    revision: str | None = None
    use_imagenet_stats: bool = True
    video_backend: str = field(default_factory=get_safe_default_codec)
    customer_transforms: bool = False
    # 仅对头部图像坐增强
    only_head_transforms: bool = False
    # 图像数据增强，为了适应相机角度可能变化。
    customer_transforms_cfg: dict = field(default_factory=lambda:{
        # "random_resize_crop":{ # 随机缩放裁剪
        #     "size":(480, 640),    # 最终输出尺寸 h w
        #     "scale":(0.7, 1.3),   # 随机缩放范围（相对于原始图像的比例）
        #     "ratio":(3/4, 4/3),    # 随机长宽比范围3/4, 4/3 |固定长宽比为 1:1
        #     "interpolation":Image.BILINEAR  # 插值方法
        # },
        "random_rotation":{  # 随机旋转
            "degrees":(-6, 6),  # 角度范围：-15° 到 15°
            "expand":False,        # 扩展图像，避免裁剪
            "fill":(255, 255, 255),  # 白色填充
            "interpolation":Image.BILINEAR  # 双线性重采样（画质更平滑）
        },
        # "colorjitter":{ # 随机亮度变化
        #     "brightness":0.2,  # 亮度 ±20%
        #     "contrast":0.2,    # 对比度 ±20%
        #     "saturation":0.2,  # 饱和度 ±20%
        #     "hue":0.1          # 色相 ±0.1（避免颜色失真）
        # },
        # "randomerase":{
        #     "p":0.8,      #触发概率80%
        #     "scale":(0.02, 0.33), #擦除区域占比2%~33%
        #     "ratio":(0.3, 3.3), 
        #     "value":0
        # },
        # "randorm_affine":{
        #     "degrees": 0,
        #     "translate":(0.1,0.1), # 平移尺度
        #     "scale":(1.0,1.0) # 缩放比例
        # }
    })
    # Timestamp tolerance in seconds for delta_timestamps validation.
    # Use 0.03 for datasets recorded with actual timestamps (perf_counter).
    # Use 1e-4 for datasets recorded with ideal timestamps (frame_index/fps).
    tolerance_s: float = 1e-4



@dataclass
class WandBConfig:
    enable: bool = False
    # Set to true to disable saving an artifact despite training.save_checkpoint=True
    disable_artifact: bool = False
    project: str = "lerobot"
    entity: str | None = None
    notes: str | None = None
    run_id: str | None = None
    mode: str | None = None  # Allowed values: 'online', 'offline' 'disabled'. Defaults to 'online'


@dataclass
class EvalConfig:
    n_episodes: int = 50
    # `batch_size` specifies the number of environments to use in a gym.vector.VectorEnv.
    batch_size: int = 50
    # `use_async_envs` specifies whether to use asynchronous environments (multiprocessing).
    use_async_envs: bool = False

    def __post_init__(self):
        if self.batch_size > self.n_episodes:
            raise ValueError(
                "The eval batch size is greater than the number of eval episodes "
                f"({self.batch_size} > {self.n_episodes}). As a result, {self.batch_size} "
                f"eval environments will be instantiated, but only {self.n_episodes} will be used. "
                "This might significantly slow down evaluation. To fix this, you should update your command "
                f"to increase the number of episodes to match the batch size (e.g. `eval.n_episodes={self.batch_size}`), "
                f"or lower the batch size (e.g. `eval.batch_size={self.n_episodes}`)."
            )
