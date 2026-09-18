#!/usr/bin/env python
"""向后兼容冒烟测试：用「改了 MDN 头之后的当前代码」加载一个 pre-MDN 的 K=1 老 checkpoint，
验证 (1) from_pretrained 键名严格匹配成功，(2) select_action 端到端能产出合法动作。

用法: python scripts/smoke_test_backcompat.py [checkpoint_dir]
  默认 = 0915_0818_repro 的 200000 checkpoint（K=1，config.json 无 mdn 字段）。
"""
import sys

import draccus
import torch

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

CKPT = sys.argv[1] if len(sys.argv) > 1 else (
    "outputs/train/train/dp_0915_0818_repro_pickup_long_noaug/checkpoints/200000/pretrained_model"
)

# 复刻 0915_0818_repro 的 launch 参数（K=1，无 mdn 字段）
argv = [
    "--policy.type=diffusion",
    "--dataset.root=/root/data2/dc_dir/datasets/dataset_0729_pickup_long_all",
    "--dataset.repo_id=dataset_0729_pickup_long_all",
    "--batch_size=2",
    "--num_workers=4",
    "--policy.horizon=8",
    "--policy.n_action_steps=1",
    "--policy.n_obs_steps=2",
    "--policy.drop_n_last_frames=6",
    "--policy.down_dims=[256,512]",
    "--policy.noise_scheduler_type=DDIM",
    "--policy.num_inference_steps=5",
    "--policy.crop_is_random=false",
    "--policy.crop_shape=[84,84]",
    "--dataset.time_warp=true",
    "--dataset.customer_transforms=false",
    "--dataset.only_head_transforms=false",
    "--policy.device=cuda",
]
cfg = draccus.parse(config_class=TrainPipelineConfig, args=argv)

print(f"[1/3] from_pretrained: {CKPT}", flush=True)
policy = DiffusionPolicy.from_pretrained(CKPT)
policy = policy.to("cuda").eval()
print(f"      -> load OK. mdn_num_components = {policy.config.mdn_num_components}", flush=True)
assert policy.config.mdn_num_components == 1, "expected K=1 (config.json has no mdn field)"
print(f"      -> final_conv 键名兼容确认（K=1 走原版 final_conv 结构）", flush=True)

print("[2/3] 加载 dataset ...", flush=True)
dataset = make_dataset(cfg)
print(f"      dataset: {dataset.num_frames} frames, {dataset.num_episodes} eps", flush=True)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=2, shuffle=True)
batch = next(iter(dataloader))
batch = {k: v.to("cuda") for k, v in batch.items() if isinstance(v, torch.Tensor)}

print("[3/3] select_action 端到端 ...", flush=True)
with torch.no_grad():
    action = policy.select_action(batch)
print(f"      -> action shape: {tuple(action.shape)}, dtype: {action.dtype}", flush=True)
print(f"      -> action 数值范围: [{action.min().item():.4f}, {action.max().item():.4f}]", flush=True)

assert action.ndim == 3, "action 应为 3D (B, horizon, action_dim)"
assert action.shape[0] == 2 and action.shape[1] == 8, f"action 前两维应为 (batch, horizon)，got {tuple(action.shape)}"
assert torch.isfinite(action).all(), "action 含 NaN/Inf"
print("\n>>> PASS：pre-MDN 老 checkpoint 用新代码加载 + 推理均正常", flush=True)
