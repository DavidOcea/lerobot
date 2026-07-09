# PyTorch → ONNX → TensorRT 部署流程

## 前置条件

| 环境 | 位置 | 用途 |
|------|------|------|
| 开发机 (H100) | `/root/workspace/dc_dir/lerobot` | 导出 ONNX |
| Orin (JetPack 36) | `/home/t/workspace/gitprj/lerobot` | 编译 TRT engine + 推理 |
| Orin 系统库 | `/usr/lib/python3.10/dist-packages/` | TensorRT 8.6 Python 绑定 |

---

## 一、导出 ONNX（开发机 H100，每次 3 分钟）

```bash
# 1. 确认 checkpoint 路径
ls /root/workspace/dc_dir/lerobot/outputs/train/<模型名>/checkpoints/<步数>/pretrained_model/

# 2. 导出 ONNX（默认 480×640，可用 --image_height/--image_width 指定分辨率）
source /root/miniconda3/etc/profile.d/conda.sh && conda activate lerobot
CUDA_VISIBLE_DEVICES=5 python scripts/export_onnx.py \
    --checkpoint outputs/train/<模型名>/checkpoints/<步数>/pretrained_model \
    --image_height 240 \
    --image_width 320 \
    --output outputs/export/backbone_encoder_<模型名>_240x320.onnx \
    --device cuda
```

**验证标准**：
- `ONNX checker passed`
- `Max absolute diff < 2e-2`

```bash
# 3. 确认文件大小
ls -lh outputs/export/backbone_encoder_<模型名>_240x320.onnx
# 预期 ~185MB
```

---

## 二、编译 TensorRT Engine（Orin，每次 5-15 分钟）

### Step 1：拷贝 ONNX 到 Orin

```bash
scp /root/workspace/dc_dir/lerobot/outputs/export/backbone_encoder_<模型名>_240x320.onnx \
    t@<Orin IP>:/home/t/workspace/dc_dir/models/onnx/
```

### Step 2：确认 TensorRT 可用（Orin）

```bash
# 确认 TensorRT Python 可见（Orin 上 ros2_env 需手动链接系统库）
# 首次使用只需执行一次：
echo "/usr/lib/python3.10/dist-packages" >> \
    /home/t/miniconda3/envs/ros2_env/lib/python3.10/site-packages/system_tensorrt.pth

# 验证
python3 -c "import tensorrt; print(tensorrt.__version__)"  # → 8.6.2
```

### Step 3：编译 TRT Engine（Orin）

```bash
python3 << 'EOF'
import tensorrt as trt, os

ONNX   = "/home/t/workspace/dc_dir/models/onnx/backbone_encoder_<模型名>_240x320.onnx"
ENGINE = "/home/t/workspace/dc_dir/models/onnx/backbone_encoder_<模型名>_240x320.engine"
H, W   = 240, 320  # ← 必须与 ONNX 导出的分辨率一致

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

with open(ONNX, 'rb') as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(f"Parse error: {parser.get_error(i)}")
        exit(1)

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)

profile = builder.create_optimization_profile()
profile.set_shape("img0", (1,3,H,W), (1,3,H,W), (1,3,H,W))
profile.set_shape("img1", (1,3,H,W), (1,3,H,W), (1,3,H,W))
profile.set_shape("img2", (1,3,H,W), (1,3,H,W), (1,3,H,W))
profile.set_shape("state", (1,120), (1,120), (1,120))
profile.set_shape("force", (1,120), (1,120), (1,120))
config.add_optimization_profile(profile)

print("Building TRT engine (FP32, ~5-10 min on Orin)...")
engine = builder.build_serialized_network(network, config)
with open(ENGINE, 'wb') as f:
    f.write(engine)

size_mb = os.path.getsize(ENGINE) / 1e6
print(f"✅ FP32 engine: {ENGINE} ({size_mb:.0f} MB)")
EOF
```

**⚠️ 不要加 `config.set_flag(trt.BuilderFlag.FP16)`！** FP16 在 Orin 上会产生 4°+ 的动作偏差。

---

## 三、部署 Engine（Orin）

### Step 1：放置 engine 到 checkpoint 目录

```bash
CHECKPOINT_DIR="/home/t/workspace/dc_dir/models/<模型名>/checkpoints/<步数>/pretrained_model"

# 备份旧 engine（如有）
mv $CHECKPOINT_DIR/backbone_encoder_p2.engine \
   $CHECKPOINT_DIR/backbone_encoder_p2.engine.bak 2>/dev/null

# 拷贝新 engine
cp /home/t/workspace/dc_dir/models/onnx/backbone_encoder_<模型名>_240x320.engine \
   $CHECKPOINT_DIR/backbone_encoder_p2.engine
```

### Step 2：清除 pyc 缓存

```bash
find /home/t/workspace/gitprj/lerobot -path "*policies*__pycache__*" -delete
find /home/t/workspace/gitprj/lerobot -path "*datasets*__pycache__*" -delete
```

### Step 3：确认加载成功（日志检查）

```bash
python -m lerobot.record \
    --policy.path=$CHECKPOINT_DIR \
    ...（其他参数不变）
```

日志中应出现（4 行）：
```
Detected TensorRT engine: .../backbone_encoder_p2.engine
Encoder backend: TensorRT (XX.X MB engine)
✅ ACTPolicyONNX ready
Using ACTPolicyONNX (TensorRT/ONNX accelerated inference)
```

---

## 四、精度验证（可选，建议每次新模型都做）

```bash
python3 << 'EOF'
import torch, numpy as np
from lerobot.policies.act.onnx_policy import ACTPolicyONNX
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset

CKPT   = "/home/t/workspace/dc_dir/models/<模型名>/checkpoints/<步数>/pretrained_model"
ENGINE = CKPT + "/backbone_encoder_p2.engine"
device = torch.device("cuda")

pt = ACTPolicy.from_pretrained(CKPT).to(device).eval()
ox = ACTPolicyONNX(ENGINE, CKPT, "cuda")

ds = LeRobotDataset("dataset_0611_pickup_long_all",
    root="/root/data2/dc_dir/datasets/dataset_0611_pickup_long_all",
    customer_transforms=False, time_warp=False)

it = ds[0]
obs_pt = {
    "observation.state": it["observation.state"].unsqueeze(0).to(device),
    "observation.force": it["observation.force"].unsqueeze(0).to(device),
    "observation.images.head_cam": it["observation.images.head_cam"].unsqueeze(0).to(device),
    "observation.images.left_wrist_cam": it["observation.images.left_wrist_cam"].unsqueeze(0).to(device),
    "observation.images.right_wrist_cam": it["observation.images.right_wrist_cam"].unsqueeze(0).to(device),
}
obs_ox = {k: (v.squeeze(0) if v.ndim==4 else v) for k,v in obs_pt.items()}

pt_a = pt.select_action(obs_pt).flatten()
ox_a = ox.select_action(obs_ox).flatten()
diff = (pt_a - ox_a).abs()
print(f"Mean diff: {diff.mean().item():.4f}°  Max diff: {diff.max().item():.4f}°")
print("✅ OK" if diff.mean().item() < 1.0 else "❌ Rebuild with FP32")
EOF
```

**合格线**：Mean diff < 1°

---

## 五、已知问题与注意事项

| 问题 | 原因 | 解决 |
|------|------|------|
| FP16 engine 偏差 4°+ | ~~FP16 subnormal weights~~ | 始终用 FP32 编译 |
| 录制时实际走 PyTorch | engine 不在 checkpoint 目录 | 确认 `backbone_encoder_p2.engine` 在 checkpoint 同级 |
| `ModuleNotFoundError: tensorrt` | ros2_env 隔离了系统 Python | 第一次加 `.pth` 文件 |
| engine 编译时 dynamic shape 错误 | ONNX 导出了 dynamic batch | 用最新 `export_onnx.py`（已固定 batch=1） |

---

## 六、分辨率选择参考

| 分辨率 | tokens | TRT 延迟 (Orin) | 精度 |
|--------|--------|----------------|------|
| 480×640 | 903 | ~72ms (14Hz) | ✅ 基准 |
| 240×320 | 243 | ~12ms (85Hz) | ✅ 实测无损 |
| 160×213 | 93 | ~6ms (174Hz) | ⚠️ 未实测 |

**推荐**：新模型默认导出 240×320，兼顾速度和精度。
