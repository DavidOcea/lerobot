# P4: PyTorch → ONNX 部署 — 最小风险分步执行方案

## 一、推理延迟对比

| 后端 | ResNet18 + Encoder | Decoder (1层自回归) | 总计 | Orin 30Hz? | 环境依赖 |
|------|-------------------|---------------------|------|-----------|---------|
| PyTorch (FP32) | ~55ms | ~25ms | **~80ms** | ❌ | 完整 PyTorch |
| PyTorch (FP16) | ~30ms | ~15ms | **~45ms** | ❌ | 完整 PyTorch |
| ONNX Runtime (CUDA) | ~18ms | — | **~18ms** | ✅ | `onnxruntime-gpu` |
| ONNX Runtime (CPU) | ~50ms | — | **~50ms** | ❌ | `onnxruntime` |

> Decoder 保留 PyTorch，不导出。因为自回归 for-loop + causal mask + RoPE 是 ONNX trace 的已知难点，且耗时占比小（~25ms），不值得冒险。

## 二、导出拆分方案：Backbone + Encoder → ONNX，Decoder 保留 PyTorch

```
PyTorch 推理（现在）:
  images(3张) ──→ Backbone ──→ img_feats
  state ──→ Linear ──→ state_token  ──╮
  force ──→ Linear ──→ force_token  ──┤
  latent ──→ Linear ──→ latent_token ──┤ ╭── Encoder ──→ encoder_out
                                        ├─┤         (8层 Self-Attn)
                                        ╰─┤
                                           ╰──→ Decoder ──→ action chunk(20)
                                                (1层, 自回归)    ↓
                                                         temporal ensemble
                                                               ↓
                                                          final action

ONNX 混合推理（目标）:
  images ──→ [Backbone+Encoder.onnx] ──→ encoder_out ──→ Decoder(PyTorch) ──→ action
  state ───→                          ↑
  force ──→                           │ 这两块合在一起导出，80% 耗时都在这里
  latent ── zeros (推理时)

改动:
  训练代码: 0 行改动
  模型文件: 0 行改动
  新增:     export_onnx.py (~80行) + onnx_policy.py (~120行)
  推理侧:   ACTPolicy.from_pretrained → ONNXPolicy.from_checkpoint (~5行)
```

## 三、算子兼容性预检

ACT Encoder 路径使用的算子:

| 算子 | 出现位置 | ONNX 兼容性 |
|------|---------|------------|
| `nn.MultiheadAttention` | Encoder 8 层 | ✅ PyTorch 2.x 原生支持 |
| `nn.Linear` | token projections | ✅ |
| `nn.LayerNorm` | Encoder 每层 | ✅ |
| `nn.ReLU` (gelu) | FFN 激活 | ✅ |
| `einops.rearrange` | 图像特征展平 (2处, line 848-849) | ⚠️ trace 时会展开为 torch ops，需验证 |
| `einops.repeat` | VAE cls embed (line 666) | ⚠️ 同上，但 VAE 推理时不用（latent=zeros） |
| `torch.cat` | token 拼接 (line 860) | ✅ |
| `torch.stack` | token 堆叠 (line 860) | ✅ |
| `F.dropout` | Encoder 每层 | ✅ 推理时自动关闭 |
| for-loop `ModuleList` | Encoder 8 层迭代 | ✅ trace 时静态展开 |
| ACTSinusoidalPositionEmbedding2d | 图像位置编码 | ✅ 标准数学运算 |

**唯一风险点：`einops.rearrange`。** 但这是 einsum 语法糖，ONNX trace 时会展开为 `torch.reshape` + `torch.permute`，标准算子。如果 trace 失败，只需手动展开 2 个 rearrange 调用即可，改动量 ~5 行。

## 四、分步执行计划

### Step 1: 算子验证脚本（30 分钟，零风险）

```python
# scripts/check_onnx_ops.py
# 目的: 验证 ACT 模型的所有子模块能否独立 trace
# 操作:
#   1. 加载 checkpoint
#   2. 分别 trace backbone + encoder + decoder
#   3. 逐个验证 torch.jit.trace 成功
#   4. 如果有 submodule trace 失败 → 定位具体算子 → 决定手写替换
# 验证通过标准: 所有 submodule trace 成功
# 可回滚: 独立脚本，不碰任何现有代码
```

### Step 2: ONNX 导出脚本（1 小时，低风险）

```python
# scripts/export_onnx.py
# 目的: 导出 backbone+encoder 为一个 .onnx 文件
#
# 输入:
#   - 3 images:  (1, 3, 480, 640) each
#   - state:     (1, 15) 或 (1, n_obs*15)
#   - force:     (1, 15) 或 (1, n_obs*15)
#
# 输出:
#   - encoder_out: (total_tokens, 512)
#
# 处理:
#   1. 创建 ACTExportModule(nn.Module):
#      - 包含 model.backbone + model.encoder_*_proj + model.encoder
#      - 不包含 model.decoder
#      - 不包含 VAE encoder (推理时 latent 为零，直接用 encoder_latent_input_proj(zeros))
#      - 不包含 normalize/unnormalize (在 PyTorch 侧做)
#      - 不包含 img_cross_atten 路径 (我们的 config 不用)
#
#   2. torch.onnx.export() → backbone_encoder.onnx
#
#   3. onnx.checker.check_model() 验证
#
#   4. onnxruntime 推理对比 PyTorch 推理 → MSE < 1e-5
#
#   5. 如果验证失败 → 排查算子 → 手动展开或替换
#
# 验证通过标准:
#   - ONNX checker 零警告
#   - ORT vs PyTorch 输出 MSE < 1e-5
#   - 在 GPU 上推理耗时 < 20ms
#
# 可回滚: 独立脚本
```

### Step 3: ONNX 推理封装层（1 小时，低风险）

```python
# src/lerobot/policies/act/onnx_policy.py
# 目的: 提供与 ACTPolicy 接口完全一致的推理类
#
# class ACTPolicyONNX:
#     def __init__(self, onnx_path, checkpoint_path):
#         # 1. 从 checkpoint 加载 config + normalize stats
#         # 2. 初始化 ONNX Runtime session (CUDA)
#         # 3. 初始化 PyTorch decoder (from checkpoint weights)
#         # 4. 初始化 temporal_ensembler
#         # 5. 初始化 state buffer (n_obs_steps > 1)
#
#     def select_action(self, obs):
#         # 1. normalize_inputs(obs)
#         # 2. state_buffer.append(state) → stack if n_obs > 1
#         # 3. ort_session.run({"images": ..., "state": ..., "force": ...}) → encoder_out
#         # 4. decoder(encoder_out) → action_chunk (PyTorch)
#         # 5. temporal_ensembler.update(chunk) → single_action
#         # 6. unnormalize(action) → return
#
#     def reset(self):
#         # 清空 buffer + temporal_ensembler
#
# 验证通过标准:
#   - select_action 输出 vs 纯 PyTorch 输出 diff < 0.005
#   - 连续 1000 帧无内存泄漏
#   - 推理延迟 < 25ms (30Hz 达标)
```

### Step 4: 集成到 temp-agent（30 分钟，极低风险）

```python
# 调用侧改动 (~5 行):
if config.use_onnx:
    from lerobot.policies.act.onnx_policy import ACTPolicyONNX
    policy = ACTPolicyONNX(
        onnx_path="outputs/export/backbone_encoder.onnx",
        checkpoint_path="outputs/train/act_xxx/checkpoints/last/pretrained_model",
    )
else:
    policy = ACTPolicy.from_pretrained("outputs/train/act_xxx/...")

# 接口完全一致:
action = policy.select_action(obs)  # 不变量
policy.reset()                       # 不变量
```

## 五、风险控制总结

```
Step  内容                    风险    改动量       验证标准
──────────────────────────────────────────────────────────
1     算子兼容性预检           零      独立脚本     所有 submodule trace 成功
2     ONNX 导出 + 精度验证     低      独立脚本     MSE < 1e-5, 延迟 <20ms
3     ONNX 推理封装            中      ~120行新文件  diff < 0.005, 1000帧稳定
4     temp-agent 集成          极低    ~5行改动      接口兼容
```

每一步独立可验证可回滚。最坏情况 Step 2 某个算子在 ONNX 中不兼容 → 回退到 PyTorch 纯推理，不阻塞部署。

## 六、和现有功能的关系

| 现有功能 | ONNX 后是否仍可用 |
|----------|-----------------|
| train.py | 零影响（ONNX 是推理专用） |
| P0 time_warp | 零影响（训练侧） |
| P1 DAgger | 零影响（推理接口一致） |
| P2 n_obs_steps | ✅ ONNX 推理封装中维护 state buffer |
| P3 aug ensemble | ✅ 不冲突（每个 view 调一次 ONNX session） |
| temporal_ensemble | ✅ PyTorch 侧保留 |
| chunk_size 配置 | ✅ 从 checkpoint config 读取 |
