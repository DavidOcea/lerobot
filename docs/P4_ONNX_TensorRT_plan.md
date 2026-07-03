# P4: ONNX + TensorRT 导出 — 最小风险分步执行计划

## 总体策略：拆分导出（Backbone + Encoder → TRT，Decoder 保留 PyTorch）

ACT 推理的三个阶段中，ResNet18 backbone 和 Transformer encoder
是标准算子组合（Conv + Linear + LayerNorm + Attention），ONNX/TensorRT
支持成熟。自回归 decoder 的 for-loop + RoPE + causal mask 是 ONNX
导出的已知难点。

**拆分方案**：只把 backbone 和 encoder 导出到 TensorRT，decoder 和
temporal ensemble 保留 PyTorch。收益占 70-80% 的总加速，风险几乎为零。


## Step 1: 纯 Backbone ONNX 导出验证（零风险，~30 行代码）

目标：验证 ResNet18 + 后处理层能正确导出 ONNX，输入输出精度无损。

操作：
```python
# export_backbone_onnx.py
# 1. 加载 ACT checkpoint
# 2. 提取 backbone + Conv2d projection 子模块
# 3. 构造 dummy 输入 (1, 3, 480, 640) × 3 cameras
# 4. torch.onnx.export() → backbone.onnx
# 5. 对比 PyTorch 输出 vs ONNX Runtime 输出 → 确保 diff < 1e-5
```

验证点：
- [ ] ONNX 导出无警告/错误
- [ ] PyTorch vs ONNX 输出 MSE < 1e-5
- [ ] 输入 shape 和 checkpoint 一致

可回滚性：完全独立脚本，不碰任何现有代码。


## Step 2: Backbone TensorRT 编译（低风险，~20 行代码）

目标：在 H100 上编译 backbone.onnx → backbone.engine，验证延迟。

操作：
```bash
trtexec --onnx=backbone.onnx --fp16 --saveEngine=backbone.engine
# 验证: trtexec --loadEngine=backbone.engine --fp16
```

验证点：
- [ ] 编译成功，无算子 fallback 警告
- [ ] FP16 输出 vs PyTorch FP32 输出 diff < 0.001
- [ ] 延迟: <5ms for 3 images (batch=1)

如果 TensorRT 编译失败某个算子，记录具体算子名，调整模型拆分方案。


## Step 3: Encoder 拆分（关键步骤，低风险，~50 行代码）

目标：把 encoder 从 ACT model 中拆出，做成独立可导出的子模块。

ACT.forward 中 encoder 的输入 tokens 来源：
```
latent_token (B, 1, 512)   ← Linear(32, 512) — 推理时输入零向量
state_token (B, 1, 512)    ← Linear(15/120, 512)
force_token (B, 1, 512)    ← Linear(15/120, 512)
image_feats (B, N, 512)     ← Backbone → Conv2d → flatten
                              (N = 15*20*3 = 900 for 3 cameras at 480×640)
```

操作：
```python
# 构造 EncoderModule(nn.Module):
#   def forward(self, image_features, state, force):
#       latent → Linear
#       state  → Linear
#       force  → Linear
#       concat + positional encoding
#       → 8 layers self-attention
#       → return encoder_output
#
#   torch.onnx.export(EncoderModule, ...) → encoder.onnx
```

验证点：
- [ ] 拆分后 PyTorch forward 输出和原模型完全一致
- [ ] ONNX 导出成功
- [ ] ONNX Runtime 推理输出 diff < 1e-5

注意：Encoder 使用标准的 nn.MultiheadAttention / 自写的 attention layer。
需要确认用的是哪种。如果是自写的，检查是否有 `einops` / `torch.chunk` 等
可能导致 ONNX trace 失败的操作。


## Step 4: Encoder TensorRT 编译（低风险，~20 行代码）

同 Step 2，编译 encoder.onnx → encoder.engine。


## Step 5: 组装 PyTorch-TRT 混合推理（中等风险，~80 行代码）

目标：创建一个 `ACTPolicyTRT` 类，内部用 TensorRT engine 跑 backbone+encoder，
decoder 和 temporal ensemble 保留 PyTorch。

```python
class ACTPolicyTRT:
    def __init__(self, checkpoint_path):
        # 1. 从 checkpoint 加载 config
        # 2. 加载 backbone.engine, encoder.engine
        # 3. 初始化 decoder (PyTorch), temporal_ensembler, normalize
        # 4. state buffer (for n_obs_steps > 1)

    def select_action(self, obs):
        # 1. 预处理图像（resize, normalize）
        # 2. backbone_engine.infer(images) → features
        # 3. encoder_engine.infer(features, state, force) → encoder_output
        # 4. decoder(encoder_output) → action chunk (PyTorch)
        # 5. temporal_ensembler.update(chunk) → single action
        # 6. unnormalize(action) → 返回
```

验证点：
- [ ] PyTorch 纯版 vs TRT 混合版输出 diff < 0.005（FP16 tolerance）
- [ ] 推理延迟 < 20ms（目标 50Hz）
- [ ] 连续 1000 帧无内存泄漏、无 NaN


## Step 6: 推理接口兼容层（低风险，~10 行代码）

目标：让 temp-agent 分支能无缝切换 PyTorch 和 TensorRT 后端。

```python
# 调用侧不变:
if use_trt:
    policy = ACTPolicyTRT.from_checkpoint(ckpt_path)
else:
    policy = ACTPolicy.from_pretrained(ckpt_path)

action = policy.select_action(obs)  # 接口完全一致
```


## 风险控制总结

```
Step  内容              风险    改动量    收益验证
─────────────────────────────────────────────────
1     Backbone ONNX      零     独立脚本   输出 diff < 1e-5
2     Backbone TRT       极低   独立脚本   延迟 <5ms
3     Encoder 拆分       低     50行新     输出完全一致
4     Encoder TRT        极低   独立脚本   延迟 <10ms
5     混合推理组装        中     80行新     延迟 <20ms
6     接口兼容层          极低   10行改动   接口不变
```

每一步都可以独立验证和回滚。建议先做 Step 1-2，确认 backbone 导出无障碍后再做 3-4。
全流程在最坏情况下可以用 ONNX Runtime 替代 TensorRT（损失 20% 加速但 100% 兼容）。

## 预估总改动量

```
新增文件:
  scripts/export_backbone_onnx.py     ~30 行
  scripts/export_encoder_onnx.py      ~50 行
  scripts/build_trt.py                ~20 行
  src/lerobot/policies/act/inference_trt.py    ~80 行

修改文件:
  推理入口（调用侧）                   ~10 行
  总计                               ~190 行
```

## 对现有功能的兼容

| 现有功能 | 影响 |
|----------|------|
| train.py | 零影响 — TRT 是推理专用 |
| P0 time_warp | 零影响 — 训练侧 |
| P1 DAgger | 零影响 — 推理接口不变 |
| P2 n_obs_steps | 需要在 TRT 推理中维护 state buffer（已设计） |
| P3 aug ensemble | 如果启用，每个 view 独立调 backbone+encoder TRT engine |
| chunk_size / temporal_ensemble | 在 PyTorch decoder 侧保留，TRT 不碰 |
