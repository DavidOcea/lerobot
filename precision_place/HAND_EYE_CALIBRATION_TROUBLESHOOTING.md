# 手眼标定问题排查指南

## 常见问题：检测不到标定板

### 症状
运行手眼标定时，界面显示 "Board NOT Detected" 或 "未检测到标定板"

### 可能原因及解决方案

#### 1. ChArUco板参数不匹配 ⭐ 最常见

**当前代码默认参数:**
- 格子数: 11x8 (X方向11格, Y方向8格)
- 格子边长: 30mm
- 标记边长: 22mm
- ArUco字典: DICT_6X6_250

**检查方法:**
```bash
# 运行诊断工具
cd /home/t/workspace/gitprj/lerobot
python precision_place/test_charuco_detection.py
```

诊断工具会提示你输入实际标定板参数,并实时显示检测结果。

**修改参数:**
在 `run.py` 的手眼标定流程中,系统会询问你标定板参数:
```
格子边长(mm) [默认30]: 
标记边长(mm) [默认22]: 
```

确保输入的参数与你的实际标定板完全一致!

#### 2. 相机内参不准确

**症状:** 能检测到角点但位姿估计不准,或者检测率很低

**解决:**
```bash
# 先运行相机内参标定
python precision_place/run.py
# 选择选项 K - 相机内参标定
```

标定完成后会生成 `camera_intrinsics.yaml` 文件,手眼标定会自动使用该文件。

#### 3. 光照和对焦问题

**检查清单:**
- ✓ 标定板清晰可见,没有模糊
- ✓ 光照均匀,没有强烈反光或阴影
- ✓ 相机对焦准确
- ✓ 标定板完整在视野内

**调试技巧:**
- 保存一帧图像检查质量: 在诊断工具中按 'S' 键
- 调整相机距离: 建议30-80cm
- 改善照明: 使用柔光,避免直射光源

#### 4. OpenCV版本问题

**检查版本:**
```python
import cv2
print(cv2.__version__)
```

- **OpenCV >= 4.7**: 使用 `CharucoDetector` (推荐)
- **OpenCV < 4.7**: 使用旧版API,可能需要升级

**升级OpenCV:**
```bash
pip install opencv-contrib-python>=4.7.0
```

#### 5. 相机索引错误

**确认相机索引:**
```bash
ls /dev/video*
```

常见配置:
- 头部相机: 0
- 左手腕相机: 2
- 右手腕相机: 4

**测试相机:**
```python
import cv2
cap = cv2.VideoCapture(0)  # 尝试不同索引
ret, frame = cap.read()
if ret:
    print(f"相机打开成功: {frame.shape}")
else:
    print("相机打开失败")
cap.release()
```

## 快速诊断流程

### 步骤1: 运行诊断工具
```bash
cd /home/t/workspace/gitprj/lerobot
python precision_place/test_charuco_detection.py
```

该工具会:
- 实时显示检测结果
- 显示检测率统计
- 提供详细诊断信息
- 允许保存测试图像

### 步骤2: 检查参数
根据诊断工具的提示,确认:
- 格子数是否正确
- 格子边长和标记边长是否匹配
- 相机内参是否准确

### 步骤3: 调整环境
- 调整相机距离 (30-80cm)
- 改善光照条件
- 确保对焦清晰

### 步骤4: 重新标定
如果诊断工具能稳定检测到标定板 (>80%检测率),再运行手眼标定。

## 高级调试

### 查看检测日志
在手眼标定过程中,控制台会输出:
```
✓ 捕获成功 (1) - 基准姿态已设置
```

如果一直看不到这条消息,说明检测失败。

### 保存调试图像
诊断工具可以保存测试图像:
```bash
# 按 'S' 键保存
test_charuco_frame.jpg
```

用图像查看器打开,检查:
- 标定板是否清晰
- 是否有足够的对比度
- 角点是否被正确标记

### 降低检测阈值 (临时调试)

如果怀疑阈值太高,可以临时修改 `hand_eye.py`:

```python
# 找到 detect_charuco 方法中的这行
if charuco_ids is None or len(charuco_ids) < 6:
    return False, None, None, None

# 临时改为 (仅用于调试)
if charuco_ids is None or len(charuco_ids) < 4:  # 降低阈值
    return False, None, None, None
```

⚠️ **注意:** 这只是调试手段,不建议长期使用低阈值。

## 常见错误信息

### "未检测到足够的ChArUco角点"
- 原因: 检测到的角点数 < 6
- 解决: 检查标定板参数、光照、距离

### "正运动学计算失败"
- 原因: URDF文件路径错误或格式问题
- 解决: 检查URDF文件路径,确保关节名称匹配

### "同步捕获失败"
- 原因: 相机读取超时或关节状态获取失败
- 解决: 检查相机连接,确认机器人已连接

## 预防措施

1. **首次使用前**,先用诊断工具测试检测效果
2. **定期校准**相机内参 (特别是更换相机后)
3. **记录**成功的标定参数,便于下次使用
4. **保持**标定板清洁,避免磨损

## 联系支持

如果以上方法都无法解决问题,请提供:
1. 诊断工具的输出截图
2. 保存的测试图像 (`test_charuco_frame.jpg`)
3. 标定板的实际参数照片
4. 控制台错误日志

---

**最后更新:** 2024-04-18
**相关工具:** `test_charuco_detection.py`, `hand_eye.py`
