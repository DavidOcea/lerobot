"""
Precision Place Config - 统一配置

所有启动脚本共用此配置
"""

# 相机索引
CAMERA_INDICES = {
    'head': 0,
    'left_wrist': 2,
    'left_wrist2': 4,
    'right_wrist': 6,
    'right_wrist2': 8
}

# 精准放置主用手腕相机
PRIMARY_WRIST_CAM = 'right_wrist'  # 索引6

# 标记颜色配置
WORKPIECE_MARKER_COLOR = "green"  # 工件标记（绿色）
SLOT_MARKER_COLOR = "red"          # 卡槽标记（红色）

# 使用的手臂
PRIMARY_ARM = "right"

# 精度参数
TOLERANCE_MM = 2.0
MAX_ITERATIONS = 15
