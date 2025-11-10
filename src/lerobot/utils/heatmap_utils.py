import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def overlay_heatmap(original_img, heatmap, alpha=0.4): # alpha 表示热力图合并透明度
    """将热力图与原始图像叠加"""
    # 热力图转为RGB
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    # 调整尺寸并叠加
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    superimposed = heatmap * alpha + original_img * (1 - alpha)
    return np.uint8(superimposed)

def generate_attention_heatmap(model, original_img, idx):
    """带交叉注意力的热力图（img_cross_atten=True）"""
    if len(model.attention_weights) == 0:
        raise ValueError("未记录注意力权重，请确保启用img_cross_atten并完成推理")
    
    # 1. 提取特征（取第一个样本，计算通道激活强度）
    attended_feat = model.attention_weights[idx][:, 0, :].cpu().numpy()  # [H*W,b, D]->[H*W, D]
    activation_map = np.mean(np.abs(attended_feat), axis=1)  # [H*W]，通道平均激活
    # 2. 恢复为特征图尺寸
    h, w = model.feature_map_size[idx]
    activation_map = activation_map.reshape(h, w)
    # 3. 归一化并插值
    activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min() + 1e-8)
    heatmap = cv2.resize(activation_map, model.original_img_size[idx], interpolation=cv2.INTER_LINEAR)
    heatmap = np.uint8(255 * heatmap)
    
    # 4. 与原始图像叠加
    original_np = np.array(original_img) * 255
    return Image.fromarray(overlay_heatmap(original_np, heatmap)), Image.fromarray(np.uint8(original_np))

def generate_activation_heatmap(model, original_img, idx):
    """无交叉注意力的热力图（img_cross_atten=False）"""
    if len(model.image_features) == 0:
        raise ValueError("未记录图像特征，请确保禁用img_cross_atten并完成推理")
    
    # 1. 提取特征图激活值（取第一个样本，计算通道平均绝对值）
    feat_map = model.image_features[idx][0].cpu().numpy()  # (b, C, h, w) -> (c, h, w)
    activation_map = np.mean(np.abs(feat_map), axis=0)  # (h, w) → 通道平均
    # import pdb; pdb.set_trace()
    # 2. 归一化并插值到原始图像尺寸
    activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min() + 1e-8)
    heatmap = cv2.resize(activation_map, model.original_img_size[idx], interpolation=cv2.INTER_LINEAR)
    heatmap = np.uint8(255 * heatmap)
    
    # 3. 与原始图像叠加
    original_np = np.array(original_img) * 255
    return Image.fromarray(overlay_heatmap(original_np, heatmap)), Image.fromarray(np.uint8(original_np)) #.convert('L')