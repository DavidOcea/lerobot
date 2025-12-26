# lerobot/policies/act/rl_models.py
import torch
import torch.nn as nn
from torchvision import models
from lerobot.policies.pretrained import PreTrainedPolicy

class ResNet18RewardModel(PreTrainedPolicy):
    """基于ResNet18的奖励模型，通过观察图像评估末端执行器与目标的相对位置"""
    def __init__(self, config):
        super().__init__(config)
        self.backbone = models.resnet18(pretrained=False)
        self.backbone.fc = nn.Identity()  # 移除最后的全连接层
        self.projection = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        # 加载预训练权重
        # self.load_pretrained_weights()

    def load_pretrained_weights(self):
        """加载训练好的奖励模型权重"""
        # 实际使用时替换为你的模型路径
        state_dict = torch.load("pretrained_reward_model.pth", map_location=self.device)
        self.load_state_dict(state_dict)
        for param in self.parameters():
            param.requires_grad = False  # 固定权重
        self.eval()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """输入图像返回0-1的奖励分数"""
        features = self.backbone(images)
        reward = torch.sigmoid(self.projection(features))
        return reward.squeeze()

    def get_reward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        """从观测中提取图像计算奖励"""
        with torch.inference_mode():
            img = next(iter(observations.values()))  # 获取第一个图像特征
            if img.dim() == 3:
                img = img.unsqueeze(0)  # 添加批次维度
            return self.forward(img)


class ResNet18DoneModel(PreTrainedPolicy):
    """基于ResNet18的终止模型，判断任务是否完成"""
    def __init__(self, config):
        super().__init__(config)
        self.backbone = models.resnet18(pretrained=False)
        self.backbone.fc = nn.Linear(512, 2)  # 二分类：完成/未完成
        # 加载预训练权重
        # self.load_pretrained_weights()

    def load_pretrained_weights(self):
        """加载训练好的终止模型权重"""
        state_dict = torch.load("pretrained_done_model.pth", map_location=self.device)
        self.load_state_dict(state_dict)
        for param in self.parameters():
            param.requires_grad = False  # 固定权重
        self.eval()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """输入图像返回是否完成的概率"""
        logits = self.backbone(images)
        return torch.softmax(logits, dim=1)[:, 1]  # 完成类别的概率

    def get_done(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        """从观测中提取图像判断是否终止"""
        with torch.inference_mode():
            img = next(iter(observations.values()))
            if img.dim() == 3:
                img = img.unsqueeze(0)
            done_prob = self.forward(img)
            return (done_prob > 0.5).float()  # 阈值判断