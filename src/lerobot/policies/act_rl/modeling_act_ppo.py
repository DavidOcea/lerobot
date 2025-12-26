# lerobot/policies/act/modeling_act_ppo.py
import torch
from torch import Tensor
from collections import deque
import numpy as np

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.rl_models import ResNet18RewardModel, ResNet18DoneModel
from lerobot.policies.act.ppo import PPO
from lerobot.policies.act.configuration_act import ACTConfig

class ACTPPOPolicy(ACTPolicy):
    def __init__(self, config: ACTConfig):
        super().__init__(config)
        self.config = config
        
        # 初始化奖励和终止模型
        self.reward_model = ResNet18RewardModel(config)
        self.done_model = ResNet18DoneModel(config)
        
        # 初始化PPO优化器
        self.ppo = PPO(self, config.ppo_config)
        
        # 轨迹缓存
        self.trajectory_buffer = []
        self.current_trajectory = []

    def reset(self):
        super().reset()
        self.current_trajectory = []

    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """选择动作并记录轨迹用于PPO更新"""
        with torch.no_grad():
            action = super().select_action(batch)
            
            # 计算动作对数概率（用于PPO）
            if self.config.image_features:
                batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
            features = self.model.get_features(batch)
            action_dist = Normal(
                loc=self.model.action_head(features),
                scale=torch.exp(self.model.log_std)
            )
            log_prob = action_dist.log_prob(action).sum()

            # 记录轨迹
            self.current_trajectory.append({
                "obs": {k: v.clone() for k, v in batch.items()},
                "action": action.clone(),
                "log_prob": log_prob.clone()
            })

            # 计算奖励和终止信号
            reward = self.reward_model.get_reward(batch)
            done = self.done_model.get_done(batch)

            # 如果终止，添加奖励并保存轨迹
            if done:
                self._finalize_trajectory(reward, done)
                
        return action

    def _finalize_trajectory(self, final_reward, final_done):
        """完成轨迹并准备PPO更新"""
        # 为轨迹添加奖励和终止信号
        for t in range(len(self.current_trajectory)):
            # 最后一步使用实际奖励，其他步骤使用0（或中间奖励）
            reward = final_reward if t == len(self.current_trajectory) - 1 else 0.0
            done = final_done if t == len(self.current_trajectory) - 1 else 0.0
            self.current_trajectory[t]["reward"] = torch.tensor(reward, device=self.device)
            self.current_trajectory[t]["done"] = torch.tensor(done, device=self.device)

        # 计算GAE
        values = [self.ppo.value_head(self.model.get_features(t["obs"])).item() 
                 for t in self.current_trajectory]
        next_value = 0.0 if final_done else self.ppo.value_head(
            self.model.get_features(self.current_trajectory[-1]["obs"])).item()
        
        advantages, returns = self.ppo.compute_gae(
            torch.tensor(values, device=self.device),
            torch.tensor([t["reward"] for t in self.current_trajectory], device=self.device),
            torch.tensor([t["done"] for t in self.current_trajectory], device=self.device),
            torch.tensor(next_value, device=self.device)
        )

        # 添加优势估计和回报
        for i in range(len(self.current_trajectory)):
            self.current_trajectory[i]["advantage"] = advantages[i]
            self.current_trajectory[i]["return"] = returns[i]

        # 添加到轨迹缓冲区
        self.trajectory_buffer.extend(self.current_trajectory)
        self.current_trajectory = []

        # 当缓冲区足够大时执行PPO更新
        if len(self.trajectory_buffer) >= self.config.ppo_config.batch_size:
            self._perform_ppo_update()
            self.trajectory_buffer = []

    def _perform_ppo_update(self):
        """执行PPO更新步骤"""
        self.train()  # 切换到训练模式
        loss_info = self.ppo.update(self.trajectory_buffer)
        self.eval()   # 切回评估模式
        print(f"PPO Update - Policy Loss: {loss_info['policy_loss']:.4f}, "
              f"Value Loss: {loss_info['value_loss']:.4f}")

    def save_pretrained(self, save_directory):
        """保存模型，确保兼容LeRobot推理"""
        super().save_pretrained(save_directory)
        # 保存PPO价值头
        torch.save(self.ppo.value_head.state_dict(), 
                  f"{save_directory}/ppo_value_head.pth")