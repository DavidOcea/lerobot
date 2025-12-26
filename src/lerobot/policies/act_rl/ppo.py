# lerobot/policies/act/ppo.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

class PPO:
    def __init__(self, policy, config):
        self.policy = policy
        self.gamma = config.gamma  # 折扣因子
        self.lam = config.lam  # GAE参数
        self.clip_epsilon = config.clip_epsilon  # PPO剪辑参数
        self.entropy_coef = config.entropy_coef  # 熵奖励系数
        self.value_coef = config.value_coef  # 价值损失系数
        self.optimizer = optim.Adam(policy.parameters(), lr=config.learning_rate)
        self.config = config

        # 价值函数头（用于PPO的状态价值估计）
        self.value_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        ).to(policy.device)

    def compute_gae(self, rewards, values, dones, next_value):
        """计算广义优势估计"""
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        last_value = next_value

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * last_value * (1 - dones[t]) - values[t]
            last_advantage = delta + self.gamma * self.lam * (1 - dones[t]) * last_advantage
            advantages[t] = last_advantage
            last_value = values[t]
        
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def update(self, trajectories):
        """执行PPO更新"""
        observations = torch.cat([t["obs"] for t in trajectories])
        actions = torch.cat([t["action"] for t in trajectories])
        old_log_probs = torch.cat([t["log_prob"] for t in trajectories])
        advantages = torch.cat([t["advantage"] for t in trajectories])
        returns = torch.cat([t["return"] for t in trajectories])

        # 多次迭代更新
        for _ in range(self.config.ppo_epochs):
            # 重新计算策略输出
            with torch.no_grad():
                # 获取ACT模型的特征表示
                features = self.policy.model.get_features(observations)
                values = self.value_head(features).squeeze()
            
            # 计算新的动作分布和对数概率
            action_dist = Normal(
                loc=self.policy.model.action_head(features),
                scale=torch.exp(self.policy.model.log_std)
            )
            new_log_probs = action_dist.log_prob(actions).sum(dim=1)

            # 计算重要性权重
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # 价值损失
            value_loss = F.mse_loss(values, returns)

            # 熵奖励（鼓励探索）
            entropy = action_dist.entropy().mean()

            # 总损失
            total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            # 优化步骤
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item()
        }
        