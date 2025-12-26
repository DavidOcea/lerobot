# lerobot/policies/act/configuration_act.py
from dataclasses import dataclass
from lerobot.policies.act import ACTConfig

@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    learning_rate: float = 3e-4
    ppo_epochs: int = 10
    batch_size: int = 512
    max_grad_norm: float = 0.5

@dataclass
class ACTRLConfig(ACTConfig):
    
    # PPO配置
    ppo_config: PPOConfig = PPOConfig()
    
    # 奖励和终止模型路径
    reward_model_path: str = "pretrained_reward_model.pth"
    done_model_path: str = "pretrained_done_model.pth"