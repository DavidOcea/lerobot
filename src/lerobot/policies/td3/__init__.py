# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TD3 policy module for residual RL."""

from .config_td3 import (
    TD3ActorConfig,
    TD3CriticConfig,
    TD3Config,
    TD3ConfigAggressive,
    TD3ConfigConservative,
    TD3ConfigFineTune,
)
from .modeling_td3 import TD3Actor, TD3Agent, TD3Critic

__all__ = [
    "TD3Actor",
    "TD3Critic",
    "TD3Agent",
    "TD3ActorConfig",
    "TD3CriticConfig",
    "TD3Config",
    "TD3ConfigConservative",
    "TD3ConfigAggressive",
    "TD3ConfigFineTune",
]