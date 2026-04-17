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

"""Residual policy module combining BC + TD3."""

from .config_residual import ResidualPolicyConfig, ResidualTrainingConfig
from .modeling_residual import ResidualPolicy, ResidualPolicyEvaluator

__all__ = [
    "ResidualPolicy",
    "ResidualPolicyEvaluator",
    "ResidualPolicyConfig",
    "ResidualTrainingConfig",
]