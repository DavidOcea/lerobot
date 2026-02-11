"""
Control module for robot action processing and smooth motion.

This module provides post-processing for robot actions to ensure smooth,
precise, and safe motion.
"""

from lerobot.control.action_post_processor import (
    ActionPostProcessor,
    PostProcessorConfig,
    create_post_processor_for_robot,
)

__all__ = [
    "ActionPostProcessor",
    "PostProcessorConfig",
    "create_post_processor_for_robot",
]
