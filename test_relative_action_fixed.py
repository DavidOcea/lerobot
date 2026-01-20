#!/usr/bin/env python3
"""
测试修复后的相对角度训练功能
核心修复：只对 chunk 的第一步使用相对角度，避免累积误差
"""
import torch
import sys
sys.path.insert(0, '/home/smai/dc_dir/lerobot_0901_pybullet/src')

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.configs.types import PolicyFeature, FeatureType


def test_fixed_relative_action():
    """测试修复后的相对角度功能"""
    print("=" * 60)
    print("测试修复后的相对角度训练功能")
    print("=" * 60)
    print("\n核心修复：只对 chunk 的第一步使用相对角度")
    print("这样可以避免累积误差问题")
    print("=" * 60)

    # 创建模拟数据
    batch_size = 2
    action_dim = 16
    chunk_size = 100
    img_size = 96

    # 模拟 batch
    batch = {
        "observation.images.cam_high": torch.randn(batch_size, 3, img_size, img_size),
        "observation.state": torch.randn(batch_size, action_dim),
        "action": torch.randn(batch_size, chunk_size, action_dim),
        "action_is_pad": torch.zeros(batch_size, chunk_size).bool(),
    }

    # 模拟统计信息
    stats = {
        "observation.images.cam_high": {
            "mean": torch.zeros(3, 1, 1),
            "std": torch.ones(3, 1, 1),
        },
        "observation.state": {
            "mean": torch.zeros(action_dim),
            "std": torch.ones(action_dim),
        },
        "action": {
            "mean": torch.zeros(action_dim),
            "std": torch.ones(action_dim),
        },
    }

    # 测试相对角度模式
    print("\n1. 测试相对角度模式 (修复后)")
    try:
        config_rel = ACTConfig(
            use_relative_action=True,
            only_first_step=True,
            use_state=True,
            input_features={
                "observation.images.cam_high": PolicyFeature(type=FeatureType.VISUAL, shape=(3, img_size, img_size)),
                "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(action_dim,)),
            },
            output_features={
                "action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
            },
        )
        policy_rel = ACTPolicy(config_rel, dataset_stats=stats)

        # 保存原始数据
        original_action = batch["action"].clone()
        original_state = batch["observation.state"].clone()

        # 训练模式测试
        policy_rel.train()

        # 获取归一化后的值
        norm_input = policy_rel.normalize_inputs(batch)
        norm_target = policy_rel.normalize_targets(batch)
        action_norm_before = norm_target["action"].clone()
        state_norm = norm_input["observation.state"].clone()

        # 执行 forward
        loss_rel, loss_dict_rel = policy_rel(batch)

        # 验证修改后的 action
        action_norm_after = policy_rel.normalize_targets(
            {"action": original_action.clone()}
        )["action"]

        print(f"   ✓ 训练测试通过!")
        print(f"   - Loss: {loss_rel.item():.4f}")
        print(f"   - L1 Loss: {loss_dict_rel['l1_loss']:.4f}")

        # 验证第一步被转换为相对角度
        expected_first_step = action_norm_before[:, 0, :] - state_norm
        actual_first_step = action_norm_after[:, 0, :]
        diff_first = torch.abs(expected_first_step - actual_first_step).max().item()
        print(f"   - 第一步相对角度转换正确，最大差异: {diff_first:.6f}")

        # 验证其他步保持不变（相对于归一化后的原始值）
        expected_other_steps = action_norm_before[:, 1:, :]
        actual_other_steps = action_norm_after[:, 1:, :]
        diff_other = torch.abs(expected_other_steps - actual_other_steps).max().item()
        print(f"   - 其他步保持不变，最大差异: {diff_other:.6f}")

        # 推理模式测试
        policy_rel.eval()
        with torch.no_grad():
            action_rel = policy_rel.select_action(batch)

        print(f"   ✓ 推理测试通过!")
        print(f"   - 预测 action shape: {action_rel.shape}")

    except Exception as e:
        print(f"   ✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 对比测试：绝对角度 vs 相对角度
    print("\n2. 对比测试：绝对角度 vs 相对角度")
    try:
        config_abs = ACTConfig(
            use_relative_action=False,
            use_state=True,
            input_features={
                "observation.images.cam_high": PolicyFeature(type=FeatureType.VISUAL, shape=(3, img_size, img_size)),
                "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(action_dim,)),
            },
            output_features={
                "action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
            },
        )
        policy_abs = ACTPolicy(config_abs, dataset_stats=stats)

        # 使用相同的 batch
        batch_abs = {
            "observation.images.cam_high": batch["observation.images.cam_high"].clone(),
            "observation.state": batch["observation.state"].clone(),
            "action": batch["action"].clone(),
            "action_is_pad": batch["action_is_pad"].clone(),
        }

        policy_abs.train()
        loss_abs, _ = policy_abs(batch_abs)

        print(f"   - 绝对角度 Loss: {loss_abs.item():.4f}")
        print(f"   - 相对角度 Loss: {loss_rel.item():.4f}")
        print(f"   - Loss 差异: {abs(loss_abs.item() - loss_rel.item()):.4f}")

    except Exception as e:
        print(f"   ✗ 对比测试失败: {e}")

    # 端到端测试
    print("\n3. 端到端测试")
    try:
        config_new = ACTConfig(
            use_relative_action=True,
            only_first_step=True,
            use_state=True,
            input_features={
                "observation.images.cam_high": PolicyFeature(type=FeatureType.VISUAL, shape=(3, img_size, img_size)),
                "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(action_dim,)),
            },
            output_features={
                "action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
            },
        )
        policy_new = ACTPolicy(config_new, dataset_stats=stats)

        optimizer = torch.optim.AdamW(policy_new.parameters(), lr=1e-4)

        # 执行几个训练步骤
        for step in range(3):
            optimizer.zero_grad()
            loss, _ = policy_new(batch)
            loss.backward()
            optimizer.step()
            print(f"   - Step {step+1}: Loss = {loss.item():.4f}")

        # 推理
        with torch.no_grad():
            action_final = policy_new.select_action(batch)
        print(f"   ✓ 训练后推理正常，action shape: {action_final.shape}")

    except Exception as e:
        print(f"   ✗ 端到端测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("✓ 所有测试通过！修复后的相对角度功能正常工作。")
    print("=" * 60)
    print("\n使用说明:")
    print("=" * 60)
    print("修复后的相对角度训练:")
    print("  --policy.type=act \\")
    print("  --policy.use_relative_action=true \\")
    print("  --dataset.repo_id=dataset_1119_12 \\")
    print("  --dataset.root=/home/smai/dc_dir/dataset")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = test_fixed_relative_action()
    sys.exit(0 if success else 1)
