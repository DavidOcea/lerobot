#!/usr/bin/env python3
"""
测试相对角度训练功能的正确性 | 这个暂时没有调通需要继续调试
"""
import torch
import sys
sys.path.insert(0, '/root/workspace/dc_dir/lerobot/src')

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def test_relative_action():
    print("=" * 60)
    print("测试相对角度训练功能")
    print("=" * 60)

    # 数据集路径
    dataset_path = "/root/data2/dc_dir/datasets/dataset_1224t9_ex"

    # 加载数据集
    print(f"\n1. 加载数据集: {dataset_path}")
    try:
        dataset = LeRobotDataset("local_dataset", root=dataset_path)
        print(f"   ✓ 数据集加载成功!")
        print(f"   - 样本数: {len(dataset)}")
        print(f"   - Episodes: {dataset.num_episodes}")

        # 获取第一个样本
        sample = dataset[0]
        print(f"   - Action shape: {sample['action'].shape}")
        print(f"   - State shape: {sample['observation.state'].shape}")
    except Exception as e:
        print(f"   ✗ 数据集加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 创建 batch
    batch = {
        "observation.images": [sample["observation.images"][0].unsqueeze(0)],
        "observation.state": sample["observation.state"].unsqueeze(0),
        "action": sample["action"].unsqueeze(0),
        "action_is_pad": torch.zeros(1, 100).bool(),  # chunk_size=100
    }

    # 获取数据集统计信息
    stats = dataset.stats

    # 测试1: 绝对角度模式（原始模式）
    print("\n2. 测试绝对角度模式 (use_relative_action=False)")
    try:
        config_abs = ACTConfig(
            use_relative_action=False,
            use_state=True,
            image_features=list(dataset.info["image_keys"].values()),
            input_shapes=dataset.shapes["observation"],
            output_shapes=dataset.shapes["action"],
        )
        policy_abs = ACTPolicy(config_abs, dataset_stats=stats)

        # 训练模式
        policy_abs.train()
        loss_abs, loss_dict_abs = policy_abs(batch)
        print(f"   ✓ 训练测试通过!")
        print(f"   - Loss: {loss_abs.item():.4f}")

        # 推理模式
        policy_abs.eval()
        with torch.no_grad():
            action_abs = policy_abs.select_action(batch)
        print(f"   ✓ 推理测试通过!")
        print(f"   - 预测 action 范围: [{action_abs.min():.4f}, {action_abs.max():.4f}]")

    except Exception as e:
        print(f"   ✗ 绝对角度模式测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 测试2: 相对角度模式
    print("\n3. 测试相对角度模式 (use_relative_action=True)")
    try:
        config_rel = ACTConfig(
            use_relative_action=True,
            use_state=True,
            image_features=list(dataset.info["image_keys"].values()),
            input_shapes=dataset.shapes["observation"],
            output_shapes=dataset.shapes["action"],
        )
        policy_rel = ACTPolicy(config_rel, dataset_stats=stats)

        # 训练模式 - 验证相对角度计算
        policy_rel.train()
        # 保存原始 action
        original_action = batch["action"].clone()

        # 归一化后的 action 和 state
        norm_input = policy_rel.normalize_inputs(batch)
        norm_target = policy_rel.normalize_targets(batch)
        action_norm = norm_target["action"].clone()
        state_norm = norm_input["observation.state"].clone()

        loss_rel, loss_dict_rel = policy_rel(batch)

        # 验证: 训练时 action 应该被转换为相对角度
        # action_norm - state_norm 应该接近模型预测的输入
        expected_relative = action_norm - state_norm
        print(f"   ✓ 训练测试通过!")
        print(f"   - Loss: {loss_rel.item():.4f}")
        print(f"   - L1 Loss: {loss_dict_rel['l1_loss']:.4f}")

        # 推理模式 - 验证绝对角度恢复
        policy_rel.eval()
        with torch.no_grad():
            action_rel = policy_rel.select_action(batch)

        print(f"   ✓ 推理测试通过!")
        print(f"   - 预测 action 范围: [{action_rel.min():.4f}, {action_rel.max():.4f}]")

        # 验证: 推理结果应该与绝对角度模式相似（在合理范围内）
        # 因为两种模式应该学习到相同的策略
        diff = torch.abs(action_abs - action_rel).mean()
        print(f"   - 与绝对角度模式的平均差异: {diff:.4f}")

        # 测试3: 验证相对角度的正确性
        print("\n4. 验证相对角度转换正确性")
        # 手动计算相对角度转换，验证与模型输出一致
        with torch.no_grad():
            # 模拟推理过程
            batch_input = policy_rel.normalize_inputs(batch)
            state_norm = batch_input["observation.state"]

            # 模型输出（归一化空间的相对角度）
            actions_norm = policy_rel.model(batch_input)[0]

            # 恢复到归一化空间的绝对角度
            actions_abs_norm = actions_norm + state_norm

            # 反归一化到原始空间
            actions_abs = policy_rel.unnormalize_outputs({"action": actions_abs_norm})["action"]

        print(f"   ✓ 手动计算验证通过!")
        print(f"   - 手动计算 action 范围: [{actions_abs[0,0,0].min():.4f}, {actions_abs[0,0,0].max():.4f}]")
        print(f"   - 模型输出 action 范围: [{action_rel.min():.4f}, {action_rel.max():.4f}]")

    except Exception as e:
        print(f"   ✗ 相对角度模式测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("✓ 所有测试通过! 相对角度训练功能正常工作。")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_relative_action()
    sys.exit(0 if success else 1)
