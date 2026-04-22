

#act train
CUDA_VISIBLE_DEVICES=1 nohup python -m lerobot.scripts.train \
  --policy.type=act \
  --dataset.root=/root/data2/dc_dir/datasets/dataset_1224t5_ex \
  --dataset.repo_id=dataset_1224t5_ex \
  --batch_size=32 \
  --steps=200000 \
  --save_freq=40000 \
  --output_dir=outputs/train/act_1225_0 \
  --job_name=act_1225_0 \
  --policy.device=cuda \
  --wandb.enable=false \
  --dataset.customer_transforms=True \
  --dataset.only_head_transforms=True \
  --save_heat_map=True \
  --policy.push_to_hub=false  >  1225_act_0.log 2>&1 &

--dataset.only_head_transforms=True \
# pretrain 
CUDA_VISIBLE_DEVICES=2 nohup python -m lerobot.scripts.train \
  --dataset.root=/root/data2/dc_dir/datasets/dataset_1119t27eg \
  --dataset.repo_id=dataset_1119t27_eg \
  --policy.path=/root/data2/dc_dir/models/train/act_1106_1/checkpoints/last/pretrained_model \
  --batch_size=32 \
  --steps=20000 \
  --save_freq=5000 \
  --output_dir=outputs/train/act_1210_2 \
  --job_name=act_1210_2 \
  --policy.device=cuda \
  --wandb.enable=false \
  --dataset.customer_transforms=True \
  --dataset.only_head_transforms=True \
  --save_heat_map=True \
  --policy.push_to_hub=false  >  1210_act_2.log 2>&1 &

smolvla | diffusion
#smolvla train
 可能需要加上 下面这句话防止日志出现打乱码
export TOKENIZERS_PARALLELISM=true ｜ false   
CUDA_VISIBLE_DEVICES=5 nohup python -m lerobot.scripts.train \
  --policy.path=/root/data2/dc_dir/models/smolvla_base \
  --dataset.root=/root/data2/dc_dir/datasets/dataset_1119t1216_eg \
  --dataset.repo_id=dataset_1119t1216_eg \
  --batch_size=64 \
  --steps=200000 \
  --save_freq=40000 \
  --output_dir=outputs/train/smla_1223_3 \
  --job_name=smla_1223_3 \
  --policy.device=cuda \
  --wandb.enable=false \
  --dataset.customer_transforms=True \
  --dataset.only_head_transforms=True \
  --policy.push_to_hub=false  >  1223_smla_3.log 2>&1 &

  --dataset.customer_transforms=True \
  --dataset.only_head_transforms=True \
--batch_size=64

#diffusion train
CUDA_VISIBLE_DEVICES=0 nohup python -m lerobot.scripts.train \
  --policy.type=diffusion \
  --dataset.root=/root/data2/dc_dir/datasets/dataset_1119t1216_eg \
  --dataset.repo_id=dataset_1119t1216_eg \
  --batch_size=64 \
  --steps=200000 \
  --save_freq=40000 \
  --output_dir=outputs/train/dp_1222_0 \
  --job_name=dp_1222_0 \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.push_to_hub=false  >  1222_dp_0.log 2>&1 &

  --batch_size=64 \
  --dataset.customer_transforms=True \
  --dataset.only_head_transforms=True \


  #pi0 train 
CUDA_VISIBLE_DEVICES=3 nohup python -m lerobot.scripts.train \
  --policy.path=/root/data2/dc_dir/models/pi0 \
  --dataset.root=/root/data2/dc_dir/datasets/dataset_1119t27eg \
  --dataset.repo_id=dataset_1119t27eg \
  --batch_size=16 \
  --steps=200000 \
  --save_freq=50000 \
  --output_dir=outputs/train/pi0_1204_1 \
  --job_name=pi0_1204_1 \
  --policy.device=cuda \
  --wandb.enable=false \
  --dataset.customer_transforms=True \
  --policy.push_to_hub=false  >  1204_pi0_1.log 2>&1 &

  #pi0.5 train 
CUDA_VISIBLE_DEVICES=5 nohup python -m lerobot.scripts.train \
  --dataset.root=/root/workspace/dc_dir/datasets/dataset_1017t27 \
  --dataset.repo_id=dataset_1017t27 \
  --policy.path=/root/workspace/dc_dir/models/pi05_base \
  --batch_size=2 \
  --steps=200000 \
  --save_freq=50000 \
  --output_dir=outputs/train/pi05_1112_test \
  --job_name=pi05_1112_1 \
  --policy.device=cuda \
  --wandb.enable=false \
  --save_heat_map=True \
  --policy.push_to_hub=false  >  1112_pi05_1.log 2>&1 &

    --policy.pretrained_path=/root/workspace/dc_dir/models/pi05_base \ 
    --policy.compile_model=true \    #采用符号执行方式解析并运行代码，将完整计算过程转化为计算图（IR）。通过在计算图中尽可能地将多个小算子融合为单一 kernel，以显著降低显存读写、GPU kernel 启动及 CPU 与 GPU 同步所带来的性能开销。其中，融合后的算子由 Triton 实现。
    --policy.gradient_checkpointing=true \  #‌梯度检查点（Gradient Checkpointing）是一种通过牺牲计算时间换取显存优化的技术，适用于训练大模型时显存不足的情况‌


  --batch_size=32
  --policy.train_expert_only=true  训练部分

  --policy.path=/home/smai/dc_dir/models/pi0fast_base \

  --save_freq
  --policy.optimizer_lr=1e-06 \
  --policy.optimizer_lr_backbone=1e-06 \

# pybullet
python -m lerobot.record \
    --robot.type=sim_robot \
    --policy.path=/root/workspace/dc_dir/lerobot/outputs/train/dp_1025_1/checkpoints/last/pretrained_model \
    --dataset.repo_id=supredata/eval_dataset_0902 \
    --dataset.single_task="Grasp the workpiece and put it in the appropriate position." \
    --dataset.episode_time_s=150 \
    --dataset.num_episodes=1 \
    --dataset.reset_time_s=10 \
    --dataset.push_to_hub=False

TORCHDYNAMO_VERBOSE=1 可以在运行脚本出错是详细数据日志


# ctrl-wold
CUDA_VISIBLE_DEVICES=1 python scripts/rollout_replay_traj.py  \
  --dataset_root_path dataset_example \
  --dataset_meta_info_path dataset_meta_info \
  --dataset_names droid_subset \
  --svd_model_path /root/workspace/stable-video-diffusion-img2vid \
  --clip_model_path /root/workspace/clip-vit-base-patch32 \
  --ckpt_path /root/workspace/Ctrl-World
  