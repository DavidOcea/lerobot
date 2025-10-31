

#act train
CUDA_VISIBLE_DEVICES=1 nohup python -m lerobot.scripts.train \
  --policy.type=act \
  --dataset.root=/home/smai/dc_dir/dataset/dataset_1017a18 \
  --dataset.repo_id=dataset_1017a18 \
  --batch_size=16 \
  --steps=200000 \
  --save_freq=50000 \
  --output_dir=outputs/train/act_1025_2 \
  --job_name=act_1025_2 \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.push_to_hub=false  >  1025_act_2.log 2>&1 &

# pretrain 
CUDA_VISIBLE_DEVICES=1 nohup python -m lerobot.scripts.train \
  --dataset.root=/home/smai/dc_dir/dataset/dataset_1017a18 \
  --dataset.repo_id=dataset_1017a18 \
  --policy.path=/home/smai/dc_dir/lerobot_0901_pybullet/outputs/train/act_1023_2/checkpoints/100000/pretrained_model \
  --batch_size=16 \
  --steps=100000 \
  --save_freq=50000 \
  --output_dir=outputs/train/act_1028_2 \
  --job_name=act_1028_2 \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.push_to_hub=false  >  1028_act_2.log 2>&1 &

smolvla | diffusion
 可能需要加上 export TOKENIZERS_PARALLELISM=true ｜ false 
#smolvla train
CUDA_VISIBLE_DEVICES=1 nohup python -m lerobot.scripts.train \
  --policy.path=/home/smai/dc_dir/models/smolvla_0925_1/pretrained_model \
  --dataset.root=/home/smai/dc_dir/dataset/dataset_1014_allv \
  --dataset.repo_id=dataset_1014_allv \
  --batch_size=64 \
  --steps=200000 \
  --save_freq=50000 \
  --output_dir=outputs/train/smla_1021_2 \
  --job_name=smla_1021_2 \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.push_to_hub=false  >  1021_smla_2.log 2>&1 &

--batch_size=64

#diffusion train
CUDA_VISIBLE_DEVICES=1 nohup python -m lerobot.scripts.train \
  --policy.type=diffusion \
  --dataset.root=/home/smai/dc_dir/dataset/dataset_1014_allv \
  --dataset.repo_id=dataset_1014_allv \
  --batch_size=64 \
  --steps=200000 \
  --save_freq=50000 \
  --output_dir=outputs/train/dp_1018_2 \
  --job_name=dp_1018_2 \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.push_to_hub=false  >  1018_dp_2.log 2>&1 &

  --batch_size=64


  #pi0.5 train
CUDA_VISIBLE_DEVICES=0 nohup python -m lerobot.scripts.train \
  --policy.type=pi05 \
  --dataset.root=/home/smai/dc_dir/dataset/dataset_1014_allv \
  --dataset.repo_id=dataset_1014_allv \
  --batch_size=16 \
  --steps=250000 \
  --save_freq=50000 \
  --output_dir=outputs/train/pi0_1016_1 \
  --job_name=pi0_1016_1 \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.push_to_hub=false  >  1016_pi0_1.log 2>&1 &

  --batch_size=32
  --policy.train_expert_only=true  训练部分

  --policy.path=/home/smai/dc_dir/models/pi0fast_base \

  --save_freq
  --policy.optimizer_lr=1e-06 \
  --policy.optimizer_lr_backbone=1e-06 \

# pybullet
python -m lerobot.record \
    --robot.type=sim_robot \
    --policy.path=/home/smai/dc_dir/lerobot_0901_pybullet/outputs/train/act_1026_4/checkpoints/last/pretrained_model \
    --dataset.repo_id=supredata/eval_dataset_0902 \
    --dataset.single_task="Grasp the workpiece and put it in the appropriate position." \
    --dataset.episode_time_s=150 \
    --dataset.num_episodes=1 \
    --dataset.reset_time_s=10 \
    --dataset.push_to_hub=False