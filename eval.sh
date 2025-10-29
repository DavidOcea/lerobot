lerobot-eval \
    --policy.path=/home/zzj/dc_space/models/act_0828_10w/pretrained_model \
    --env.type=pybullet \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --use_amp=false \
    --device=cuda