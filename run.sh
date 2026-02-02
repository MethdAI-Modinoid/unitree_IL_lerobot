# conda deactivate
# conda activate unitree_lerobot_synced
# python -s unitree_lerobot/eval_robot/eval_g1.py  \
#     --policy.path=/home/deepansh/drive2/humanoid_ws/src/lerobot_dir/unitree_IL_lerobot_synced/unitree_lerobot/lerobot/outputs_octopus/pi05_training/checkpoints/008000/pretrained_model \
#     --repo_id=deepansh-methdai/octopus \
#     --root="" \
#     --episodes=0 \
#     --frequency=30 \
#     --arm="G1_29" \
#     --ee="dex3" \
#     --send_real_robot=true \
#     --policy.n_action_steps=30 \
#     --custom_task="Pick up the blue toy and place it in the brown box"

cd /home/deepansh/drive2/humanoid_ws/src/lerobot_dir/unitree_IL_lerobot_synced/unitree_lerobot/lerobot && \
rm -rf /home/deepansh/drive2/humanoid_ws/src/lerobot_dir/unitree_IL_lerobot_synced/unitree_lerobot/lerobot/outputs_test && \
HF_LEROBOT_HOME=/home/deepansh/drive2/humanoid_ws/src/lerobot_dir/unitree_IL_lerobot_synced/lerobot_data \
 python -s src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=deepansh-methdai/octopus \
    --policy.type=pi05 \
    --output_dir=./outputs_test/pi05_training \
    --job_name=pi05_training \
    --policy.repo_id=deepansh-methdai/pi05_octopus \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --policy.optimizer_lr=2.5e-4 \
    --wandb.enable=true \
    --policy.dtype=bfloat16 \
    --steps=100 \
    --policy.device=cuda \
    --batch_size=4 \
    --save_freq=50 \
    --log_freq=50

