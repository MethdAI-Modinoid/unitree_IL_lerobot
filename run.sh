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

# KEY="single_arm_dual_cam"
# cd /home/deepansh/drive2/humanoid_ws/src/lerobot_dir/unitree_IL_lerobot_synced/unitree_lerobot/lerobot && \
# rm -rf /home/deepansh/drive2/humanoid_ws/src/lerobot_dir/unitree_IL_lerobot_synced/unitree_lerobot/lerobot/outputs_pi05_single_arm_dual_cam && \
# HF_LEROBOT_HOME=/home/deepansh/drive2/humanoid_ws/src/lerobot_dir/unitree_IL_lerobot_synced/lerobot_data \
#  python -s src/lerobot/scripts/lerobot_train.py \
#     --dataset.repo_id=deepansh-methdai/single_arm_dual_cam \
#     --policy.type=pi05 \
#     --output_dir=./outputs_pi05_single_arm_dual_cam/ \
#     --job_name=pi05_training \
#     --policy.repo_id=deepansh-methdai/single_arm_dual_cam \
#     --policy.pretrained_path=lerobot/pi05_base \
#     --policy.compile_model=true \
#     --policy.compile_mode=reduce-overhead \
#     --policy.gradient_checkpointing=true \
#     --policy.freeze_vision_encoder=true \
#     --policy.optimizer_lr=2.5e-5 \
#     --wandb.enable=true \
#     --policy.dtype=bfloat16 \
#     --steps=20000 \
#     --policy.device=cuda \
#     --batch_size=32 \
#     --save_freq=2000 \
#     --log_freq=50 \
#     --wandb.notes="Single Arm Dual Camera, freezed vision encoder, 3B trainable parameters, 2.5e-5 learning rate, 64 batch size, 20k steps"

## RESUME TRAINING FOR PI05
cd /home/deepansh/drive2/humanoid_ws/src/lerobot_dir/unitree_IL_lerobot_synced/unitree_lerobot/lerobot && \
HF_LEROBOT_HOME=/home/deepansh/drive2/humanoid_ws/src/lerobot_dir/unitree_IL_lerobot_synced/lerobot_data \
python -s src/lerobot/scripts/lerobot_train.py \
  --resume=true \
  --config_path=/home/deepansh/drive2/humanoid_ws/src/lerobot_dir/unitree_IL_lerobot_synced/unitree_lerobot/lerobot/outputs_pi05_single_arm_dual_cam/checkpoints/020000/pretrained_model/train_config.json \
  --steps=40000 \
  --save_freq=4000


# ## INFERENCE
# python -s unitree_lerobot/eval_robot/eval_g1.py  \
#     --policy.path=./unitree_lerobot/lerobot/outputs_pi05_single_arm_dual_cam/checkpoints/020000/pretrained_model \
#     --repo_id=deepansh-methdai/single_arm_dual_cam \
#     --root="" \
#     --frequency=30 \
#     --arm="G1_29" \
#     --ee="dex3" \
#     --send_real_robot=true \
#     # --policy.n_action_steps=30 \

# ## OPEN LOOP EVAL
# python -s unitree_lerobot/eval_robot/eval_g1_dataset.py  \
#     --policy.path=./unitree_lerobot/lerobot/outputs_pi05_single_arm_dual_cam/checkpoints/020000/pretrained_model \
#     --repo_id=deepansh-methdai/single_arm_dual_cam \
#     --root="" \
#     --episodes=10 \
#     --frequency=30 \
#     --arm="G1_29" \
#     --ee="dex3" \
#     --visualization=false \
#     --send_real_robot=false

# ## SYNTHETIC OPEN LOOP EVAL
# python -s unitree_lerobot/eval_robot/eval_g1_dataset_synthetic.py \
#     --policy.path=./unitree_lerobot/lerobot/outputs_pi05_single_arm_dual_cam/checkpoints/020000/pretrained_model \
#     --repo_id=deepansh-methdai/single_arm_dual_cam \
#     --root="" --episodes=10 --frequency=30 \
#     --arm="G1_29" --ee="dex3" \
#     --visualization=false --send_real_robot=false \
#     --frame_mode=white