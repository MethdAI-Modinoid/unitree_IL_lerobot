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

# cd /home/deepansh/drive2/humanoid_ws/src/lerobot_dir/unitree_IL_lerobot_synced/unitree_lerobot/lerobot && \
# rm -rf /home/deepansh/drive2/humanoid_ws/src/lerobot_dir/unitree_IL_lerobot_synced/unitree_lerobot/lerobot/outputs_pi05_single_arm_dual_cam && \
# HF_LEROBOT_HOME=/home/deepansh/drive2/humanoid_ws/src/lerobot_dir/unitree_IL_lerobot_synced/lerobot_data \
#  python -s src/lerobot/scripts/lerobot_train.py \
#     --dataset.repo_id=deepansh-methdai/single_arm_dual_cam \
#     --policy.type=pi05 \
#     --output_dir=./outputs_pi05_single_arm_dual_cam/ \
#     --job_name=pi05_training \
#     --policy.repo_id=deepansh-methdai/single_arm_dual_cam_new \
#     --policy.pretrained_path=lerobot/pi05_base \
#     --policy.compile_model=true \
#     --policy.compile_mode=reduce-overhead \
#     --policy.gradient_checkpointing=true \
#     --policy.freeze_vision_encoder=true \
#     --policy.optimizer_lr=2.5e-5 \
#     --wandb.enable=true \
#     --policy.dtype=bfloat16 \
#     --steps=30000 \
#     --policy.device=cuda \
#     --batch_size=32 \
#     --save_freq=5000 \
#     --log_freq=50 \
#     --wandb.notes="Single Arm Dual Camera, freezed vision encoder, 3B trainable parameters, 2.5e-5 learning rate, 64 batch size, 30k steps"

# cd ./unitree_lerobot/lerobot && \
# rm -rf outputs_pi05_26_02 && \
# HF_LEROBOT_HOME=./lerobot_data \
#  python -s src/lerobot/scripts/lerobot_train.py \
#     --dataset.repo_id=deepansh-methdai/single_arm_dual_cam \
#     --policy.type=pi05 \
#     --output_dir=./outputs_pi05_26_02/ \
#     --job_name=pi05_training \
#     --policy.repo_id=deepansh-methdai/single_arm_dual_cam_26_02 \
#     --policy.pretrained_path=lerobot/pi05_base \
#     --policy.compile_model=true \
#     --policy.compile_mode=reduce-overhead \
#     --policy.gradient_checkpointing=true \
#     --policy.freeze_vision_encoder=true \
#     --policy.optimizer_lr=2.5e-5 \
#     --wandb.enable=true \
#     --policy.dtype=bfloat16 \
#     --steps=40000 \
#     --policy.device=cuda \
#     --batch_size=32 \
#     --save_freq=2000 \
#     --log_freq=50 \
#     --wandb.notes="Single Arm Dual Camera, freezed vision encoder, 3B trainable parameters, 2.5e-5 learning rate, 64 batch size, 40k steps"

# ## RESUME TRAINING FOR PI05
# HF_LEROBOT_HOME=/home/deepansh/drive2/humanoid_ws/src/lerobot_dir/unitree_IL_lerobot_synced/lerobot_data \
# python -s src/lerobot/scripts/lerobot_train.py \
#   --resume=true \
#   --config_path=/home/deepansh/drive2/humanoid_ws/src/lerobot_dir/unitree_IL_lerobot_synced/unitree_lerobot/lerobot/outputs_pi05_single_camera_single_gripper/checkpoints/007500/pretrained_model/train_config.json \
#   --steps=30000 \
#   --batch_size=32
#   --save_freq=5000


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

export LEROBOT_REPO_DIR=/workspace/unitree_lerobot/lerobot
export HF_LEROBOT_HOME=/workspace/lerobot_data
# export LEROBOT_REPO_DIR=./unitree_lerobot/lerobot
# export HF_LEROBOT_HOME=./lerobot_data

# ## OPEN LOOP EVAL
# python -s unitree_lerobot/eval_robot/eval_g1_dataset_synthetic.py  \
#     --policy.path=outputs_pi/best_config/checkpoints/models--deepansh-methdai--pi05_single_arm_dual_cam_20k/snapshots/b89d7a7dc6b2d5d5955f020e95d5c8f6743c939e/020000/pretrained_model \
#     --repo_id=deepansh-methdai/single_arm_dual_cam \
#     --root="" \
#     --episodes=10 \
#     --frequency=30 \
#     --arm="G1_29" \
#     --ee="dex3" \
#     --visualization=false \
#     --send_real_robot=false \
#     --frame_mode=white

# ## SYNTHETIC OPEN LOOP EVAL
# python -s unitree_lerobot/eval_robot/eval_g1_dataset_synthetic.py \
#     --policy.path=./unitree_lerobot/lerobot/outputs_pi05_single_arm_dual_cam/checkpoints/020000/pretrained_model \
#     --repo_id=deepansh-methdai/single_arm_dual_cam \
#     --root="" --episodes=10 --frequency=30 \
#     --arm="G1_29" --ee="dex3" \
#     --visualization=false --send_real_robot=false \
#     --frame_mode=white

# cd ${LEROBOT_REPO_DIR} && \
# rm -rf ./outputs_pi05_unfrozen_vision_20k_temp && \
# HF_LEROBOT_HOME=${HF_LEROBOT_HOME} \
# python -s src/lerobot/scripts/lerobot_train.py \
#     --dataset.repo_id=deepansh-methdai/single_arm_dual_cam \
#     --policy.type=pi05 \
#     --output_dir=./outputs_pi05_unfrozen_vision_20k_temp/ \
#     --job_name=pi05_unfrozen_vision_temp \
#     --policy.repo_id=deepansh-methdai/pi05_unfrozen_vision_20k_temp \
#     --policy.pretrained_path=lerobot/pi05_base \
#     --policy.compile_model=true \
#     --policy.compile_mode=reduce-overhead \
#     --policy.gradient_checkpointing=true \
#     --policy.freeze_vision_encoder=false \
#     --policy.train_expert_only=false \
#     --policy.optimizer_lr=2.5e-5 \
#     --policy.dtype=bfloat16 \
#     --policy.device=cuda \
#     --steps=6000 \
#     --batch_size=32 \
#     --save_freq=5000 \
#     --log_freq=50 \
#     --wandb.enable=true \
#     --wandb.project=pi05_g1_evaluation_temp \
#     --wandb.run_id=vision_unfrozen_20k_temp \
#     --wandb.notes="Vision encoder ablation: unfrozen vision encoder, full finetune, 20k steps"


# cd ${LEROBOT_REPO_DIR} && \
# rm -rf ./outputs_pi05_primary_30k && \
# HF_LEROBOT_HOME=${HF_LEROBOT_HOME} \
# python -s src/lerobot/scripts/lerobot_train.py \
#     --dataset.repo_id=deepansh-methdai/single_arm_dual_cam \
#     --policy.type=pi05 \
#     --output_dir=./outputs_pi05_primary_30k/ \
#     --job_name=primary_vision_frozen_30k \
#     --policy.repo_id=deepansh-methdai/pi05_primary_vision_frozen_30k \
#     --policy.pretrained_path=lerobot/pi05_base \
#     --policy.compile_model=true \
#     --policy.compile_mode=reduce-overhead \
#     --policy.gradient_checkpointing=true \
#     --policy.freeze_vision_encoder=true \
#     --policy.train_expert_only=false \
#     --policy.optimizer_lr=2.5e-5 \
#     --policy.dtype=bfloat16 \
#     --policy.device=cuda \
#     --steps=30000 \
#     --batch_size=32 \
#     --save_freq=10000 \
#     --log_freq=50 \
#     --wandb.enable=true \
#     --wandb.project=pi05_g1_evaluation \
#     --wandb.run_id=primary_vision_frozen_30k_new \
#     --wandb.notes="Primary checkpoint: frozen vision encoder, full finetune, 30k steps"

cd ${LEROBOT_REPO_DIR} && \
rm -rf ./outputs_pi05_expert_only_20k && \
HF_LEROBOT_HOME=${HF_LEROBOT_HOME} \
python -s src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=deepansh-methdai/single_arm_dual_cam \
    --policy.type=pi05 \
    --output_dir=./outputs_pi05_expert_only_20k/ \
    --job_name=pi05_expert_only \
    --policy.repo_id=deepansh-methdai/pi05_expert_only_20k \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --policy.freeze_vision_encoder=true \
    --policy.train_expert_only=true \
    --policy.optimizer_lr=2.5e-5 \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --steps=20000 \
    --batch_size=32 \
    --save_freq=10000 \
    --log_freq=50 \
    --wandb.enable=true \
    --wandb.project=pi05_g1_evaluation \
    --wandb.run_id=expert_only_frozen_20k \
    --wandb.notes="Expert-only ablation: frozen vision encoder, expert-only training, 20k steps"