```py
cd /home/deepansh/drive2/humanoid_ws/src/lerobot_dir/unitree_IL_lerobot_synced/unitree_lerobot/lerobot && \
HF_LEROBOT_HOME=/home/deepansh/drive2/humanoid_ws/src/lerobot_dir/unitree_IL_lerobot_synced/lerobot_data \
python -s src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=deepansh-methdai/single_arm_dual_cam \
    --policy.type=act \
    --output_dir=outputs_act/act_single_arm_dual_cam \
    --job_name=act_single_arm_dual_cam \
    --policy.device=cuda \
    --wandb.enable=true \
    --policy.repo_id=deepansh-methdai/act_policy
    --policy.compile_model=true \
    --policy.compile_mode=reduce-overhead \
    --policy.gradient_checkpointing=true \
    --wandb.enable=true \
    --policy.dtype=bfloat16 \
    --steps=5000 \
    --policy.device=cuda \
    --batch_size=32 \
    --save_freq=5000 \
    --log_freq=50 \
    --wandb.notes="Single Arm Dual Camera, ACT, default config testing"

```