```sh
rm -rf /home/deepansh/drive2/humanoid_ws/src/lerobot_dir/unitree_IL_lerobot_synced/unitree_lerobot/lerobot/outputs_act && \
cd /home/deepansh/drive2/humanoid_ws/src/lerobot_dir/unitree_IL_lerobot_synced/unitree_lerobot/lerobot && \
HF_LEROBOT_HOME=/home/deepansh/drive2/humanoid_ws/src/lerobot_dir/unitree_IL_lerobot_synced/lerobot_data \
python -s src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=deepansh-methdai/single_arm_dual_cam \
    --policy.type=act \
    --output_dir=outputs_act/act_single_arm_dual_cam \
    --job_name=act_single_arm_dual_cam \
    --policy.device=cuda \
    --wandb.enable=true \
    --policy.repo_id=deepansh-methdai/act_policy \
    --wandb.enable=true \
    --steps=5000 \
    --policy.device=cuda \
    --batch_size=32 \
    --save_freq=5000 \
    --log_freq=50 \
    --wandb.notes="Single Arm Dual Camera, ACT, default config testing"

```

```sh
HF_LEROBOT_HOME=/home/deepansh/drive2/humanoid_ws/src/lerobot_dir/unitree_IL_lerobot_synced/lerobot_data \
python -s src/lerobot/scripts/lerobot_train.py \
  --resume=true \
  --config_path=/home/deepansh/drive2/humanoid_ws/src/lerobot_dir/unitree_IL_lerobot_synced/unitree_lerobot/lerobot/outputs_pi05_single_camera_single_gripper/checkpoints/007500/pretrained_model/train_config.json \
  --steps=20000 \
  --batch_size=64
  --save_freq=2500
```
