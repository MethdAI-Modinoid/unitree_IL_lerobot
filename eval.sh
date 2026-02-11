## INFERENCE
python -s unitree_lerobot/eval_robot/eval_g1.py  \
    --policy.path=./unitree_lerobot/lerobot/outputs_pi05_single_arm_dual_cam/checkpoints/020000/pretrained_model \
    --repo_id=deepansh-methdai/single_arm_dual_cam \
    --root="" \
    --frequency=30 \
    --arm="G1_29" \
    --ee="dex3" \
    --send_real_robot=true \
    # --custom_task="Place the starfruit in brown box" \
    # --motion=True \
    # --policy.n_action_steps=30 \