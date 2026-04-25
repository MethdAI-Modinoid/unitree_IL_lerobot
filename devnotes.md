export HF_HUB_DISABLE_XET=1

python unitree_lerobot/utils/sort_and_rename_folders.py \
        --data_dir datasets/three_camera/

```py
HF_LEROBOT_HOME=./lerobot_data python unitree_lerobot/utils/convert_unitree_json_to_lerobot.py \
    --raw-dir datasets/three_camera \
    --repo-id deepansh-methdai/three_camera \
    --robot_type Unitree_G1_Dex3 \
    --push_to_hub

python unitree_lerobot/utils/convert_unitree_json_to_lerobot.py \
    --raw-dir datasets/three_objects \
    --repo-id deepansh-methdai/octopus \
    --robot_type Unitree_G1_Dex3 \
    --push_to_hub

python src/lerobot/datasets/v30/augment_dataset_quantile_stats.py \
    --repo-id=deepansh-methdai/apple_box20

hf upload --folder-path /mnt/drive2/.cache/huggingface/lerobot/deepansh-methdai/apple_box deepansh-methdai/apple_box
huggingface-cli upload deepansh-methdai/apple_box /mnt/drive2/.cache/huggingface/lerobot/deepansh-methdai/apple_box .
```

### DATASET DOF CONVERSION

# Convert dataset to different DOF configurations
# DOF modes: dual_gripper (16D), dual_full (28D), single_gripper (8D), single_full (14D)
# Single arm modes use the RIGHT arm only

# Dual arm with gripper (28D -> 16D)
python -s unitree_lerobot/utils/convert_hands_to_gripper.py \
    --repo-id deepansh-methdai/three_camera \
    --output-repo-id deepansh-methdai/three_camera_gripper \
    --dof-mode dual_gripper \
    --root ./lerobot_data/deepansh-methdai/three_camera

# Single arm with gripper (28D -> 8D), remove wrist cameras
python -s unitree_lerobot/utils/convert_hands_to_gripper.py \
    --repo-id deepansh-methdai/three_camera \
    --output-repo-id deepansh-methdai/single_arm_dual_cam \
    --dof-mode single_gripper \
    --remove-cameras observation.images.cam_left_wrist \
    --root ./lerobot_data/deepansh-methdai/three_camera \
    --push-to-hub

# Single arm full hand (28D -> 14D)
python -s unitree_lerobot/utils/convert_hands_to_gripper.py \
    --repo-id deepansh-methdai/three_camera \
    --output-repo-id deepansh-methdai/three_camera_single_full \
    --dof-mode single_full \
    --root ./lerobot_data/deepansh-methdai/three_camera

# Dry run to preview conversion
python -s unitree_lerobot/utils/convert_hands_to_gripper.py \
    --repo-id deepansh-methdai/three_camera \
    --output-repo-id deepansh-methdai/test_output \
    --dof-mode dual_gripper \
    --remove-cameras observation.images.cam_left_high,observation.images.cam_left_wrist \
    --root ./lerobot_data/deepansh-methdai/three_camera \
    --dry-run

### TRAINING

rm -rf /home/deepansh/drive2/humanoid_ws/src/lerobot_dir/unitree_IL_lerobot_synced/unitree_lerobot/lerobot/outputs && \
HF_LEROBOT_HOME=/home/deepansh/drive2/humanoid_ws/src/lerobot_dir/unitree_IL_lerobot_synced/lerobot_data \
 python -s src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=deepansh-methdai/apple_box \
    --policy.type=pi05 \
    --output_dir=./outputs/pi05_training \
    --job_name=pi05_training \
    --policy.repo_id=deepansh-methdai/pi05_test \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --policy.freeze_vision_encoder=true \
    --policy.train_expert_only=true \
    --wandb.enable=true \
    --policy.dtype=bfloat16 \
    --steps=10000 \
    --policy.device=cuda \
    --batch_size=256 \
    --save_freq=1000 \
    --log_freq=50

### RESUME TRAINING


HF_LEROBOT_HOME=/home/deepansh/drive2/humanoid_ws/src/lerobot_dir/unitree_IL_lerobot_synced/lerobot_data \
python -s src/lerobot/scripts/lerobot_train.py \
  --resume=true \
  --config_path=/home/deepansh/drive2/humanoid_ws/src/lerobot_dir/unitree_IL_lerobot_synced/unitree_lerobot/lerobot/outputs_pi05_single_camera_single_gripper/checkpoints/007500/pretrained_model/train_config.json \
  --steps=20000 \
  --batch_size=64
  --save_freq=2500


### OPEN LOOP EVAL

python -s unitree_lerobot/eval_robot/eval_g1_dataset.py  \
    --policy.path=/home/deepansh/drive2/humanoid_ws/src/lerobot_dir/unitree_IL_lerobot_synced/unitree_lerobot/lerobot/outputs_octopus/pi05_training/checkpoints/008000/pretrained_model \
    --repo_id=deepansh-methdai/octopus \
    --root="" \
    --episodes=10 \
    --frequency=30 \
    --arm="G1_29" \
    --ee="dex3" \
    --visualization=false \
    --send_real_robot=false

python -s unitree_lerobot/eval_robot/eval_g1_dataset.py  \
    --policy.path=/home/deepansh/drive2/humanoid_ws/src/lerobot_dir/unitree_IL_lerobot_synced/unitree_lerobot/lerobot/outputs_pi05_single_camera_single_gripper/checkpoints/020000/pretrained_model \
    --repo_id=deepansh-methdai/single_camera_single_gripper \
    --root="" \
    --episodes=10 \
    --frequency=30 \
    --arm="G1_29" \
    --ee="dex3" \
    --visualization=false \
    --send_real_robot=false

### REPLAY

python -s unitree_lerobot/eval_robot/replay_robot.py \
    --repo_id=deepansh-methdai/single_camera_single_gripper \
    --root="" \
    --episodes=0 \
    --frequency=30 \
    --arm="G1_29" \
    --ee="dex3" \
    --send_real_robot=true \
    --visualization=false

### REPLAY (GRIPPER DATASET)
# For datasets converted to gripper representation (16D state/action)
# The gripper converter automatically expands gripper values to full hand poses

python -s unitree_lerobot/eval_robot/replay_robot.py \
    --repo_id=deepansh-methdai/three_camera_gripper \
    --root="" \
    --episodes=0 \
    --frequency=30 \
    --arm="G1_29" \
    --ee="dex3" \
    --send_real_robot=true \
    --visualization=false


### INFERENCE

python unitree_lerobot/eval_robot/eval_g1.py  \
    --policy.path=/home/deepansh/drive2/humanoid_ws/src/lerobot_dir/unitree_IL_lerobot_synced/unitree_lerobot/lerobot/outputs/pi05_training/checkpoints/002000/pretrained_model \
    --repo_id=deepansh-methdai/apple_box \
    --root="" \
    --episodes=0 \
    --frequency=30 \
    --arm="G1_29" \
    --ee="dex3" \
    --rename_map='{"observation.images.cam_left_high": "observation.images.camera"}'

python -s unitree_lerobot/eval_robot/eval_g1.py  \
    --policy.path=/home/deepansh/drive2/humanoid_ws/src/lerobot_dir/unitree_IL_lerobot_synced/unitree_lerobot/lerobot/outputs/pi05_training/checkpoints/011000/pretrained_model \
    --repo_id=deepansh-methdai/apple_box \
    --root="" \
    --episodes=0 \
    --frequency=10 \
    --arm="G1_29" \
    --ee="dex3" \
    --rename_map='{"observation.images.cam_left_high": "observation.images.camera"}' \
    --send_real_robot=true
    --custom_task="pick up the apple"

python -s unitree_lerobot/eval_robot/eval_g1.py  \
    --policy.path=/home/deepansh/drive2/humanoid_ws/src/lerobot_dir/unitree_IL_lerobot_synced/unitree_lerobot/lerobot/outputs_pi05_single_camera_single_gripper/checkpoints/020000/pretrained_model \
    --repo_id=deepansh-methdai/single_camera_single_gripper \
    --root="" \
    --frequency=30 \
    --arm="G1_29" \
    --ee="dex3" \
    --send_real_robot=true \
    --custom_task="pick up the cube and place it in the brown box"


### SLATE

rm -rf /home/deepansh/drive2/humanoid_ws/src/lerobot_dir/unitree_IL_lerobot_synced/unitree_lerobot/lerobot/outputs_octopus && \
HF_LEROBOT_HOME=/home/deepansh/drive2/humanoid_ws/src/lerobot_dir/unitree_IL_lerobot_synced/lerobot_data \
 python -s src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=deepansh-methdai/octopus \
    --policy.type=pi05 \
    --output_dir=./outputs_octopus/pi05_training \
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
    --batch_size=32 \
    --save_freq=50 \
    --log_freq=50


# VERSION PINNING:

requirements_freezed_synced.txt


## TESTS
Implement RTC (most important)



#####################################################################
## GRIPPER STATE OPEN/CLOSE:

RIGHT_HAND_OPEN = np.array([
        0.0672,  # thumb0
        0.5666,  # thumb1
        -0.0679,  # thumb2
        -0.0211,  # middle0
        -0.0112,  # middle1
        -0.0162,  # index0
        -0.0283,  # index1
    ])

RIGHT_HAND_CLOSE = np.array([
        -0.03833567723631859,  # thumb0
       -0.36572766304016113,  # thumb1
        -0.024161333218216896,  # thumb2
         0.9473425149917603,  # middle0
        -0.044050849974155426,  # middle1
        0.9455186128616333,  # index0
        -0.06319903582334518,  # index1
    ])

def gripper_state(hand_state: np.ndarray, margin=0.1) -> int:
        d_open = np.linalg.norm(hand_state - RIGHT_HAND_OPEN)
        d_close = np.linalg.norm(hand_state - RIGHT_HAND_CLOSE)
        margin = 0.1
        if d_open + margin < d_close:
            return 0
            # print("OPEN")
        elif d_close + margin < d_open:
            return 1
            # print("CLOSE")
        else:
            return 0 
            # print("OPEN")

#########################################################################

#########################################################################
PROMPT:

use existing lerobot dataset tools and utils to modify a given unitree g1 dataset:
1) read the observation.state data
2) there will be 28 keys
3) read the last 14 keys for left hand and right hand states
4) estimate the opennes/closeness of the left and right hands, using sample gripper control
5) remove the old lerobot feature observation.state
6) Add the new observation.state, with the first 14 left and right arm keys, and then 2 left and right gripper states, totalling a lenght 16 vector.
7) repeat the same steps 1 to 5 with action feature. read the last 14 keys, estimate gripper states, remove old action and add new action feature.
8) Add a flag to perform lerobot push, using the datasets class. take suggestion from the unitree json to lerobot conversion script
9)  verify everything before committing the changes
10) use dataset tools functions, this way, you won't need to change the original dataset, you would be aple to create a new repo, appended modified at the end

points to note:
observe the info.json file
observe the dataset_tools module to modify dataset
observe the modify_features module
use the relevant add feature and remove feature function
keep lerobot dataset schema in mind

environment config:
before starting the debugging in terminal, run conda deactivate
then run conda activate unitree_lerobot_synced
Use the -s flag when running python scripts



#########################################################################


# Dataset conversion from hands to gripper

```py
python -s unitree_lerobot/utils/convert_hands_to_gripper.py --repo-id deepansh-methdai/three_camera --root ./lerobot_data/deepansh-methdai/three_camera --push-to-hub
```

# Dry run mode (verify without modifying):
```py
python -s unitree_lerobot/utils/convert_hands_to_gripper.py --repo-id deepansh-methdai/three_camera --root ./lerobot_data/deepansh-methdai/three_camera --dry-run
```

# Custom output directory:
```py
python -s unitree_lerobot/utils/convert_hands_to_gripper.py --repo-id deepansh-methdai/three_camera --root ./lerobot_data/deepansh-methdai/three_camera --output-dir ./lerobot_data/deepansh-methdai/three_camera_gripper
```


## Remove dataset features
```sh
lerobot-edit-dataset \
    --repo_id deepansh-methdai/three_camera_gripper \
    --operation.type remove_feature \
    --operation.feature_names "['observation.images.cam_left_wrist', 'observation.images.cam_right_wrist']"
```

## SYNTHETIC OPEN LOOP EVAL
python -s unitree_lerobot/eval_robot/eval_g1_dataset_synthetic.py \
    --policy.path=<checkpoint> \
    --repo_id=deepansh-methdai/single_camera_single_gripper \
    --root="" --episodes=10 --frequency=30 \
    --arm="G1_29" --ee="dex3" \
    --visualization=false --send_real_robot=false \
    --frame_mode=random

