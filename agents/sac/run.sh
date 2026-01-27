#!/bin/bash

if [ "$1" == "sim" ]; then
    python3 sac_cleanrl.py \
        --render_mode rgb_array \
        --dt 0.12 \
        --env_id SimEmbodiedAnt \
        --learning_starts 2000 \
        --batch_size 256 \
        --task_type forward \
        --model_path ../../sim/assets/embodied_mujoco_ant.xml \
        --capture_video \
        --exp_name  sim_learning_to_walk
fi

# Learn on hardware.
if [ "$1" == "hw" ]; then
    python3 sac_cleanrl.py \
        --render_mode rgb_array \
        --dt 0.12 \
        --seed 1 \
        --env_id HwEmbodiedAnt \
        --hw_config ../../embodied_ant_env/ant1.json \
        --learning_starts 2000 \
        --task_type forward \
        --exp_name ant_forward \
        --weights_path runs/sim_learning_to_walk_with_video_forward_20260122-112812_seed_1/weights_and_args/ \
        --eval True
        # --capture_video \
fi