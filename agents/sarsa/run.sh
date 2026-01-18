#!/bin/bash

if [ "$1" == "sim" ]; then
    python3 sarsa.py \
        --render_mode rgb_array \
        --dt 0.12 \
        --env_id SimEmbodiedAnt \
        --capture_video \
        --exp_name sarsa_ant_forward
fi

# learn on hardware
if [ "$1" == "hw" ]; then
    python3 sac_cleanrl.py \
        --render_mode rgb_array \
        --dt 0.12 \
        --seed 1 \
        --env_id HwEmbodiedAnt \
        --hw_config ../../embodied_ant_env/ant12.json \
        --task_type forward \
        --exp_name sarsa_ant_forward
fi