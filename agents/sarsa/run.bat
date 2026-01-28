@echo off

REM Learn on simulation
if "%1"=="sim" (
    python sarsa.py ^
        --render_mode rgb_array ^
        --dt 0.12 ^
        --env_id SimEmbodiedAnt ^
        --exp_name sarsa_ant_forward
        REM --capture_video
)

REM Learn on hardware
if "%1"=="hw" (
    python sac_cleanrl.py ^
        --render_mode rgb_array ^
        --dt 0.12 ^
        --seed 1 ^
        --env_id HwEmbodiedAnt ^
        --hw_config ../../embodied_ant_env/ant12.json ^
        --task_type forward ^
        --exp_name sarsa_ant_forward
)
