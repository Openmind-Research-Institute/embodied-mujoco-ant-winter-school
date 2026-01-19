import os
import io
import sys
import cv2
import time
import json
import pickle
import zipfile
import imageio
import argparse
import datetime
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Custom imports.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../sim')))
from ant_mujoco import AntEnv
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))
from tilecoding import IHT, tiles
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../embodied_ant_env')))
from embodied_ant_env import make_ant_env, ForwardTask
from reward import RewardTracker

np.set_printoptions(precision=4, suppress=True, linewidth=120, threshold=1000)


# Ramp function.
def linear_ramp(start_pos: float, end_pos: float, duration: float):
    num = round(duration / args.dt)
    input_pos_list = np.linspace(start_pos, end_pos, num)
    return input_pos_list

class OptionEnv:
    def __init__(self, env, options, discount=0.99):
        self.env = env
        self.options = options
        self.discount = discount
        self.joint_action = np.zeros(env.action_space.shape[0])

        # Reward tracker.
        self.reward_tracker = RewardTracker(env_dt=args.dt,
                                    env_id=f"run_{args.env_id}",
                                    time_window=120.0,
                                    log_folder=log_dir)
        self.info = None

    def step(self, option_idx: int):
        opt = self.options[option_idx]

        # Populate the joint action trajectory.
        hip_joint_idx = opt['hip_joint_idx']
        knee_joint_idx = opt['knee_joint_idx']

        hip_traj = linear_ramp(self.joint_action[hip_joint_idx], opt['hip_target'], opt['duration'])
        num_steps = len(hip_traj)
        if opt['hip_target'] != self.joint_action[hip_joint_idx]:
            time = np.linspace(0, opt['duration'], num_steps)
            knee_traj = opt['knee_amplitude'] * np.sin(np.pi * time / opt['duration'])
        else:
            # NOTE: This is done to avoid the knee from flopping unnecessarily when the hip is not moving.
            knee_traj = np.full(num_steps, self.joint_action[knee_joint_idx])

        total_reward = 0.0
        gamma_i = 1.0

        for i in range(self.duration_steps(option_idx)):
            self.joint_action[hip_joint_idx] = hip_traj[i]
            self.joint_action[knee_joint_idx] = knee_traj[i]
            obs, reward, terminated, truncated, self.info = self.env.step(self.joint_action)
            original_reward = reward
            reward *= args.reward_scaling

            # Average reward update.
            self.reward_tracker.update(original_reward)
            self.reward_tracker.log()

            total_reward += gamma_i * reward
            gamma_i *= self.discount
            if terminated or truncated:
                return obs, total_reward, terminated, truncated, self.info

        return obs, total_reward, terminated, truncated, self.info

    def reset(self, seed=None):
        self.joint_pos = np.zeros(self.env.action_space.shape[0])
        return self.env.reset(seed=args.seed if seed is None else seed)

    def render(self):
        return self.env.render_with_arrow(self.info)
    
    def duration_steps(self, option_idx: int):
        return round(self.options[option_idx]['duration'] / args.dt)

options = []
for i in range(4):  # 4 legs
    options.append({
        "name": "sinusoid_forward",
        "hip_joint_idx": 2*i,
        "hip_target": np.radians(45),
        "knee_joint_idx": 2*i + 1,
        "knee_amplitude": np.radians(45),
        "duration": 0.6
    })
    options.append({
        "name": "sinusoid_backward",
        "hip_joint_idx": 2*i,
        "hip_target": -np.radians(45),
        "knee_joint_idx": 2*i + 1,
        "knee_amplitude": np.radians(45),
        "duration": 0.6
    })
    options.append({
        "name": "stance_forward",
        "hip_joint_idx": 2*i,
        "hip_target": np.radians(45),
        "knee_joint_idx": 2*i + 1,
        "knee_amplitude": np.radians(-20), # This is so it pushes into the ground for better contact.
        "duration": 0.6
    })
    options.append({
        "name": "stance_backward",
        "hip_joint_idx": 2*i,
        "hip_target": -np.radians(45),
        "knee_joint_idx": 2*i + 1,
        "knee_amplitude": np.radians(-20),
        "duration": 0.6
    })

print(len(options), "options defined.")

# Tile coding.
class SuttonTileCoderWrapper:
    def __init__(self, iht: IHT, tiles_per_dim, value_limits, tilings):
        self.iht = iht
        self.tiles_per_dim = np.asarray(tiles_per_dim, dtype=np.int32)
        self.tilings = int(tilings)
        self.limits = np.asarray(value_limits, dtype=np.float64)
        self.scaling = np.array(tiles_per_dim) / (self.limits[:, 1] - self.limits[:, 0])
        assert self.limits.shape == (self.tiles_per_dim.shape[0], 2)

    def __getitem__(self, x):
        x = np.asarray(x, dtype=np.float64)
        idxs = tiles(self.iht, self.tilings, self.scaling * x)
        return np.asarray(idxs, dtype=np.int64)

    @property
    def n_tiles(self):
        return self.iht.size

def q_of(w, idx, o):
    return w[o, idx].sum()

def select_greedy_option(w, T, state, num_options):
    idx = T[state]
    q_vals = np.array([w[o, idx].sum() for o in range(num_options)], dtype=np.float64)
    # Tie-break among maxima, in case of ties.
    maxq = q_vals.max()
    best = np.flatnonzero(q_vals == maxq)
    # plt.clf()
    # plt.bar(range(len(q_vals)), q_vals)
    # # Color the highest q-value in red.
    # plt.bar(np.argmax(q_vals), q_vals[np.argmax(q_vals)], color='red')
    # plt.title('Q-values for each option')
    # plt.xlabel('Option')
    # plt.ylabel('Q-value')
    # plt.pause(0.01)
    return int(np.random.choice(best)), q_vals

def select_option_epsilon_greedy(S, epsilon, w, T):
    # ε-greedy over options using tile-coded T(s).
    if np.random.rand() < epsilon:
        return np.random.randint(num_options)
    O_greedy, _ = select_greedy_option(w, T, S, num_options)
    return O_greedy

def clip_state_to_limits(S, limits):
    S = np.asarray(S, dtype=np.float64)
    return np.clip(S, limits[:, 0], limits[:, 1])


# Parser.
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--hw_config', type=str, default=None)
parser.add_argument('--learn', type=bool, default=True)
parser.add_argument('--load_previous_weights', type=bool, default=False)
parser.add_argument("--render_mode", type=str, default="rgb_array",
                        help="render mode")
parser.add_argument('--dt', type=float, default=0.05)
parser.add_argument('--env_id', type=str, default='SimEmbodiedAnt')
parser.add_argument('--capture_video', action='store_true')
parser.add_argument('--exp_name', type=str, default='sarsa_ant_forward')
parser.add_argument('--reward_scaling', type=float, default=10.0)
parser.add_argument('--load_weights_from_dir', type=str, default=None)

args = parser.parse_args()
SEED = args.seed
np.random.seed(SEED)


# Directories.
file_name = args.exp_name + '_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_dir = os.path.join(os.path.dirname(__file__), 'logs', file_name)
os.makedirs(log_dir, exist_ok=True)

weights_iht_folder = os.path.join(log_dir, "weights_iht")
if not os.path.exists(weights_iht_folder):
    os.makedirs(weights_iht_folder)

frames_folder = os.path.join(log_dir, "frames")
if not os.path.exists(frames_folder):
    os.makedirs(frames_folder)

folder_trajectory = os.path.join(log_dir, "trajectory")
if not os.path.exists(folder_trajectory):
    os.makedirs(folder_trajectory)

# Environment.
joint_config = {
    'hip_zero': 0.0,
    'knee_zero': -np.radians(60),
    'hip_range': np.radians(45),
    'knee_range': np.radians(45),
}


hw_config = args.hw_config if args.hw_config is not None else None
if args.hw_config is None:
    env = AntEnv(
        control_dt=args.dt,
        render_mode=args.render_mode,
        task=ForwardTask(),
        joint_config=joint_config,
        model_path=os.path.join(os.path.dirname(__file__), '../../sim/assets/embodied_mujoco_ant.xml'),
    )
    if args.capture_video:
        print('RecordVideo')
        env = gym.wrappers.RecordVideo(env, os.path.join(log_dir, "videos", args.env_id),
                                        step_trigger=lambda x: x % 1000 == 0)

else:
    with open(args.hw_config, 'r') as f:
        cfg = json.load(f)
    env = make_ant_env(cfg, render_mode=args.render_mode,
                        dt=args.dt,
                        joint_config=joint_config,
                        task=ForwardTask(),
                        )


# Constants.
MAX_OPTIONS_PER_TIMELIMIT_EPISODE = 300
EPSILON = 0.05
EPSILON_START = EPSILON
DISCOUNTING = 0.99

DIM_TILING = 10 # Number of tiles per dimension.
TILINGS = 4*env.observation_space.shape[0] # Number of offset tilings.
IHT_SIZE = 2**20

USE_DECAYING_EPSILON = False

# Environment.
options_env = OptionEnv(env, options)

# Limits from observation space.
state_limits = np.array([env.observation_space.low, env.observation_space.high]).T  # [state_dim, 2]
num_options = len(options)

# Load previous weights.
if args.load_weights_from_dir is None:
    iht = IHT(IHT_SIZE)
    w = np.zeros((num_options, iht.size), dtype=np.float32)
else:
    w = np.load(os.path.join(args.load_weights_from_dir, 'weights_iht/weights.npy'))
    with open(os.path.join(args.load_weights_from_dir, "weights_iht/iht.pkl"), "rb") as f:
        iht = pickle.load(f)
    print('Loaded weights from ', args.load_weights_from_dir)
    if args.learn == False:
        EPSILON = 0.0


# IHT table size.
tiles_per_dim = [DIM_TILING] * state_limits.shape[0]
T = SuttonTileCoderWrapper(iht=iht,
                           tiles_per_dim=tiles_per_dim,
                           value_limits=state_limits,
                           tilings=TILINGS)
step_size = 0.1 / TILINGS # Step-size, see: http://incompleteideas.net/tiles/tiles3.html.


with open(os.path.join(log_dir, "config.json"), "w") as f:
    json.dump({
        "tiles_per_dim": tiles_per_dim,
        "tilings": TILINGS,
        "state_limits": state_limits.tolist(),
        "iht_size": IHT_SIZE,
        "step_size": step_size,
        "discount": DISCOUNTING,
        "epsilon": EPSILON,
        "epsilon_start": EPSILON_START,
        "max_options_per_timelimit_episode": MAX_OPTIONS_PER_TIMELIMIT_EPISODE,
        "use_decaying_epsilon": USE_DECAYING_EPSILON,
        "log_dir": log_dir,
        "env_id": args.env_id,
        "dt": args.dt,
        "learn": args.learn,
        "load_previous_weights": args.load_previous_weights,
        "joint_config": joint_config,
    }, f, indent=2)


logging_data = {
    "timelimit_episode": [],
    "return_per_timelimit": [],
    "real_time_seconds": [],
    "reward_per_option": [],
}
logging_data_df = pd.DataFrame(logging_data)

idx_options = 0
return_per_timelimit = 0.0
idx_timelimit_episode = 0
real_time_seconds = 0.0

# Reset environment.
S, _ = options_env.reset(seed=SEED)
O = select_option_epsilon_greedy(S, EPSILON, w, T)

while True:
    if USE_DECAYING_EPSILON:
        EPSILON = max(0.05, EPSILON_START - idx_timelimit_episode * 0.015)
        print(f"Decaying epsilon to {EPSILON}")

    # Step.
    S_prime, R, terminated, truncated, info = options_env.step(O)

    # Next option (ε-greedy).
    O_prime = select_option_epsilon_greedy(S_prime, EPSILON, w, T)

    # TD.
    k = options_env.duration_steps(O)
    idx_S = T[S]
    idx_S_prime = T[S_prime]

    target = R + (DISCOUNTING ** k) * q_of(w, idx_S_prime, O_prime)
    pred = q_of(w, idx_S,  O)
    TD_error = target - pred

    # Update weights.
    if args.learn == True:
        w[O, idx_S] += step_size * TD_error

    S = S_prime
    O = O_prime

    return_per_timelimit += R
    real_time_seconds += options_env.duration_steps(O) * args.dt

    idx_options += 1

    if terminated or truncated:
        print('Terminated', terminated, 'truncated', truncated)
        S, _ = options_env.reset(seed=SEED)
        O = select_option_epsilon_greedy(S, EPSILON, w, T)

    if idx_options >= MAX_OPTIONS_PER_TIMELIMIT_EPISODE:
        logging_data["timelimit_episode"].append(idx_timelimit_episode)
        logging_data["return_per_timelimit"].append(return_per_timelimit)
        logging_data["reward_per_option"].append(R)
        logging_data["real_time_seconds"].append(real_time_seconds)

        print(f"Ep. {idx_timelimit_episode} | Return: {return_per_timelimit:.4f} | Time in sec: {(real_time_seconds):.4f} | Time in hours: {(real_time_seconds) / 3600:.4f} | Average reward: {options_env.reward_tracker.average_reward_per_second:.4f}")

        # Save logging data.
        new_row = {
            "timelimit_episode": logging_data["timelimit_episode"][-1],
            "return_per_timelimit": logging_data["return_per_timelimit"][-1],
            "reward_per_option": logging_data["reward_per_option"][-1],
            "real_time_seconds": logging_data["real_time_seconds"][-1],
        }
        logging_data_df = pd.concat([logging_data_df, pd.DataFrame([new_row])], ignore_index=True)
        logging_data_df.to_csv(os.path.join(log_dir, "logging_data.csv"), index=False)
        
        # Reset logging_data to avoid accumulation
        logging_data = {
            "timelimit_episode": [],
            "return_per_timelimit": [],
            "real_time_seconds": [],
            "reward_per_option": [],
        }

        # Save weights.
        np.save(os.path.join(weights_iht_folder, f"weights.npy"), w)
        pickle.dump(iht, open(os.path.join(weights_iht_folder, f"iht.pkl"), "wb"))

        idx_timelimit_episode += 1
        idx_options = 0
        return_per_timelimit = 0.0