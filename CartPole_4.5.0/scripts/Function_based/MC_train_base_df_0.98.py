"""Script to train RL agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import wandb
wandb.login(key="88baa274f550d2a9eee583bbb7bef8d179637368")

from RL_Algorithm.Function_based.MC_REINFORCE_test import MC_REINFORCE 

from tqdm import tqdm

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
from datetime import datetime
import random

import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
# from omni.isaac.lab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import CartPole.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

steps_done = 0

@hydra_task_config(args_cli.task, "sb3_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with stable-baselines agent."""
    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # ==================================================================== #
    # ========================= Can be modified ========================== #

    # hyperparameters
    device = None
    num_of_action = 7
    action_range = [-2.5, 2.5]
    n_observations= 4
    hidden_dim = 64
    dropout = 0.5
    learning_rate = 0.01
    discount_factor = 0.98
    n_episodes = 2000

    # print(f"action space dim: {env.action_space.shape[0]}")
    # set up matplotlib
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    plt.ion()

    # if GPU is to be used
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    print("device: ", device)

    task_name = str(args_cli.task).split('-')[0]  # Stabilize, SwingUp
    name_train = "MC_base_df_0.98"
    Algorithm_name = "MC"

    agent = MC_REINFORCE(
        device=device,
        num_of_action=num_of_action,
        n_observations=n_observations,
        hidden_dim=hidden_dim,
        dropout=dropout,
        learning_rate=learning_rate,
        discount_factor=discount_factor
    )
    wandb.init(project="HW3", name=name_train)

    # reset environment
    obs, _ = env.reset()
    timestep = 0
    sum_count = 0
    sum_reward = 0
    cumulative_reward = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        # with torch.inference_mode():
        
        for episode in tqdm(range(n_episodes)):
            episode_reward, loss, trajectory, count = agent.learn(env)
            # print(f"episode_reward: {episode_reward} and trajectory: {trajectory}" )

            cumulative_reward += episode_reward

            wandb.log({
                "episode": episode,
                "cumulative_reward": cumulative_reward,
                "episode_length": len(trajectory),
            })

            sum_count += count
            sum_reward += episode_reward

            if episode % 100 == 0:

                print(f"avg_score: {sum_reward / 100.0}")

                wandb.log({
                    "sum_reward": sum_reward / 100.0,
                    "count": sum_count / 2000.0,
                    "loss": loss,
                })

                # Save Q-Learning agent
                w_file = f"{Algorithm_name}_{episode}_{num_of_action}_{action_range[1]}.json"
                full_path = os.path.join(f"w/{task_name}", Algorithm_name)
                agent.save_w(full_path, w_file)

                sum_count = 0
                sum_reward = 0
            
        print('Complete')
        agent.plot_durations(show_result=True)
        plt.ioff()
        plt.show()
            
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        break
    # ==================================================================== #

    # close the simulator
    env.close()
    wandb.finish()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()