from pathlib import Path
import os

import cv2

import gym_aloha
import gymnasium as gym
import imageio
import numpy
import numpy as np
import torch
from huggingface_hub import snapshot_download

# Policy initialization
from lerobot.common.policies.act.modeling_act import ACTPolicy

device = torch.device("cuda")

pretrained_policy_path = Path("HumanoidTeam/hackathon_sim_aloha")

policy = ACTPolicy.from_pretrained(pretrained_policy_path)
policy.eval()
policy.to(device)

# TODO: add some participant ID here?
output_directory = Path("hackathon/outputs/eval")
output_directory.mkdir(parents=True, exist_ok=True)
results_log_fname = "results_ood.log"
rollouts_log = []

# Eval environment part
task_id = "gym_aloha/AlohaHackathon-v0"

env = gym.make(
    task_id,
    obs_type="pixels_agent_pos",
    max_episode_steps=1000
)

num_rollouts = 100
num_success = 0
random_seed = 667

for idx_rollout in range(num_rollouts):
    policy.reset()

    numpy_observation, info = env.reset(seed=random_seed + idx_rollout)

    rewards = []
    frames = []

    # Render frame of the initial state
    frames.append(env.render()["top"])

    step = 0
    done = False
    while not done:
        state = torch.from_numpy(numpy_observation["agent_pos"])
        state = state.to(torch.float32)
        state = state.to(device, non_blocking=True)
        state = state.unsqueeze(0)

        images = {}
        for cam in numpy_observation["pixels"]:
            image = torch.from_numpy(numpy_observation["pixels"][cam])
            image = image.to(torch.float32) / 255
            image = image.permute(2, 0, 1)
            image = image.to(device, non_blocking=True)
            image = image.unsqueeze(0)
            images[cam] = image


        # Create the policy input dictionary
        observation = {
            "observation.state": state,
            "observation.images.top": images["top"],
            "observation.images.left_wrist": images["left_wrist"],
            "observation.images.right_wrist": images["right_wrist"],
        }

        # TODO: here we would send observations to the corresponding topic
        # Policy module
        # Predict the next action with respect to the current observation
        with torch.inference_mode():
            action = policy.select_action(observation)

        # Prepare the action for the environment
        numpy_action = action.squeeze(0).to("cpu").numpy()

        # Here we would wait for the message with action from the policy module

        # Step through the environment and receive a new observation
        numpy_observation, reward, terminated, truncated, info = env.step(numpy_action)
        print(f"{idx_rollout=} {step=} {reward=} {terminated=}")

        # Keep track of all the rewards and frames
        rewards.append(reward)
        frames.append(env.render()["top"])

        done = terminated | truncated | done
        step += 1

    if terminated:
        print("Success!")
        num_success += 1
    else:
        print("Failure!")

    fps = env.metadata["render_fps"]

    # Encode all frames into a mp4 video.
    video_path = output_directory / ("rollout_%s_%s.mp4" % (idx_rollout, terminated))
    imageio.mimsave(str(video_path), numpy.stack(frames), fps=fps)

    rollouts_log.append("rollout: %s, success: %s" % ( idx_rollout, terminated))

success_rate = num_success / num_rollouts * 100
print("Success rate: %4.2f%% over %s rollouts" % (success_rate, num_rollouts))
rollouts_log.append("Total success rate: %4.2f%% over %s rollouts" % (success_rate, num_rollouts))

with open(os.path.join(output_directory, results_log_fname), "w") as fp:
    for s in rollouts_log:
        fp.write(s + "\n")
