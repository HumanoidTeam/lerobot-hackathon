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
# TODO: to be moved to a separate ROS module if we want to decouple policy and env
from lerobot.common.policies.act.modeling_act import ACTPolicy

device = torch.device("cuda")

pretrained_policy_path = Path("HumanoidTeam/hackathon_sim_aloha")

policy = ACTPolicy.from_pretrained(pretrained_policy_path, local_files_only=False)
policy.eval()
policy.to(device)

# TODO: add some participant ID here?
output_directory = Path("hackathon/outputs/eval")
output_directory.mkdir(parents=True, exist_ok=True)
results_log_fname = "results.log"
rollouts_log = []
tasks_log = []

# Eval environment part
task_id = "gym_aloha/AlohaHackathon-v0"

objects_names = ["sphere", "cube", "cylinder"]
objects_colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
objects_probabilities = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

num_rollouts_per_object = 100
total_num_success = 0

random_seed = 667
idx_total = 0

for idx_object, object_name in enumerate(objects_names):
    env_options = {"weights": objects_probabilities[idx_object],
                   "color": objects_colors[idx_object]}

    env = gym.make(
        task_id,
        obs_type="pixels_agent_pos",
        max_episode_steps=1000
    )

    num_success = 0

    for idx_rollout in range(num_rollouts_per_object):
        policy.reset()  # TODO: this would be in the policy module if we decouple

        numpy_observation, info = env.reset(seed=random_seed + idx_total,
                                            options=env_options)

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
            print(f"{object_name=} {idx_rollout=} {step=} {reward=} {terminated=}")

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
        video_path = output_directory / ("%s_rollout_%s_%s.mp4" % (object_name, idx_rollout, terminated))
        imageio.mimsave(str(video_path), numpy.stack(frames), fps=fps)

        rollouts_log.append("Object: <%s>, rollout: %s, success: %s" % (object_name, idx_rollout, terminated))

    task_success_rate = num_success / num_rollouts_per_object * 100
    task_str = "Object <%s> success rate: %4.2f%% over %s rollouts" % (object_name, task_success_rate, num_rollouts_per_object)
    print(task_str)
    tasks_log.append(task_str)

    total_num_success += num_success

    idx_total += 1

total_success_rate = total_num_success / (num_rollouts_per_object * len(objects_names)) * 100
total_str = "Total success rate: %4.2f%% over (%s rollouts x %s objects)" % (total_success_rate, num_rollouts_per_object, len(objects_names))
print(total_str)
tasks_log.append(total_str)

with open(os.path.join(output_directory, results_log_fname), "w") as fp:
    for s in rollouts_log:
        fp.write(s + "\n")
    fp.write("\n\n")
    for s in tasks_log:
        fp.write(s + "\n")

