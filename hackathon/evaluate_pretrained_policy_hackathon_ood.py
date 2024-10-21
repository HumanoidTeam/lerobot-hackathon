from pathlib import Path
import os
import argparse
import json

import gym_aloha
import gymnasium as gym
import imageio
import numpy
import numpy as np
import torch
from huggingface_hub import snapshot_download

from lerobot.scripts.eval import get_pretrained_policy_path

from lerobot.common.policies.act.modeling_act import ACTPolicy


def run(pretratined_policy_path: Path | None = None,
        out_dir: str | None = None,
        num_rollouts: int | None = None,
        num_videos: int | None = None,
        device: str | None = None ):

    device = torch.device("cpu" if device is None else device)
    pretrained_policy_path = Path(pretratined_policy_path)

    policy = ACTPolicy.from_pretrained(pretrained_policy_path)
    policy.eval()
    policy.to(device)

    output_directory = Path("hackathon/outputs/eval") if out_dir is None else Path(out_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    results_log_fname = "results_ood.log"
    results_json_fname = "results_ood.json"
    rollouts_log = []

    # Eval environment part
    task_id = "gym_aloha/AlohaHackathon-v0"

    env = gym.make(
        task_id,
        obs_type="pixels_agent_pos",
        max_episode_steps=1000
    )

    num_rollouts = 100 if num_rollouts is None else int(num_rollouts)
    num_success = 0
    num_videos_saved = 0
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

        if num_videos is None or num_videos_saved < int(num_videos):
            # Encode all frames into a mp4 video.
            video_path = output_directory / ("rollout_%s_%s.mp4" % (idx_rollout, terminated))
            imageio.mimsave(str(video_path), numpy.stack(frames), fps=fps)
            num_videos_saved += 1

        rollouts_log.append("rollout: %s, success: %s" % ( idx_rollout, terminated))

    success_rate = num_success / num_rollouts * 100
    print("Success rate: %4.2f%% over %s rollouts" % (success_rate, num_rollouts))
    rollouts_log.append("Total success rate: %4.2f%% over %s rollouts" % (success_rate, num_rollouts))

    with open(os.path.join(output_directory, results_log_fname), "w") as fp:
        for s in rollouts_log:
            fp.write(s + "\n")

    info_success_rates = {"total": success_rate}
    with open(os.path.join(output_directory, results_json_fname), "w") as fp:
        json.dump(info_success_rates, fp, indent=3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-p",
        "--pretrained-policy-name-or-path",
        required=True,
        help="Either the repo ID of a model hosted on the Hub or a path to a directory containing weights "
             "saved using `Policy.save_pretrained`.")

    parser.add_argument(
        "-o",
        "--out-dir",
        help=(
            "Where to save the evaluation outputs. If not provided, outputs are saved in "
            "hackathon/outputs/eval"
        ),
    )

    parser.add_argument(
        "-n",
        "--num-rollouts",
        help=(
            "Number of rollouts, 100 by default"
        ),
    )

    parser.add_argument(
        "-v",
        "--num-videos",
        help=(
            "Number of videos to save, all saved if not specified"
        ),
    )

    parser.add_argument(
        "-d",
        "--device",
        help=(
            "Device"
        ),
    )

    args = parser.parse_args()

    pretrained_policy_path = get_pretrained_policy_path(args.pretrained_policy_name_or_path)
    run(pretratined_policy_path=pretrained_policy_path,
        out_dir=args.out_dir,
        num_rollouts=args.num_rollouts,
        num_videos=args.num_videos,
        device=args.device)

