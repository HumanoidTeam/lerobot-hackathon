from pathlib import Path
import os
import argparse

import gym_aloha
import gymnasium as gym
import imageio
import numpy
import torch
import json

from lerobot.scripts.eval import get_pretrained_policy_path

from lerobot.common.policies.act.modeling_act import ACTPolicy


def run(pretratined_policy_path: Path | None = None,
        out_dir: str | None = None,
        num_rollouts_per_object: int | None = None,
        num_videos_per_object: int | None = None,
        device: str | None = None ):

    device = torch.device("cpu" if device is None else device)
    pretrained_policy_path = Path(pretratined_policy_path)

    policy = ACTPolicy.from_pretrained(pretrained_policy_path)
    policy.eval()
    policy.to(device)

    output_directory = Path("hackathon/outputs/eval") if out_dir is None else Path(out_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    results_log_fname = "results.log"
    results_json_fname = "results.json"
    rollouts_log = []
    tasks_log = []

    # Eval environment part
    task_id = "gym_aloha/AlohaHackathon-v0"

    objects_names = ["sphere", "cube", "cylinder"]
    objects_colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
    objects_probabilities = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    num_rollouts_per_object = 100 if num_rollouts_per_object is None else int(num_rollouts_per_object)
    total_num_success = 0

    random_seed = 667
    idx_total = 0

    info_success_rates = {}

    for idx_object, object_name in enumerate(objects_names):
        env_options = {"weights": objects_probabilities[idx_object],
                       "color": objects_colors[idx_object]}

        env = gym.make(
            task_id,
            obs_type="pixels_agent_pos",
            max_episode_steps=1000
        )

        num_success = 0
        num_videos_saved = 0

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

            idx_total += 1

            fps = env.metadata["render_fps"]

            if num_videos_per_object is None or num_videos_saved < int(num_videos_per_object):
                # Encode all frames into a mp4 video.
                video_path = output_directory / ("%s_rollout_%s_%s.mp4" % (object_name, idx_rollout, terminated))
                imageio.mimsave(str(video_path), numpy.stack(frames), fps=fps)
                num_videos_saved += 1

            rollouts_log.append("Object: <%s>, rollout: %s, success: %s" % (object_name, idx_rollout, terminated))

        task_success_rate = num_success / num_rollouts_per_object * 100
        task_str = "Object <%s> success rate: %4.2f%% over %s rollouts" % (object_name, task_success_rate, num_rollouts_per_object)
        print(task_str)
        tasks_log.append(task_str)

        info_success_rates[object_name] = task_success_rate

        total_num_success += num_success



    total_success_rate = total_num_success / (num_rollouts_per_object * len(objects_names)) * 100
    total_str = "Total success rate: %4.2f%% over (%s rollouts x %s objects)" % (total_success_rate, num_rollouts_per_object, len(objects_names))
    print(total_str)
    tasks_log.append(total_str)

    info_success_rates["total"] = total_success_rate

    with open(os.path.join(output_directory, results_log_fname), "w") as fp:
        for s in rollouts_log:
            fp.write(s + "\n")
        fp.write("\n\n")
        for s in tasks_log:
            fp.write(s + "\n")


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
            "Number of rollouts per each object type, 100 by default"
        ),
    )

    parser.add_argument(
        "-v",
        "--num-videos",
        help=(
            "Number of videos to save per each object type, all saved if not specified"
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
        num_rollouts_per_object=args.num_rollouts,
        num_videos_per_object=args.num_videos,
        device=args.device)


