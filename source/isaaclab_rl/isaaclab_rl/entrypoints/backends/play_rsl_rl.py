# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint of an RL agent from RSL-RL."""

import argparse
import contextlib
import importlib.metadata as metadata
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.seed import configure_seed
from isaaclab.utils.string import list_intersection, string_to_callable

from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args
from isaaclab_rl.entrypoints.common import (
    CHECKPOINT_SELECTORS,
    EpisodeSuccessEvaluator,
    add_frontend_args,
    apply_video_recording,
    create_isaaclab_env,
    pre_launch_video_config,
    request_determinism,
    resolve_checkpoint_selector,
    resolve_play_task_name,
    show_run_summary,
    startup_screen,
)
from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_rl.utils.pretrained_checkpoint import (
    get_pretrained_checkpoint_backend_names,
    get_published_pretrained_checkpoint,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import (
    get_checkpoint_path,
    setup_preset_cli,
)
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

# -- argparse ----------------------------------------------------------------
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument(
    "--video_length",
    type=int,
    default=None,
    help="Length of each recorded video clip in env steps. Overrides the value in VideoRecorderCfg.",
)
parser.add_argument(
    "--video_interval",
    type=int,
    default=None,
    help="Interval between video clips in env steps. Overrides the value in VideoRecorderCfg.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--eval_episodes", type=int, default=None, help="Evaluate exactly this many episodes, then exit.")
parser.add_argument("--eval_success_term", default="success", help="Manager termination term used to count successes.")
parser.add_argument(
    "--eval_output", type=str, default=None, help="Evaluation JSON path (default: checkpoint directory/eval)."
)
parser.add_argument(
    "--video_grid", action="store_true", help="Record a headless Newton GL camera grid, one tile per environment."
)
parser.add_argument(
    "--video_eye", type=float, nargs=3, default=(4.0, -4.0, 3.0), help="Grid camera offset from each env origin [m]."
)
parser.add_argument("--video_lookat", type=float, nargs=3, default=(0.0, 0.0, 0.0), help="Grid look-at offset [m].")
parser.add_argument(
    "--video_closeup_env", type=int, default=None, help="Replace the bottom-right 2x2 tiles with this env."
)
parser.add_argument(
    "--video_closeup_eye", type=float, nargs=3, default=(0.0, -0.35, 0.525), help="Closeup eye offset from target [m]."
)
parser.add_argument(
    "--video_closeup_offset",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.0),
    help="Closeup target offset from env origin [m].",
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--train_env_cfg",
    action="store_true",
    default=False,
    help="Play with the training environment configuration as-is, skipping play-mode overrides.",
)
parser.add_argument("--external_callback", default=None, help="Fully qualified path to an externally defined callback.")
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
add_frontend_args(parser)
args_cli, remaining_args = setup_preset_cli(parser, agent_library="rsl_rl")
args_cli.task = resolve_play_task_name(args_cli.task)

if args_cli.eval_episodes is not None and args_cli.eval_episodes <= 0:
    parser.error("--eval_episodes must be positive.")
if args_cli.eval_output and args_cli.eval_episodes is None:
    parser.error("--eval_output requires --eval_episodes.")
if args_cli.video_closeup_env is not None and not args_cli.video_grid:
    parser.error("--video_closeup_env requires --video_grid.")
if args_cli.video_grid:
    if args_cli.visualizer not in (None, ["newton_gl"]):
        parser.error("--video_grid requires --visualizer newton_gl (or no explicit visualizer).")
    args_cli.video = True
    args_cli.visualizer = ["newton_gl"]

if args_cli.video:
    args_cli.enable_cameras = True


# an external callback lets downstream code register its environments; it returns
# the arguments it did not consume
remaining_args_env_registration = None
if args_cli.external_callback:
    external_callback_function = string_to_callable(args_cli.external_callback, separator=".")
    remaining_args_env_registration = external_callback_function()

# hand the arguments consumed by neither this parser nor the callback over to Hydra
remaining_args = list_intersection(remaining_args, remaining_args_env_registration)
sys.argv = [sys.argv[0]] + remaining_args

installed_version = metadata.version("rsl-rl-lib")


def _configure_evaluation(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg) -> None:
    """Validate evaluation and configure an optional grid before launching simulation."""
    if args_cli.eval_episodes is not None:
        if not isinstance(env_cfg, ManagerBasedRLEnvCfg):
            raise ValueError("--eval_episodes requires a manager-based environment with a success termination term.")
        success_cfg = getattr(env_cfg.terminations, args_cli.eval_success_term, None)
        if isinstance(env_cfg.terminations, dict):
            success_cfg = env_cfg.terminations.get(args_cli.eval_success_term)
        if success_cfg is None or success_cfg.time_out:
            raise ValueError(f"Expected an active non-timeout termination term: {args_cli.eval_success_term!r}.")
    if args_cli.video_grid:
        from isaaclab_rl.entrypoints._newton_gl_video import _NewtonGLVideoGridCfg

        num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        if args_cli.video_closeup_env is not None:
            if num_envs < 9 or not 0 <= args_cli.video_closeup_env < num_envs:
                raise ValueError("The 2x2 closeup requires at least 9 environments and a valid --video_closeup_env.")
        env_cfg.sim.visualizer_cfgs = [
            _NewtonGLVideoGridCfg(
                grid_num_envs=num_envs,
                grid_eye=tuple(args_cli.video_eye),
                grid_lookat=tuple(args_cli.video_lookat),
                closeup_env=args_cli.video_closeup_env,
                closeup_eye=tuple(args_cli.video_closeup_eye),
                closeup_offset=tuple(args_cli.video_closeup_offset),
            )
        ]


@hydra_task_config(args_cli.task, args_cli.agent, play_mode=not args_cli.train_env_cfg)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    _configure_evaluation(env_cfg)
    pre_launch_video_config(env_cfg, args_cli=args_cli)
    with startup_screen(args_cli, num_stages=3) as screen:
        show_run_summary(screen, args_cli, env_cfg, library="rsl_rl", action="play")
        screen.stage("Launching simulation")
        with launch_simulation(env_cfg, args_cli):
            task_name = args_cli.task.split(":")[-1]
            train_task_name = task_name.replace("-Play", "")

            agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
            env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
            # Warp reads its determinism mode at module build time, so request it before the env exists.
            request_determinism(args_cli, env_cfg)

            agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

            # note: certain randomizations occur in the environment initialization so we set the seed here
            env_cfg.seed = agent_cfg.seed
            env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

            log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
            log_root_path = os.path.abspath(log_root_path)
            print(f"[INFO] Loading experiment from directory: {log_root_path}")
            if args_cli.checkpoint == "pretrained":
                backend_names = get_pretrained_checkpoint_backend_names(env_cfg)
                resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name, *backend_names)
                if not resume_path:
                    return
            elif args_cli.checkpoint in CHECKPOINT_SELECTORS:
                resume_path = resolve_checkpoint_selector(
                    log_root_path,
                    args_cli.checkpoint,
                    library="rsl_rl",
                    task=train_task_name,
                    checkpoint_pattern=r"model_.*\.pt",
                    metadata={"agent": args_cli.agent},
                )
            elif args_cli.checkpoint and os.path.isdir(args_cli.checkpoint):
                resume_path = get_checkpoint_path(
                    os.path.dirname(args_cli.checkpoint),
                    os.path.basename(args_cli.checkpoint),
                    agent_cfg.load_checkpoint,
                )
            elif args_cli.checkpoint:
                resume_path = retrieve_file_path(args_cli.checkpoint)
            else:
                resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

            log_dir = os.path.dirname(resume_path)

            env_cfg.log_dir = log_dir
            apply_video_recording(env_cfg, log_dir, args_cli, subdir="play", checkpoint_path=resume_path)
            if args_cli.video_grid:
                for recorder_cfg in env_cfg.video_recorders:
                    recorder_cfg.source = "visualizer:newton_gl"
                    recorder_cfg.output_filename_prefix += "_gl_grid"

            screen.stage("Creating environment")
            env = create_isaaclab_env(
                args_cli.task,
                env_cfg,
                args_cli,
                convert_marl_to_single_agent=isinstance(env_cfg, DirectMARLEnvCfg),
            )

            screen.stage("Loading policy")
            env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

            print(f"[INFO]: Loading model checkpoint from: {resume_path}")
            if agent_cfg.class_name == "OnPolicyRunner":
                runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            elif agent_cfg.class_name == "DistillationRunner":
                runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            else:
                raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
            # configure_seed must run after runner construction so torch determinism does not disturb its initialization
            if args_cli.deterministic:
                configure_seed(env_cfg.seed, torch_deterministic=True)
            runner.load(resume_path)

            policy = runner.get_inference_policy(device=env.unwrapped.device)

            export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

            if version.parse(installed_version) >= version.parse("4.0.0"):
                runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
                runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
                policy_nn = None  # Not needed for rsl-rl >= 4.0.0
            else:
                if version.parse(installed_version) >= version.parse("2.3.0"):
                    policy_nn = runner.alg.policy
                else:
                    policy_nn = runner.alg.actor_critic

                if hasattr(policy_nn, "actor_obs_normalizer"):
                    normalizer = policy_nn.actor_obs_normalizer
                elif hasattr(policy_nn, "student_obs_normalizer"):
                    normalizer = policy_nn.student_obs_normalizer
                else:
                    normalizer = None

                export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
                export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

            dt = env.unwrapped.step_dt

            screen.close()
            obs = env.get_observations()
            evaluator = None
            if args_cli.eval_episodes is not None:
                evaluator = EpisodeSuccessEvaluator(env.num_envs, args_cli.eval_episodes, env.unwrapped.device)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                eval_output = (
                    Path(args_cli.eval_output)
                    if args_cli.eval_output
                    else (Path(log_dir) / "eval" / f"{Path(resume_path).stem}_{timestamp}.json")
                )
                print(f"[EVAL] Evaluating {args_cli.eval_episodes} episodes; results: {eval_output}")
            timestep = 0
            print("[INFO] Policy playback is running, press Ctrl+C to exit...")
            try:
                while True:
                    start_time = time.time()
                    with torch.inference_mode():
                        actions = policy(obs)
                        obs, _, dones, _ = env.step(actions)
                        # reset recurrent states for episodes that have terminated
                        if version.parse(installed_version) >= version.parse("4.0.0"):
                            policy.reset(dones)
                        else:
                            policy_nn.reset(dones)
                        if evaluator is not None:
                            # TerminationManager retains these flags across the automatic reset in step().
                            successes = env.unwrapped.termination_manager.get_term(args_cli.eval_success_term)
                            if evaluator.update(dones, successes):
                                completed = len(evaluator.episodes)
                                print(
                                    f"[EVAL] {completed}/{evaluator.num_episodes} episodes, "
                                    f"successes={evaluator.successes}, SR={evaluator.successes / completed:.2%}",
                                    flush=True,
                                )
                    if evaluator is not None and evaluator.complete:
                        break
                    if args_cli.video and evaluator is None:
                        timestep += 1
                        video_stop = args_cli.video_length
                        if video_stop is None:
                            recorders = getattr(env_cfg, "video_recorders", [])
                            video_stop = recorders[0].video_length + recorders[0].step_offset if recorders else None
                        if video_stop is not None and timestep >= video_stop:
                            break

                    sleep_time = dt - (time.time() - start_time)
                    if args_cli.real_time and sleep_time > 0:
                        time.sleep(sleep_time)

            except KeyboardInterrupt:
                pass
            finally:
                if evaluator is not None:
                    completed = len(evaluator.episodes)
                    result = {
                        "task": args_cli.task,
                        "checkpoint": str(Path(resume_path).resolve()),
                        "seed": env_cfg.seed,
                        "num_envs": env.num_envs,
                        "success_term": args_cli.eval_success_term,
                        "requested_episodes": evaluator.num_episodes,
                        "completed_episodes": completed,
                        "complete": evaluator.complete,
                        "successes": evaluator.successes,
                        "success_rate": evaluator.successes / completed if completed else None,
                        "step_dt": dt,
                        "episode_quotas": evaluator.quotas.tolist(),
                        "episodes": evaluator.episodes,
                    }
                    eval_output.parent.mkdir(parents=True, exist_ok=True)
                    eval_output.write_text(json.dumps(result, indent=2) + "\n")
                    print(f"[EVAL] Saved {'complete' if evaluator.complete else 'partial'} results: {eval_output}")
                env.close()


if __name__ == "__main__":
    main()
