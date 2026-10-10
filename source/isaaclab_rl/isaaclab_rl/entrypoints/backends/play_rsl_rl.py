# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL playback backend of the unified reinforcement learning entrypoint."""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs import DirectMARLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils import to_dict
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.seed import configure_seed
from isaaclab.utils.string import list_intersection

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, resolve_task_config, setup_preset_cli

from ...rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from ...utils.wandb import is_wandb_checkpoint, resolve_wandb_checkpoint
from ..common import (
    CHECKPOINT_SELECTORS,
    EpisodeSuccessEvaluator,
    add_common_play_args,
    apply_env_overrides,
    apply_video_recording,
    close_env,
    create_isaaclab_env,
    enable_cameras_for_video,
    normalize_task_name,
    pre_launch_video_config,
    resolve_checkpoint_selector,
    resolve_published_checkpoint,
    run_playback,
    set_hydra_args,
    show_run_summary,
    startup_screen,
)
from . import cli_args_rsl_rl as cli_args

logger = logging.getLogger(__name__)

# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse RSL-RL playback arguments."""
    parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RSL-RL.")
    add_common_play_args(
        parser,
        agent_default="rsl_rl_cfg_entry_point",
        agent_help="Name of the RL agent configuration entry point.",
    )
    parser.add_argument(
        "--external_callback", default=None, help="Fully qualified path to an externally defined callback."
    )
    parser.add_argument(
        "--eval_episodes", type=int, default=None, help="Evaluate exactly this many episodes, then exit."
    )
    parser.add_argument(
        "--eval_success_term", default="success", help="Manager termination term used to count successes."
    )
    parser.add_argument(
        "--eval_output", type=str, default=None, help="Evaluation JSON path (default: checkpoint directory/eval)."
    )
    parser.add_argument(
        "--video_grid", action="store_true", help="Record a headless Newton GL camera grid, one tile per environment."
    )
    parser.add_argument(
        "--video_eye",
        type=float,
        nargs=3,
        default=(4.0, -4.0, 3.0),
        help="Grid camera offset from each env origin [m].",
    )
    parser.add_argument("--video_lookat", type=float, nargs=3, default=(0.0, 0.0, 0.0), help="Grid look-at offset [m].")
    parser.add_argument(
        "--video_closeup_env", type=int, default=None, help="Replace the bottom-right 2x2 tiles with this env."
    )
    parser.add_argument(
        "--video_closeup_eye",
        type=float,
        nargs=3,
        default=(0.0, -0.35, 0.525),
        help="Closeup eye offset from target [m].",
    )
    parser.add_argument(
        "--video_closeup_offset",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        help="Closeup target offset from env origin [m].",
    )
    cli_args.add_rsl_rl_args(parser)
    add_launcher_args(parser)
    remaining_args_env_registration = cli_args.register_external_tasks(argv)
    args_cli, remaining_args = setup_preset_cli(parser, argv)
    if args_cli.eval_episodes is not None and args_cli.eval_episodes <= 0:
        parser.error("--eval_episodes must be positive.")
    if args_cli.eval_output and args_cli.eval_episodes is None:
        parser.error("--eval_output requires --eval_episodes.")
    if args_cli.video_closeup_env is not None and not args_cli.video_grid:
        parser.error("--video_closeup_env requires --video_grid.")
    if args_cli.video_grid:
        if args_cli.visualizer not in (None, [], ["newton_gl"]):
            parser.error("--video_grid requires --visualizer newton_gl (or no explicit visualizer).")
        args_cli.video = "viz:newton_gl"
    enable_cameras_for_video(args_cli)
    set_hydra_args(list_intersection(remaining_args, remaining_args_env_registration))
    return args_cli


def _configure_evaluation(env_cfg: object, args_cli: argparse.Namespace) -> None:
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
        from .._newton_gl_video import _NewtonGLVideoGridCfg

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


def _resolve_checkpoint(
    args_cli: argparse.Namespace, agent_cfg: RslRlBaseRunnerCfg, env_cfg: object, log_root_path: str
) -> str | None:
    """Resolve the checkpoint to play, or None when no published checkpoint exists."""
    if args_cli.checkpoint and is_wandb_checkpoint(args_cli.checkpoint):
        return resolve_wandb_checkpoint(args_cli.checkpoint)
    if args_cli.checkpoint == "pretrained":
        return resolve_published_checkpoint("rsl_rl", args_cli.task, env_cfg)
    if args_cli.checkpoint in CHECKPOINT_SELECTORS:
        return resolve_checkpoint_selector(
            log_root_path,
            args_cli.checkpoint,
            library="rsl_rl",
            task=normalize_task_name(args_cli.task),
            checkpoint_pattern=r"model_.*\.pt",
            metadata={"agent": args_cli.agent},
        )
    if args_cli.checkpoint and os.path.isdir(args_cli.checkpoint):
        return get_checkpoint_path(
            os.path.dirname(args_cli.checkpoint), os.path.basename(args_cli.checkpoint), agent_cfg.load_checkpoint
        )
    if args_cli.checkpoint:
        return retrieve_file_path(args_cli.checkpoint)
    return get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)


def run(argv: list[str]) -> None:
    """Play a checkpoint of an RSL-RL agent."""
    args_cli = _parse_args(argv)
    with startup_screen(args_cli, num_stages=3) as screen:
        env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, play_mode=not args_cli.train_env_cfg)
        _configure_evaluation(env_cfg, args_cli)
        pre_launch_video_config(env_cfg, args_cli)
        screen.stage("Launching simulation")
        with launch_simulation(env_cfg, args_cli), contextlib.ExitStack() as cleanup:
            show_run_summary(screen, args_cli, env_cfg, library="rsl_rl", action="play")
            agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
            apply_env_overrides(args_cli, env_cfg)
            # certain randomizations occur in the environment initialization so we set the seed here
            env_cfg.seed = agent_cfg.seed

            log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
            logger.info(f"Loading experiment from directory: {log_root_path}")
            resume_path = _resolve_checkpoint(args_cli, agent_cfg, env_cfg, log_root_path)
            if resume_path is None:
                return
            log_dir = os.path.dirname(resume_path)
            env_cfg.log_dir = log_dir
            apply_video_recording(env_cfg, log_dir, args_cli, subdir="play", checkpoint_path=resume_path)
            if args_cli.video_grid:
                for recorder_cfg in env_cfg.video_recorders:
                    recorder_cfg.source = "viz:newton_gl"
                    recorder_cfg.output_filename_prefix += "_gl_grid"

            screen.stage("Creating environment")
            env = create_isaaclab_env(
                args_cli.task,
                env_cfg,
                args_cli,
                convert_marl_to_single_agent=isinstance(env_cfg, DirectMARLEnvCfg),
            )
            cleanup.callback(lambda: close_env(env))

            screen.stage("Loading policy")
            env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
            logger.info(f"Loading model checkpoint from: {resume_path}")
            if agent_cfg.class_name == "OnPolicyRunner":
                runner = OnPolicyRunner(env, to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
            elif agent_cfg.class_name == "DistillationRunner":
                runner = DistillationRunner(env, to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
            else:
                raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
            # configure_seed must run after runner construction so torch determinism does not disturb its initialization
            if args_cli.deterministic:
                configure_seed(env_cfg.seed, torch_deterministic=True)
            runner.load(resume_path)
            policy = runner.get_inference_policy(device=env.unwrapped.device)

            export_model_dir = os.path.join(log_dir, "exported")
            runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
            runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")

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
                logger.info(f"Evaluating {args_cli.eval_episodes} episodes; results: {eval_output}")

            def step() -> None:
                nonlocal obs
                obs, _, dones, _ = env.step(policy(obs))
                # reset recurrent states for episodes that have terminated
                policy.reset(dones)
                if evaluator is not None:
                    successes = env.unwrapped.termination_manager.get_term(args_cli.eval_success_term)
                    if evaluator.update(dones, successes):
                        completed = len(evaluator.episodes)
                        logger.info(
                            f"{completed}/{evaluator.num_episodes} episodes, "
                            f"successes={evaluator.successes}, SR={evaluator.successes / completed:.2%}"
                        )

            screen.close()
            if evaluator is None:
                run_playback(step, dt=env.unwrapped.step_dt, args_cli=args_cli, env_cfg=env_cfg)
            else:
                try:
                    with contextlib.suppress(KeyboardInterrupt):
                        while not evaluator.complete:
                            start_time = time.time()
                            with torch.inference_mode():
                                step()
                            sleep_time = env.unwrapped.step_dt - (time.time() - start_time)
                            if args_cli.real_time and sleep_time > 0:
                                time.sleep(sleep_time)
                finally:
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
                        "step_dt": env.unwrapped.step_dt,
                        "episode_quotas": evaluator.quotas.tolist(),
                        "episodes": evaluator.episodes,
                    }
                    eval_output.parent.mkdir(parents=True, exist_ok=True)
                    eval_output.write_text(json.dumps(result, indent=2) + "\n")
                    logger.info(f"Saved {'complete' if evaluator.complete else 'partial'} results: {eval_output}")
