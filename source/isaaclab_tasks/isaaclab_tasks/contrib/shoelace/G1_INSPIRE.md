# G1 Inspire shoelace scene

The task `IsaacContrib-Shoelace-G1Inspire` uses Isaac Lab's `G1_INSPIRE_FTP_CFG` with a fixed pelvis,
two Cartesian arm controllers, and an open/pinch command for each Inspire hand. The shoe and pedestal
are kinematic. This first stage isolates manipulation from balance and walking.

## Scene and controls

- The pedestal is 78 cm high, with a 6×6 cm column and an 8×26 cm top. The sole overhangs the top
  so the hands can approach the laces from the sides without a wide table underneath them.
- The robot pelvis is 79.5 cm above the ground, 38 cm behind the shoe, facing it.
- Initial arm angles were measured with IK over the nominal baked free tails. The configured TCPs
  start 6 cm above those tails; this leaves clearance for the open little fingers.
- The 14 actions are left relative wrist pose (6), left open/pinch (1), right relative wrist pose (6),
  and right open/pinch (1). Positive hand actions open; nonpositive actions close.
- The imported articulation has 53 joints and 54 bodies, including 12 finger joints per hand.
  All finger joints participate in the hand synergy; they are not independently controlled by the policy.
- Contact observations use each thumb's distal body and index finger's intermediate body. Finger
  closure increases toward 1.2 rad, and the reward and closure residual account for that direction.
- Keep MJWarp's `implicitfast` integrator. Euler became nonfinite in the isolated stock-drive probe.
  Cable dynamics and coupling retain the Franka task's VBD and ADMM settings.

## Run

Run from the repository's uv environment:

```bash
uv run --no-sync isaaclab train --rl_library rsl_rl --task IsaacContrib-Shoelace-G1Inspire \
  --num_envs 64 --max_iterations 3 --logger tensorboard env.episode_length_s=0.2
```

The command reuses the existing environment. The short episode override exercises batch resets
during a smoke test. For learning runs,
omit it to retain the normal 10-second episodes and choose an appropriate iteration budget.

The task's configured Newton GL camera frames the robot, shoe, and pedestal. Add
`--visualizer newton_gl --max_visible_envs 1` to inspect a run, or use the same task ID with
`isaaclab play --rl_library rsl_rl --checkpoint <checkpoint-path>`.
Use a checkpoint trained with the G1 task: Franka checkpoints have different joint and TCP semantics.

The shoe assets must exist in `data/` beside this file, including `shoelace.usda` and its variants,
textures, and referenced shoe geometry. This checkout's global ignore rules exclude USD text assets.
For the local validation, the existing assets were copied from the sibling
`IsaacLab_shoelace_rl_example` worktree; the G1 USD remains the standard Isaac Lab asset.

## Validation and next stage

On an NVIDIA L20, the isolated G1 passed a 64-environment, 240-physics-step probe, including hand
motion, finite Jacobians, and partial joint reset isolation. The complete pedestal scene passed a
64-environment PPO smoke run with three updates and a 256-environment run with two updates, both
using six-step episode resets. Observations, rewards, and policy losses stayed finite.

A scripted bilateral open-hand pregrasp followed the free-tail positions 6 cm above them for
25 control steps. Final TCP errors were about 0.4 mm and 1.7 mm, with no robot-pedestal collision
candidates along that path. This checks this path only; a policy can still hit the support.

This is a runnable scene and training baseline. Thumb-index pinch targets and the TCP's change
during closure still need contact calibration and scripted grasp/hold/pull checks before a long
untying run. Successful untying, free-base balance, and multi-GPU scaling have not been validated.
