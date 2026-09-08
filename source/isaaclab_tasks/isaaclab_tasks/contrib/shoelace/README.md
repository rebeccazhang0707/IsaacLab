<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Dual-Franka shoelace task

This task trains two Franka arms to approach, grasp, retain, and pull the two free shoelace tails until the knot is
open. The production objective is a policy that completes the full task in Newton without grasp assistance.

Detailed experiment commands, metrics, rejected candidates, and diagnosis history live in
[`TRAINING_CHANGELOG.md`](TRAINING_CHANGELOG.md). This README only describes the task contract and recommended
training flow.

## Architecture

The environment contains:

- two independently controlled Franka arms;
- one deformable shoelace split into two simulated cable assets around the pinned shoe region;
- a Newton MJWarp cable solver coupled to the rigid robots;
- an optional diagnostic grasp spring that is disabled for production training and evaluation.

The shoelace assets are resolved from the repository-local root
`scripts/demos/assets/shoelace/`. Robot, shoe, cable, and texture paths are derived from that single root.

Production runs use ADMM two-way coupling. Robot motion affects the cable through contact, and cable reaction forces
feed back into the robot solve. This feedback is required for meaningful grasp retention under load; a one-way or
lagged contact approximation can make the gripper move without enforcing the same coupled contact constraint.

## Policy interface

The actor receives 68 values:

- both arm joint positions and velocities;
- both driven finger positions;
- tail-to-TCP, hand-frame contact-socket error, and tail-to-knot vectors;
- tail velocities;
- the previous filtered 14-dimensional action.

The critic receives the actor observation plus a privileged knot-throat density scalar. The actor does not receive a
Boolean grasp flag, contact force, curriculum phase, or complete cable centerline.

The 14-dimensional action is:

```text
left Cartesian delta pose (6) + left gripper (1)
+ right Cartesian delta pose (6) + right gripper (1)
```

Arm commands pass through a short reset warm-up and an exponential moving average before differential IK. A negative
gripper command closes toward the physical lace aperture; a non-negative command opens the gripper.

## Task phases

```text
approach both tails
  -> acquire both grasps
  -> retain both tails under load
  -> pull both tails away from the knot
  -> reduce throat occupancy and increase tail distance/separation
  -> success
```

The dense reward is phase-gated:

- before acquisition it rewards bilateral approach toward calibrated contact regions and a valid closing aperture;
- after acquisition it rewards reset-relative knot opening and synchronized motion in the reference pull directions;
- event rewards mark bilateral acquisition, success, and exclusive failure.

A success requires all of the following at the same step:

- sufficiently few cable segments remain in the knot throat;
- both tails are far enough from the knot;
- the two tails are sufficiently separated;
- both tails remain close to their assigned TCPs with a valid finger aperture;
- cable and robot state pass the safety checks.

Acquisition is accepted only when each tail enters a calibrated hand-frame contact region. Lost grasp is stateful:
after valid bilateral acquisition, retention uses a wider proximity condition and requires a short consecutive
release window, so normal contact motion or a single noisy frame does not terminate an episode.

## Curriculum

The checked-in curriculum has 97 reset levels, indexed `0` through `96`. It was built while training with PPO alone:

- early levels learn pulling from an already retained grasp;
- middle levels open the grippers and cross contact-release boundaries;
- later levels move the arms from contact into progressively farther approach poses;
- level 96 is the complete authored pregrasp reset.

These levels remain useful as a reproducible regression suite. They should not all be treated as mandatory sequential
training gates once expert demonstrations are available. In particular, the transition from level 91 to level 92 is
a measured approach-geometry discontinuity and must not be hidden by adding more reward weight.

## Recommended shortest training flow

The current recommendation is to reuse the learned contact policy and add imitation only for the missing approach and
handoff behavior:

1. Generate closed-loop demonstrations with ADMM coupling and grasp assistance disabled. Use a collision-free
   pregrasp waypoint, a hand-frame grasp socket, grasp verification, and slip-aware pulling.
2. Combine new level-92-to-96 demonstrations with successful level-80-to-91 rollouts from the retained policy.
3. Initialize from the retained policy and run low-learning-rate behavior cloning. Regularize the policy on the old
   state distribution so approach learning does not overwrite contact and pulling behavior.
4. Run a short DAgger cycle to label states reached after BC rollout errors.
5. Fine-tune with PPO on a mixture of a few anchor reset distributions rather than requiring promotion through every
   historical micro-level.
6. Expand to eight-GPU PPO only after the candidate passes the fixed-policy gates below.

The anchor set should cover retained pull, contact release, difficult contact entry, near approach, middle approach,
and the full level-96 reset. The exact anchors are selected from measured rollout failures; the legacy 97-level table
is retained for comparison.

## Validation gates

A candidate is accepted only if it:

- uses the production ADMM configuration, including matching time discretization, with grasp assistance disabled;
- completes the exact success termination rather than only moving the TCPs;
- retains the existing level-80-to-91 contact and pulling behavior;
- improves held-out approach rollouts at levels 92, 94, and 96;
- has no non-finite or unsafe terminations;
- reproduces across multiple seeds before a long PPO launch.

Exact checkpoint recommendations and their gate results are kept in the training changelog because they change as
experiments progress. The current development path reuses a validated contact-rich policy, while robust right-tail
load transfer through the approach-to-pull handoff remains the unresolved phase.

The frozen stage milestone solves strict bilateral socket acquisition and exact untying from the deterministic
level-91 near-contact reset with a statistically repeatable success rate, but it fails at the level-92 reset. This
is a contact-acquisition-and-pull milestone, not an end-to-end approach policy; its artifact, lineage, reset state,
evaluation command, and measured boundary are recorded in `TRAINING_CHANGELOG.md`.

## Implementation map

- Environment and curriculum: `shoelace_env_cfg.py`
- Reset logic: `mdp/events.py`
- Observations: `mdp/observations.py`
- Filtered Cartesian actions: `mdp/actions.py` and `mdp/actions_cfg.py`
- Dense and event rewards: `mdp/rewards.py`
- Success and failure terms: `mdp/terminations.py`
- RSL-RL configuration: `agents/rsl_rl_ppo_cfg.py`
- Scripted expert diagnostics: `scripts/demos/shoelace_open_loop_pull.py`
- Frozen 62-observation checkpoint evaluation: `scripts/demos/shoelace_eval_model20.sh`
- Detailed training record: `TRAINING_CHANGELOG.md`
