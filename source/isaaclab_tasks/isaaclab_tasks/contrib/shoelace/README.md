<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Dual-Franka shoelace training contract

This document records the current source-level environment contract. It describes what the policy and
asymmetric critic receive, how the 14 policy outputs are interpreted, and how rewards and terminations are
computed. Command-line overrides can change these defaults; the source of truth for a saved run remains its
`params/env.yaml`, `params/agent.yaml`, and recorded Git diff.

## Runtime context

| Setting | Current source default |
|---|---:|
| Task | `IsaacContrib-Shoelace-DualFranka` |
| Workflow | Manager-based RL environment |
| RL library | RSL-RL PPO |
| Physics | Newton with one-way proxy coupling |
| Environments | 32 |
| Physics timestep | `1/120 s` |
| Control decimation | 4 |
| Policy timestep, `dt` | `1/30 s` |
| Episode duration | `20 s`, or 600 policy steps |
| Grasp assistance | Disabled |
| PPO rollout horizon | 16 policy steps per environment |
| PPO checkpoint interval | 50 iterations |
| PPO configured maximum | 1,000 iterations |

Setting `env.coupling_mode=admm` switches the runtime to symmetric ADMM two-way coupling. The current
ADMM defaults are iterations `5`, rho `400`, gamma `7.5e-5`, Baumgarte `0.75`, and rigid-contact matching
`latest`.

The actor and critic both use empirical observation normalization. The actor is an MLP with hidden
dimensions `[256, 128]` and a 14-dimensional Gaussian output initialized with action standard deviation `0.3`.
The critic has the same hidden dimensions and receives one additional privileged observation.

## Observation design

Observation corruption is disabled and terms are concatenated in the order below.

### Actor observation: `policy`, 62 dimensions

| Order | Term | Shape | Scale | Meaning |
|---:|---|---:|---:|---|
| 0 | `left_joint_pos` | 7 | 1 | Left arm joint positions relative to the configured default positions [rad]. |
| 1 | `right_joint_pos` | 7 | 1 | Right arm joint positions relative to the configured default positions [rad]. |
| 2 | `left_joint_vel` | 7 | 0.05 | Left arm joint velocities [rad/s]. |
| 3 | `right_joint_vel` | 7 | 0.05 | Right arm joint velocities [rad/s]. |
| 4 | `left_finger_pos` | 1 | 25 | Driven left finger position relative to its default position [m]. |
| 5 | `right_finger_pos` | 1 | 25 | Driven right finger position relative to its default position [m]. |
| 6 | `tails_to_tcp` | 6 | 10 | Assigned tail-to-TCP vectors, one XYZ vector per arm, expressed in each controlling robot's root frame [m]. |
| 7 | `tails_to_knot` | 6 | 10 | Tail positions relative to the knot center, two XYZ vectors in environment axes [m]. |
| 8 | `tail_velocities` | 6 | 1 | Linear velocities of the two free-tail regions in environment axes [m/s]. |
| 9 | `last_action` | 14 | 1 | Previous executed EMA arm commands and raw binary-gripper actions. |

Dimension check:

```text
4 * 7 arm states + 2 * 1 finger states + 3 * 6 lace states + 14 previous actions = 62
```

The actor does **not** directly observe contact forces, a Boolean grasp flag, throat density, episode
phase, or the reference pull directions. It must infer grasping from tail/TCP geometry, finger position,
lace motion, and action history.

### Critic observation: 63 dimensions

The asymmetric critic receives the complete 62-dimensional `policy` group plus:

| Term | Shape | Meaning |
|---|---:|---|
| `throat_density` | 1 | Fraction of free shoelace segments less than `0.025 m` from the knot center. |

This privileged scalar affects value estimation only; it is not available to the deployed actor.

## Action design

The runner clips raw policy actions to `[-1, 1]`. The 14 values are ordered as:

```text
[left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
```

### Arm actions

Each arm uses a six-dimensional relative Cartesian pose command:

```text
[dx, dy, dz, dRx, dRy, dRz]
```

| Component | Scale for raw action `1` | Controller |
|---|---:|---|
| Translation | `0.005 m` per policy step | Differential IK |
| Rotation | `0.01 rad` per policy step | Differential IK |

The IK controller uses damped least squares with `lambda=0.01` and controls the `panda_hand` frame with
the configured TCP offset. Commands are relative to the current TCP pose and are converted to targets for
the seven Panda arm joints.

Before scaling, each arm command uses an exponential moving average with newest-command weight `alpha=0.75`:

```text
arm_executed[t] = 0.25 * arm_executed[t - 1] + 0.75 * arm_raw[t]
```

The first command after each individual environment reset is unfiltered. The policy's `last_action` observation
contains `arm_executed`, not `arm_raw`, so its recurrent action history matches the command received by the IK
controller. Setting both arm-action `alpha` values to `1.0` restores the former unfiltered contract.

### Gripper actions

Each gripper contributes one continuous policy output but is mapped to a binary joint-position target:

| Raw action | Command | Driven finger target |
|---:|---|---:|
| `< 0` | Close | `0.0015 m` |
| `>= 0` | Open | `0.01 m` |

Only `panda_finger_joint1` is directly actuated by the action term. The second finger is configured as a
passive joint for the Newton articulation.

## Shared grasp and task geometry

Several reward and termination terms reuse the following state:

- `d_i`: distance from tail region `i` to its assigned TCP.
- `q_i`: driven finger-joint position for arm `i`.
- `C`: number of free shoelace segments within `0.025 m` of the knot center.
- `D_i`: distance from tail `i` to the knot center.
- `S`: distance between the two tail regions.

A strict per-tail acquisition is inferred when both conditions hold:

```text
d_i <= distance threshold
0.0003 m <= q_i <= 0.0025 m
```

The `0.0015 m` close target lies inside this strict interval. Acquisition, dense task shaping, and lost-grasp
retention all keep the `0.0025 m` upper bound. Success alone permits finger rebound up to `0.0035 m`, and only
while the tail remains within the tighter `0.015 m` success distance and every untying margin is complete. This
accommodates the `0.003 m` shoelace contact thickness under load without weakening training-time grasp shaping.
The lower bound rejects over-closed states below `0.0003 m`; tail-to-TCP proximity is always required, so aperture
alone is not treated as a grasp.

## Reward design

Isaac Lab evaluates each reward as:

```text
step_reward_i = raw_term_i * configured_weight_i * dt
total_step_reward = sum(step_reward_i)
```

where `dt = 1/30 s`. TensorBoard `Episode_Reward/<term>` values are the mean episodic sums divided by the
configured `20 s` maximum episode duration; they are therefore not raw term outputs or unnormalized
episode returns.

| Term | Weight | Active phase | Effective behavior |
|---|---:|---|---|
| `dense_task` | `+10` | Phase-dependent | Reset-relative approach, acquisition, task progress, and synchronized pull shaping. |
| `grasp_acquisition` | `+10` | Once per episode | Gives exactly `+10` when both strict grasps satisfy the `0.012 m` threshold. A valid grasp already present at reset seeds the latch without earning this bonus. |
| `success` | `+60` | Terminal event | Gives exactly `+60` when the `success` termination fires. |
| `failure` | `-12` | Terminal event | Gives exactly `-12` when either `unsafe` or `lost_grasp` fires. Timeout is not penalized by this term. |
| `action_rate` | `-0.01` | Always | Penalizes the squared difference between consecutive raw actions. |
| `left_joint_velocity` | `-1e-4` | Always | Penalizes the sum of squared left-arm joint velocities, with the raw penalty capped at 100. |
| `right_joint_velocity` | `-1e-4` | Always | Penalizes the sum of squared right-arm joint velocities, with the raw penalty capped at 100. |

### `dense_task`

The dense term is a rate-based, stateful reward. Its previous potentials and reset geometry are cleared for
the relevant environment IDs at every episode reset.

Before bilateral grasp acquisition it uses two potentials:

```text
reach_i = 1 - tanh(d_i / 0.05)
proximity_i = 1 - tanh(d_i / 0.012)
```

The finger aperture score is one throughout the strict grasp interval and falls toward zero outside it:

```text
aperture_i = clamp(min(q_i / 0.0003, (0.01 - q_i) / (0.01 - 0.0025)), 0, 1)
```

Reach, proximity, aperture, and the two arms are combined using the Hamacher soft-AND:

```text
H(a, b) = a*b / (a + b - a*b + epsilon)
```

This produces an approach potential `A` and a bilateral acquisition potential `G`. Their finite-difference
rates are clipped to `[-3, 3] 1/s`.

At reset, the dense term checks the `0.020 m` distance with the strict `0.0025 m` aperture and seeds its
bilateral-acquisition latch if both tails are already grasped. Otherwise, one policy step satisfying that condition
records bilateral acquisition and enables task progress. Task and pull shaping continue only while this same
strict aperture remains valid. The task potential `P` is measured relative to the reset state and combines four
normalized margins:

1. reduction in knot-throat segment count toward `C <= 52`;
2. left-tail distance progress toward `D_left >= 0.09 m`;
3. right-tail distance progress toward `D_right >= 0.09 m`; and
4. tail-separation progress toward `S >= 0.18 m`.

The four margins are clipped to `[0, 1]`. Their arithmetic mean supplies 75% of `P`; a shifted soft minimum with
temperature `0.05` supplies the remaining 25%:

```text
P = 0.75 * mean(progress margins) + 0.25 * shifted_soft_min(progress margins)
```

Consequently `P=0` at reset, individual margins can provide progress before every margin moves, and the
weakest margin still influences the combined score.

The unweighted dense output is:

```text
before acquisition:
    0.10 * dA/dt + 0.25 * dG/dt

while the bilateral grasp is currently retained:
    1.00 * dP/dt
    + 0.25 * tanh(min_i(dot(tail_velocity_i, pull_direction_i)) / 0.04)
```

Using the minimum projected velocity requires both tails to move synchronously along their reference pull
directions. Pulling backward produces negative credit. Task-progress and pull credit are disabled whenever
the current strict bilateral grasp is absent.

### Event rewards

`grasp_acquisition`, `success`, and `failure` return an impulse divided by `dt`. RewardManager multiplies
them by `dt` again, so their configured weights are their exact per-event returns. An initial grasp detected
during reset seeds the acquisition latch and does not emit an acquisition event:

```text
new bilateral acquisition after reset: +10 once
bilateral grasp already valid at reset: 0
successful termination: +60 once
unsafe, lost-grasp, missed-grasp, or timeout termination: -12 once
failure concurrent with success: 0 failure reward
```

## Termination design

### `success`

An episode succeeds only when all conditions are simultaneously true:

```text
C <= 52
min(D_left, D_right) >= 0.09 m
S >= 0.18 m
both tail-to-TCP distances <= 0.015 m
both finger positions in [0.0003, 0.0035] m
state is not unsafe
```

Success is a normal terminal condition, not a timeout, and produces the `+60` event reward.

### `unsafe`

An episode terminates as unsafe if any of the following occurs:

- shoelace position or velocity contains a non-finite value;
- either robot's joint position, joint velocity, body position, or body orientation contains a non-finite value;
- the minimum shoelace segment height is below `-0.003 m`; or
- the maximum distance of any shoelace segment from the shoelace mean position exceeds `0.6 m`.

Unsafe termination produces the `-12` failure event reward.

### `lost_grasp`

Without grasp assistance, acquisition and retention are inferred geometrically:

1. At reset, any tail already satisfying the `0.012 m` acquisition distance and strict finger aperture seeds
   its acquisition state; an initially valid bilateral grasp seeds the bilateral latch.
2. After reset, one policy step within `0.012 m` and the strict finger aperture confirms acquisition.
3. The term remembers per-tail acquisition and whether both tails have ever been acquired simultaneously.
4. After bilateral acquisition, retention permits distance up to `0.020 m` while keeping the strict finger range.
5. A release candidate must remain true for 6 consecutive policy steps (`0.2 s`) before the episode
   terminates. Any recovered retention within that window clears the release counter.

Non-finite tail-to-TCP distances also trigger this termination. Lost-grasp termination produces the `-12`
failure event reward.

### `missed_grasp`

Once both tails simultaneously enter the `0.012 m` acquisition range, the task allows eight policy steps to
complete bilateral acquisition. This trigger is latched, so opening the grippers and moving a tail back out of
range cannot evade the deadline. The term is disabled permanently after bilateral acquisition; later release is
handled by `lost_grasp`. It therefore does not penalize the approach phase before both tails are reachable.

### `time_out`

The episode ends after 600 policy steps (`20 s`). The task uses a finite-horizon contract, so RSL-RL does not
bootstrap the critic at this boundary. A timeout also produces the `-12` exclusive failure reward unless success
occurs on the same step.

## Reset and curriculum context

Every reset restores the authored cable pose and zero cable velocity. The 74-level curriculum then applies
calibrated arm and gripper states:

- levels 0--68 keep both arms at the settled grasp pose while the gripper target moves from `0.002` to the fully
  open `0.01 m` state;
- the contact-release intervals around `0.0035--0.003625 m` and `0.004--0.004328 m` use measured micrometre-scale
  reset increments;
- six measured bridge states between `0.005` and `0.006 m` cover the next post-acquisition success frontier;
- levels 69--73 keep the gripper fully open and progressively move the arms through calibrated approach poses
  toward the full pre-grasp pose.

The curriculum begins at level 0 and promotes after one qualifying window with at least 128 episodes and a
success rate of at least 0.5. Reward and lost-grasp terms inspect the post-reset geometry to seed
their stateful grasp latches, so an initially retained grasp is treated as task state rather than a newly earned
acquisition event.

## Phase and information flow

```text
reset
  -> approach both tails
  -> preserve an initial grasp or acquire both strict grasps
  -> retain both grasps while pulling in the reference directions
  -> reduce throat occupancy and increase both tail distances and separation
  -> success, missed/lost-grasp or unsafe failure, or timeout
```

The actor sees enough geometry to approach and track the tails but does not see the privileged throat-density
scalar, the complete shoelace centerline, safety geometry summaries, or explicit grasp/contact flags. Reward
and termination logic therefore use state that the actor cannot observe directly. In particular,
post-acquisition task shaping is unavailable until the stateful bilateral-grasp gate has been crossed, and
the actor cannot directly distinguish different internal knot shapes that share the same tail state.

## Implementation references

- Environment configuration: `shoelace_env_cfg.py`
- Shoelace observations: `mdp/observations.py`
- Shared grasp and geometry utilities: `mdp/utils.py`
- Stateful dense and event rewards: `mdp/rewards.py`
- Success, unsafe, and lost-grasp terminations: `mdp/terminations.py`
- RSL-RL PPO configuration: `agents/rsl_rl_ppo_cfg.py`

## Training and troubleshooting log

Training decisions are based on task metrics rather than total return alone. In particular, a lower aggregate
success rate immediately after a curriculum promotion is expected because the reset distribution has moved to a
harder frontier. The primary signals are curriculum level and exposure, frontier-window success, bilateral grasp
acquisition, success, lost grasp, timeout, and unsafe termination.

### Reset-relative reward and dense gripper curriculum

The original absolute-state dense task reward could increase without producing the missing tail-untying behavior.
It also did not distinguish progress already present in an authored reset from progress caused by the policy. The
replacement reward records reset geometry, pays bounded potential differences, and gates task/pull credit behind a
confirmed bilateral strict grasp. The acquisition event now requires both tails simultaneously and seeds its latch
from an initially grasped reset, which prevents reset-and-drop reward farming. Lost grasp is likewise seeded from
reset and requires a six-step release window so that one-frame contact fluctuations do not terminate an episode.

The first reset gripper positions were changed from `0.000`, `0.0025`, `0.005`, and `0.0075 m` to `0.002`, `0.003`,
`0.004`, and `0.006 m`. This preserves a physical lace aperture at level 0 and makes the transition from retained
grasp to policy-acquired grasp denser.

Two fresh 8-GPU, 128-environment-per-rank, seed-42 validation runs isolated the curriculum-density change:

| Run | Early gripper levels [m] | First success | Promotions | Final success | Final lost grasp | Final timeout | Final acquisition reward |
|---|---|---:|---|---:|---:|---:|---:|
| `2026-09-04_17-14-22_admm_rho400_reset_latch_seeded_blended_dense_8gpu_300_fresh` | `0.002/0.004/0.006/0.008` | 92 | level 1 at 174 | 32.91% | 53.61% | 13.48% | 0.0313 |
| `2026-09-04_18-09-31_admm_rho400_dense_gripper_curriculum_8gpu_300_fresh` | `0.002/0.003/0.004/0.006` | 78 | level 1 at 143; level 2 at 192 | 24.37% | 70.95% | 4.69% | 0.2875 |

The denser run reached one additional level within the same 300 iterations and produced substantially more
bilateral acquisitions. Its lower final success and higher lost-grasp rate were measured after promotion to the
harder level 2 distribution and therefore are not evidence of regression.

### Production run: 20,000 iterations

The production run uses ADMM two-way coupling and starts from a fresh policy:

```bash
uv run --frozen isaaclab train_multigpu \
  --rl_library rsl_rl \
  --num_gpus 8 \
  --max_iterations 20000 \
  --viz none \
  --seed 42 \
  --tee 3 \
  --log_dir logs/torchrun/shoelace_admm_rho400_dense_gripper_curriculum_8gpu_20000_fresh_retry1 \
  --task IsaacContrib-Shoelace-DualFranka \
  --num_envs 128 \
  --run_name admm_rho400_dense_gripper_curriculum_8gpu_20000_fresh_retry1 \
  env.coupling_mode=admm
```

The first launch aborted on rank 0 while Newton was replicating the scene with glibc allocator errors
(`corrupted double-linked list` and `unaligned tcache chunk`). The other seven ranks completed scene creation,
which localized the failure to native scene initialization rather than PPO or NCCL. All processes and GPU memory
were released, and an unchanged retry started successfully. The failed launch logs remain under
`logs/torchrun/shoelace_admm_rho400_dense_gripper_curriculum_8gpu_20000_fresh`; the successful retry is under the
`..._retry1` directory from the command above.

RSL-RL checkpoints save the policy, optimizer, normalization state, and learning iteration, but not Isaac Lab
environment-manager state. Loading a checkpoint therefore reconstructs `PullToGraspCurriculum` from its configured
`initial_level`; it does not restore the frontier level, exposure stage, or partial success window. After an
interruption, inspect the last integer `current_level` in TensorBoard and explicitly override `initial_level` when
resuming an unchanged reward/observation/action contract. A resume adds the configured number of learning
iterations to the checkpoint iteration. Do not resume across a reward or policy contract change; begin a fresh run
instead.

At iteration 164, the retry had completed 2,703,360 environment transitions at approximately 2,220 steps/s. All
logged scalars were finite; unsafe and timeout termination were zero. The first success occurred at iteration 80,
and the latest success/lost-grasp rates were 66.26%/33.74%, with a ten-iteration mean success rate of 64.92%. The
frontier remained at level 0 with maximum 50% exposure because its latest completed 128-episode window success was
67.69%, below the 70% promotion threshold. This is a delayed but valid progression relative to the validation run,
not a curriculum-state mismatch: the saved environment and agent configurations differ only in run name and
maximum iteration count.

At iteration 177, all distributed ranks had promoted to level 1 after the frontier-window success rose to 77.34%.
The iteration-179 success/lost-grasp rates were 79.93%/20.07%. This first promotion occurred 34 iterations later
than in the short validation run but required no threshold or reward change, confirming that the scheduler was
waiting for evidence rather than stuck.

By iteration 189, the bilateral acquisition reward was nonzero at 0.0818, with a ten-iteration mean of 0.0810.
This confirms that level-1 episodes starting from a `0.003 m` gripper position were actively reacquiring both tails
rather than relying only on the retained level-0 reset. The corresponding ten-iteration success/lost-grasp means
were 80.75%/19.25%, with no timeout or unsafe termination.

Level 1 reached 50% exposure at iteration 208 and recorded its first qualifying promotion window at iteration 219
with 89.23% frontier success. All ranks reached level 2 at iteration 221 after the next window reported 93.13%.
At iteration 227, bilateral acquisition was 0.2755 and the ten-iteration success/lost-grasp means were
89.92%/10.08%. The production run therefore reproduced the short validation run's first two promotions, although
level 1 and level 2 were reached 34 and 29 iterations later, respectively.

The first two completed level-2 frontier windows reported only 5.47% success at iteration 328 and 3.10% at
iteration 455, so the curriculum correctly stayed at its minimum 20% frontier exposure. This low frontier result
must not be confused with the replay-dominated aggregate success metric. Bilateral acquisition remained near 0.30,
while lost-grasp dominated completed episodes, identifying the active gap as post-acquisition retention and task
completion rather than initial grasp acquisition. Aggregate success nevertheless recovered from a ten-iteration
mean of 18.90% at iteration 367 to 55.67% at iteration 460. Because a 128-episode frontier window takes more than
100 iterations when only 20% of slots run long level-2 episodes, the next decision point is a later frontier window,
not a transient aggregate-reward fluctuation.

A third level-2 frontier window reached only 6.25% near iteration 572 even though replay-dominated aggregate
success had recovered above 60% and bilateral acquisition remained near 0.30. Training was intentionally stopped
at iteration 576 for a fixed-frontier state trace. The last periodic checkpoint before shutdown is `model_550.pt`.

### Level-2 success and dense-margin diagnosis

A deterministic 32-environment rollout of `model_550.pt` fixed every reset at level 2 and recorded acquisition,
gripper actions, finger positions, tail-to-TCP distances, success geometry, and the exact termination. A level-1
rollout with the same checkpoint and seed provided the control:

| Configuration | Acquisition | Success | Lost grasp |
|---|---:|---:|---:|
| Level 1, original thresholds | 32/32 | 30/32 (93.8%) | 2/32 (6.2%) |
| Level 2, original thresholds | 32/32 | 13/32 (40.6%) | 19/32 (59.4%) |
| Level 2, `0.0030 m` aperture everywhere | 32/32 | 19/32 (59.4%) | 13/32 (40.6%) |
| Level 2, `0.0035 m` aperture everywhere | 32/32 | 22/32 (68.8%) | 10/32 (31.2%) |
| Level 2, strict dense/lost, `0.0035 m` success, 12-step release | 32/32 | 22/32 (68.8%) | 10/32 (31.2%) |

All level-2 episodes acquired both tails at step 9 in the original trace. The policy never issued an open command
on the left after acquisition; the mean right open-command fraction among original lost-grasp episodes was 1.4%,
and most terminal actions still commanded closure. Nevertheless, the original lost-grasp group had a right finger
position median of `0.00258 m`, just above the shared `0.00250 m` limit, while its tail often remained only about
`0.004 m` from the TCP. Successful episodes averaged `0.00246 m` on the same finger. This showed that cable-load
rebound, rather than deliberate release or missing acquisition, accumulated the six-step lost-grasp counter just
before several trajectories completed the final separation margin.

The first attempted fix widened the aperture for dense shaping, success, and lost-grasp retention to `0.0035 m`.
Although the old checkpoint then reached 20/32 success in an integration trace, a fresh 8-GPU run named
`2026-09-04_20-31-53_admm_rho400_retained_grasp_hysteresis_8gpu_300_fresh` exposed a training regression. At
iteration 100 it still had zero success, mean episode length 211, and dense task reward -0.013; the baseline at the
same iteration had 16.5% success, mean episode length 80, and dense task reward 0.363. A fixed level-0 rollout of
the new `model_100.pt` ended in 32/32 true grasp losses after a mean 237 steps, with mean throat count 79 and tail
separation only `0.118 m`. Widening retention had delayed the failure signal enough for PPO to learn static holding.
The run was stopped after saving model 100, and the wider dense/lost-grasp gate was removed.

A second fresh 8-GPU run kept the strict dense/lost aperture, permitted `0.0035 m` only at success, and extended
release confirmation from 6 to 12 steps. Dense task reward recovered to 0.409 by iteration 100, but success was
still zero. Its fixed level-0 `model_100.pt` trace showed that all 32 policies had already met the throat target
(mean 43.7 segments) and both tail-distance targets (`0.095/0.112 m`), while separation alone stalled at
`0.155 m`. The wider release window was therefore not the missing signal and was restored to 6 steps.

The next candidate kept the strict acquisition/dense/lost-grasp contract, permitted `0.0035 m` only for success,
and increased the shifted-soft-min share of reset-relative task progress from 25% to 75%. The fresh 8-GPU run
`2026-09-04_21-14-47_admm_rho400_softmin75_success3p5_8gpu_300_fresh` disproved that change. Success first appeared
at iteration 132 and remained only 3.22% at iteration 162, compared with 16.45% at iteration 100 for the 25%
baseline. A fixed level-0 rollout of model 100 ended in 32/32 lost grasps after a mean 91 steps; its right tail
remained only `0.073 m` from the knot and escaped to a mean `0.032 m` from its TCP. Model 150 still produced 32/32
deterministic lost grasps: the tail distances averaged `0.104/0.107 m`, but throat count was 63 and separation was
only `0.150 m`, with large tail-to-TCP excursions. The run was intentionally stopped at iteration 162. Raising the
soft-min share had suppressed useful independent early-margin credit and produced unstable metric switching rather
than a balanced pull.

The following candidate restored the 25% shifted-soft-min share and assigned weight 2 to separation in the 75%
arithmetic branch. The fresh run
`2026-09-04_21-41-23_admm_rho400_sep2_softmin25_success3p5_8gpu_300_fresh` still had zero success at iteration 112.
Its fixed level-0 model 100 trace ended in 32/32 lost grasps: throat count reached 48 and the right tail reached
`0.103 m`, but the left tail remained at `0.085 m` and separation regressed to `0.131 m`. Static terminal-margin
weighting had selected another asymmetric shortcut rather than teaching the TCPs to remain coupled to both moving
tails, so the added weighting API was removed.

The next candidate left the original four-margin potential intact and added a post-acquisition worst-tail
retention-potential rate. The old healthy model 100 supplied an apples-to-apples control: its seven deterministic
successes never exceeded the initial `0.0062 m` tail-to-TCP distance, whereas failed episodes commonly reached
`0.05-0.09 m`. The retention rate was intended to turn that early physical distinction into immediate credit
assignment without rewarding a static pose.

The first launch failed before PPO iteration 0 when rank 2 received `SIGSEGV` inside Newton USD parsing. All
worker processes and GPU allocations terminated, and a completely unchanged retry passed scene construction on
all eight ranks. This was a native startup failure rather than evidence about the reward. At iteration 50, a
fixed level-0 rollout of the retry ended in 32/32 lost grasps after a mean 68.7 steps, but it isolated the remaining
asymmetry: the left tail-to-TCP distance never exceeded `0.0063 m`, while the right distance reached a mean maximum
of `0.0320 m`. The equal-iteration baseline model 50 lost both sides, with mean maximum distances of
`0.1061/0.0412 m`.

The improvement did not persist. The retry still had zero aggregate success at iteration 114, compared with
16.45% at iteration 100 for the baseline. Its deterministic model 100 rollout produced 32/32 lost grasps after a
mean 157.8 steps. Across the entire trajectories, the best mean geometry was throat count 77.1, tail distances
`0.0876/0.0705 m`, and separation `0.1422 m`; none met the corresponding `52`, `0.09 m`, and `0.18 m` success
targets. At maximum separation the tails remained within `0.0045/0.0117 m` of their TCPs, but the policy later
lost the left tail and separation regressed to `0.1079 m`. The standalone retention rate had selected delayed
single-side failure rather than task completion, so training was stopped and the term was removed.

Source inspection then exposed a narrower accounting asymmetry in the original task potential. Its previous value was
updated every step, but `dP/dt` was paid only while the current strict bilateral grasp remained true. A lace
rebound during the six-step release window therefore lowered the stored potential while its negative reward was
masked; reacquisition could not recover the missing debit. The next candidate changed only that gate: positive
task progress still requires a strict bilateral grasp, while negative post-acquisition task regression is always
charged. It added no end-effector trajectory target and gave no reward for static retention.

The regression test failed under the old gate and passed under the changed gate, confirming the accounting
difference. Training nevertheless disproved it as a learning intervention. The fresh 8-GPU run
`2026-09-04_22-30-02_admm_rho400_task_regression_unmasked_8gpu_300_fresh` still had zero aggregate success when
stopped at iteration 119. Model 50 briefly approached the target with mean best throat count 55.7, tail distances
`0.1221/0.0766 m`, and separation `0.1798 m`, but model 100 regressed to 65.2, `0.1209/0.0733 m`, and `0.1736 m`.
All 32 deterministic model-100 episodes lost grasp after a mean 104.3 steps. Charging rebound regressions had
made early returns more pessimistic without resolving the under-pulled right tail, so the original gate was
restored.

The current candidate instead changes only curriculum density while retaining the known healthy baseline reward.
The old 11-level schedule jumped from `0.003` to `0.004 m` at the frontier where the production run stalled, and
also contained two identical settled-arm/open-gripper reset states. The new 14-level schedule inserts
`0.0025`, `0.0035`, `0.005`, and `0.008 m` gripper states, removes that duplicate, and preserves all five calibrated
arm-approach states. Its fresh validation command is:

```bash
uv run --frozen isaaclab train_multigpu \
  --rl_library rsl_rl \
  --num_gpus 8 \
  --max_iterations 300 \
  --viz none \
  --seed 42 \
  --tee 3 \
  --log_dir logs/torchrun/shoelace_admm_rho400_curriculum14_8gpu_300_fresh_retry1 \
  --task IsaacContrib-Shoelace-DualFranka \
  --num_envs 128 \
  --run_name admm_rho400_curriculum14_8gpu_300_fresh_retry1 \
  env.coupling_mode=admm
```

The first launch failed before PPO iteration 0: rank 4 aborted with an allocator error in Newton USD parsing,
while rank 2 segfaulted during MJWarp/Warp code generation. The launcher reaped all workers and released every
GPU allocation. The unchanged `..._retry1` command above completed scene construction and parameter
synchronization on all eight ranks, so the native startup failure is not treated as curriculum evidence.

The retry reached level 1 on every rank at iteration 176. Its aggregate success peaked at 90.28% at iteration
192, but the 128-episode level-1 frontier windows subsequently declined from 72.56% to 67.19%, 64.26%, 63.28%,
and 44.53%. Aggregate success fell to 58.54% by iteration 253 even though unsafe and timeout remained zero. The
run was stopped after `model_250.pt` rather than spending the remaining validation budget on a declining
frontier.

Fixed-level deterministic traces separated reset difficulty from policy regression:

| Checkpoint | Level 0 (`0.002 m`) | Level 1 (`0.0025 m`) | Level 2 (`0.003 m`) |
|---|---:|---:|---:|
| `model_200.pt` | 32/32 success | 32/32 success | 32/32 success |
| `model_250.pt` | 31/32 success | 19/32 success | 19/32 success |

Every level-2 trace acquired both tails at step 6, neither failed group issued an open-gripper command, and the
level-1 and level-2 outcomes were identical. The added `0.0025 m` reset was therefore not intrinsically
unsolvable, nor did it provide a distinct post-acquisition behavior. More importantly, a deterministically
perfect `model_200.pt` did not advance because stochastic training had to produce two consecutive 70% windows;
continued PPO updates then moved the policy away from that solution. The next focused candidate keeps the reward,
termination thresholds, 14 reset levels, 128-episode window, and 70% threshold unchanged, and reduces only
`promotion_window_count` from 2 to 1. A full window still measures 128 frontier episodes per rank, while avoiding
overtraining a level after a policy has already demonstrated deterministic mastery.

The single-window candidate was validated in a fresh 8-GPU run:

```bash
uv run --frozen isaaclab train_multigpu \
  --rl_library rsl_rl \
  --num_gpus 8 \
  --max_iterations 300 \
  --viz none \
  --seed 42 \
  --tee 3 \
  --log_dir logs/torchrun/shoelace_admm_rho400_curriculum14_window1_8gpu_300_fresh \
  --task IsaacContrib-Shoelace-DualFranka \
  --num_envs 128 \
  --run_name admm_rho400_curriculum14_window1_8gpu_300_fresh \
  env.coupling_mode=admm
```

All eight ranks reached levels 1, 2, and 3 at iterations 199, 248, and 290, respectively. This run learned level 0
more slowly than the two-window run, but the first complete 71.97% window promoted it before a second confirmation
could overtrain the solved policy. It then crossed level 1 with a 70.54% window and level 2 with a 79.69% window.
At iteration 299, aggregate success/lost-grasp rates were 82.28%/17.72%, dense task reward was 0.4765, acquisition
reward was 0.2578, and unsafe and timeout were both zero. The launcher exited normally after 4,915,200 environment
transitions.

A fixed-level deterministic evaluation of `model_299.pt` confirmed that the logged success represented the task
geometry and localized the next frontier:

| Reset level | Gripper position [m] | Acquisition | Success | Lost grasp | Timeout |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.0020 | 32/32 | 32/32 | 0/32 | 0/32 |
| 1 | 0.0025 | 32/32 | 32/32 | 0/32 | 0/32 |
| 2 | 0.0030 | 32/32 | 31/32 | 1/32 | 0/32 |
| 3 | 0.0035 | 32/32 | 30/32 | 2/32 | 0/32 |
| 4 | 0.0040 | 5/32 | 0/32 | 5/32 | 27/32 |

The policy therefore mastered the inserted `0.0035 m` bridge while the former `0.003` to `0.004 m` acquisition
gap remained visible at the untrained next level. This is the intended curriculum boundary rather than reward
hacking: levels 0--3 terminated on the exact success geometry, while level 4 mostly failed to acquire and timed
out. The validated single-window scheduler is used for the next fresh 20,000-iteration production run:

```bash
uv run --frozen isaaclab train_multigpu \
  --rl_library rsl_rl \
  --num_gpus 8 \
  --max_iterations 20000 \
  --viz none \
  --seed 42 \
  --tee 3 \
  --log_dir logs/torchrun/shoelace_admm_rho400_curriculum14_window1_8gpu_20000_fresh \
  --task IsaacContrib-Shoelace-DualFranka \
  --num_envs 128 \
  --run_name admm_rho400_curriculum14_window1_8gpu_20000_fresh \
  env.coupling_mode=admm
```

The production launch completed scene creation and parameter synchronization on all eight ranks, then entered
PPO normally. Its RSL-RL artifacts are under
`logs/rsl_rl/shoelace_dual_franka/2026-09-05_00-12-25_admm_rho400_curriculum14_window1_8gpu_20000_fresh` and its
per-rank launcher logs are under the `logs/torchrun` directory from the command above.

The first strict success occurred at iteration 86. At iteration 105, aggregate success/lost-grasp rates were
39.70%/61.08%, dense task reward was 0.5261, mean reward was 28.89, and unsafe and timeout were both zero. The
production run therefore entered successful level-0 learning faster than the preceding validation run while
retaining the same saved environment and agent contract.

All ranks reached level 1 at iteration 132 after the completed level-0 frontier window reported 73.64% success.
Level-1 exposure increased from 20% to 35% at iteration 155 and to its 50% maximum at iteration 167. A 92.19%
frontier window then promoted every rank to level 2 by iteration 175. At iteration 181, aggregate success/lost-grasp
rates were 95.46%/4.54%, dense task reward was 0.5178, and bilateral acquisition reward was 0.1182, with no unsafe
or timeout termination. The nonzero acquisition term confirms active reacquisition from the `0.003 m` level-2
reset. The aggregate success still contains replay from levels 0 and 1, so level-2 mastery remains unproven until
its own completed frontier windows raise exposure and pass the 70% promotion threshold.

Level 2 reached 35% exposure at iteration 195 after a 93.75% frontier window and reached its 50% maximum at
iteration 205. Increasing the difficult-reset share initially reduced its completed maximum-exposure windows to
41.09% and 42.19%. The later windows recovered through 42.97%, 52.31%, and 60.77% before crossing the promotion
threshold at 71.09%; all ranks reached level 3 at iteration 251. Bilateral acquisition remained nonzero throughout
the temporary regression, which distinguished ongoing level-2 learning from an acquisition-dead policy. Allowing
the improving frontier to finish its windows, rather than reacting to the replay-mixed aggregate-success minimum,
preserved a valid curriculum transition. The transition checkpoint is `model_250.pt`.

Level 3 then advanced without a regression. Its 79.69% window raised exposure to 35% at iteration 272, its 89.06%
window raised exposure to 50% at iteration 284, and a 93.08% maximum-exposure window moved all ranks to level 4
at iteration 292. Aggregate success was 87.21%, bilateral acquisition reward was 0.3776, and unsafe termination
remained zero at that transition. Timeout first became measurable at 0.54% as the `0.004 m` level-4 reset entered
the distribution. This is the previously localized acquisition boundary; its completed frontier windows and
acquisition trend, rather than the replay-dominated aggregate success, determine whether another intervention is
needed.

The level-4 distribution did not recover. Training was stopped intentionally at iteration 420 after saving
`model_400.pt`: over the final ten iterations, aggregate bilateral acquisition fell to 9.38%, success to 47.29%,
and timeout rose to 39.27%, while unsafe termination remained zero. The last level-3 frontier success metric stayed
at 93.08% because the timeout-heavy 20% level-4 stratum had not yet completed its first 128-episode window. Waiting
for that stale scheduler metric would have required roughly another timeout wave while the directly observed
acquisition signal was already collapsing.

Fixed-level deterministic traces distinguished a reset cliff from general policy forgetting:

| Checkpoint | Reset | Acquisition | Success | Lost grasp | Timeout |
|---|---|---:|---:|---:|---:|
| `model_400.pt` | level 3, `0.0035 m` | 32/32 | 30/32 | 2/32 | 0/32 |
| `model_300.pt` | level 4, `0.0040 m` | 5/32 | 0/32 | 5/32 | 27/32 |
| `model_400.pt` | level 4, `0.0040 m` | 0/32 | 0/32 | 0/32 | 32/32 |

The trained policy retained 93.8% deterministic level-3 success, but another 100 PPO updates reduced level-4
acquisition from 15.6% to zero. This rules out an overly strict success threshold as the primary failure: most
level-4 episodes never crossed the bilateral acquisition gate at all.

A one-variable sweep kept `model_300.pt`, the settled arm pose, reward, termination, seed, and physics unchanged
while varying only the reset finger position:

| Reset finger position | Acquisition | Success | Lost grasp | Timeout |
|---:|---:|---:|---:|---:|
| `0.003625 m` | 32/32 | 28/32 | 4/32 | 0/32 |
| `0.003750 m` | 32/32 | 0/32 | 32/32 | 0/32 |
| `0.003875 m` | 32/32 | 0/32 | 32/32 | 0/32 |
| `0.004000 m` | 5/32 | 0/32 | 5/32 | 27/32 |

Every acquired trajectory continued commanding both grippers closed. The transition is therefore a physical
contact-release and right-tail retention boundary, not deliberate gripper opening. The curriculum now inserts
`0.003625`, `0.003750`, and `0.003875 m` levels between `0.0035` and `0.0040 m`, increasing the total from 14 to
17 levels while preserving all later gripper states and the five calibrated arm-approach states. No reward,
termination, PPO, or coupling parameter changed. Because the reset distribution changed, the next validation and
production runs must start from fresh policies rather than resume `model_400.pt`.

The first fresh 17-level validation reached levels 1, 2, 3, 4, and 5 at approximately iterations 105, 141,
177, 207, and 232. The new `0.003625 m` level raised exposure from 20% to 35% with a 51.56% frontier window,
reached 50% exposure with 63.85%, and promoted with 72.52%. This eliminates the former dead transition: no
timeout occurred while level 4 was active, and acquisition reward remained nonzero. At level 5 (`0.003750 m`),
frontier success improved from 14.06% to 26.56%, but the mean policy then regressed. Aggregate success fell from
74.27% at iteration 250 to 33.98% at iteration 299, while timeout rose to 6.25%; the run completed normally after
300 iterations and 4,915,200 environment transitions.

Fixed-level traces showed that `model_250.pt` had already learned useful deterministic behavior: it achieved
32/32 success at level 3 and 22/32 at level 5. By `model_299.pt`, level 3 had regressed to 32/32 lost grasp,
level 0 reached only 7/32 success, level 4 reached 12/32, and level 5 produced 4/32 acquisitions followed by
4 lost grasps and 28 timeouts. The collapse was therefore continued stochastic PPO drift, not an unsolvable
level-5 reset or a success-threshold bug. Adaptive learning rate was already at its `1e-5` floor during most of
the regression, so a learning-rate explosion was also excluded.

The actor normalizer changed substantially over the same interval, so crossed checkpoints isolated that possible
confounder. `model_299.pt` actor weights combined with the `model_250.pt` normalizer still acquired level 3 in
32/32 episodes but lost grasp in all 32. Conversely, `model_250.pt` weights combined with the `model_299.pt`
normalizer retained 32/32 level-3 success. The regression therefore resides in the learned actor weights rather
than the online observation-normalization buffers.

The stochastic training/evaluation mismatch localized the next intervention. With `model_250.pt` fixed at level
5, deterministic inference reached 22/32 success. Stochastic inference using the checkpoint's approximately
`0.229` action standard deviation reached only 9/32 success; forcing standard deviations `0.20`, `0.15`, `0.10`,
and `0.05` produced 10/32, 15/32, 21/32, and 21/32 success, respectively. The `0.10` setting also increased
bilateral acquisition from 81.25% to 90.63%. Exploration noise was therefore physically perturbing the
contact-sensitive policy and keeping stochastic frontier windows below deterministic capability, which caused
unnecessary updates after a usable mean policy had formed.

A fresh eight-GPU validation then changed only actor `init_std` from `0.3` to `0.1`. It remained at level 0 and
was stopped after `model_100.pt`: strict success stayed at zero, dense task reward peaked near `0.204` before
falling to `0.051`, and the deterministic checkpoint acquired both tails in 32/32 episodes but lost grasp in all
32. Lower noise protected established contact in the fixed-checkpoint sweep but removed too much early
exploration for learning the bilateral pull from scratch. The failed setting was reverted to `0.3`; reward,
termination, curriculum, horizon, and other PPO settings were unchanged by this ablation.

A fresh validation retained `init_std=0.3` and changed the Gaussian output to state-dependent log-standard
deviation. The learned mean standard deviation fell from `0.30` to `0.20` by iteration 102, but strict success
remained zero and dense task reward had receded from its local peak to `0.090`. A deterministic `model_100.pt`
trace acquired both tails in all 32 level-0 episodes but lost grasp in all 32; mean maximum tail separation was
only `0.1469 m` and mean minimum throat count remained `70.9`. The state-dependent head therefore reduced noise
before the mean policy had discovered the complete bilateral pull, reproducing the low-initial-noise failure.
This candidate was stopped and rejected.

The next validation instead resumes the compatible state-independent `model_250.pt`, whose mean policy already
reached 22/32 level-5 success, lowers only its action standard deviation to `0.1`, clears the stale optimizer
moments for that parameter, and restarts the environment curriculum at level 5. Resume is valid here because the
observation, action, reward, termination, reset-level, and network contracts are unchanged. This separates early
exploration, which needs the original `0.3` noise, from contact-sensitive consolidation after a viable mean policy
has formed.

The low-noise continuation promoted all ranks from level 5 to level 6 after 12 updates. Its completed level-5
frontier window reached 73.48% success while the mean action standard deviation remained `0.10`. Training was
stopped at `model_300.pt` for fixed-level evaluation after the first level-6 frontier windows settled near 40%
success. The deterministic comparison confirmed real learning without replay-level forgetting:

| Checkpoint | Reset | Acquisition | Success | Lost grasp | Timeout |
|---|---|---:|---:|---:|---:|
| original `model_250.pt` | level 5, `0.003750 m` | 32/32 | 22/32 | 10/32 | 0/32 |
| continued `model_300.pt` | level 5, `0.003750 m` | 32/32 | 23/32 | 9/32 | 0/32 |
| original `model_250.pt` | level 6, `0.003875 m` | 32/32 | 0/32 | 32/32 | 0/32 |
| continued `model_300.pt` | level 6, `0.003875 m` | 32/32 | 8/32 | 24/32 | 0/32 |

This rejects both an overly strict success threshold and old-level catastrophic forgetting as the immediate
level-6 cause: the policy always acquired both tails, learned nonzero level-6 success, and preserved level 5.
The remaining bottleneck is contact retention after acquisition. A 20,000-update continuation now starts from
that `model_300.pt` checkpoint at curriculum level 6. Its RSL-RL artifacts are under
`logs/rsl_rl/shoelace_dual_franka/2026-09-05_03-32-02_admm_rho400_curriculum17_resume_m300_std0p1_level6_8gpu_20000`
and its launcher logs are under
`logs/torchrun/shoelace_admm_rho400_curriculum17_resume_m300_std0p1_level6_8gpu_20000`.

That continuation was paused at `model_350.pt` after the online level-6 frontier window rose from approximately
40% to 49% and then fluctuated back to 35%. A fixed deterministic level-6 trace resolved the ambiguity:
`model_350.pt` acquired both tails in 32/32 episodes and succeeded in 13/32, compared with 8/32 for
`model_300.pt`. The mean policy therefore continued improving despite the noisy frontier estimate. Training
resumed from `model_350.pt` for the remaining 19,950 updates, preserving the original target of iteration 20,300.
The resumed artifacts are under
`logs/rsl_rl/shoelace_dual_franka/2026-09-05_03-41-16_admm_rho400_curriculum17_resume_m350_std0p1_level6_8gpu_remaining19950`
and `logs/torchrun/shoelace_admm_rho400_curriculum17_resume_m350_std0p1_level6_8gpu_remaining19950`.

Two preceding launches failed before agent preparation with a native Newton `SIGSEGV` in
`newton._src.utils.import_usd.parse_usd`, first on rank 2 and then rank 4. All sibling workers and GPU allocations
were cleaned before retrying, and the identical third launch completed scene creation. These failures did not
execute an RL update and are treated as intermittent multi-process USD-import startup failures rather than
training regressions.

Small differences between repeated fixed-seed counts come from contact-solver variation across fresh Newton
processes. Stopped checkpoints are not resumed across reward-contract changes.

Monitoring should intervene only with phase-specific evidence. A high reward without success is insufficient. At
each frontier, inspect whether its completed-window success reaches 70%, whether the curriculum advances after one
qualifying window at maximum exposure, and whether acquisition becomes nonzero after resets cease to begin in a
strict grasp. If a frontier stalls, replay a checkpoint deterministically and record TCP/tail distances, finger
positions, geometric success margins, per-term reward, and the exact termination before changing one training
variable.

### Exclusive failure reward and 19-level fresh validation

The next reward-contract change made success and failure terminal rewards mutually exclusive: a transition that
meets `success` no longer receives the `lost_grasp` failure penalty on the same step. Two reset finger positions,
`0.0038125 m` and `0.0039375 m`, were also inserted around the remaining contact-release boundary, increasing the
curriculum from 17 to 19 levels. Because both the reward and reset distribution changed, validation restarted from
a randomly initialized policy. The implementation, focused tests, and changelog were committed as
`caa99e6ce` (`Stabilize shoelace contact-release curriculum`).

The fresh eight-GPU validation used 128 environments per rank and completed 300 iterations under
`logs/rsl_rl/shoelace_dual_franka/2026-09-05_04-43-01_admm_rho400_curriculum19_exclusive_failure_fresh_8gpu_300`.
It advanced to levels 1, 2, 3, and 4 at approximately iterations 150, 189, 220, and 250. At iteration 299 the
aggregate success rate was 73.19%, the current level-4 completed-window success was 66.67%, mean reward was
60.10, and unsafe termination remained zero. A successful transition's failure reward was exactly zero, directly
validating the exclusive terminal-reward contract. Fractional logged levels immediately before promotion are the
mean of asynchronously updated per-rank curriculum states, not fractional reset indices.

Fixed-level deterministic evaluation separated policy capability from stochastic rollout noise. `model_299.pt`
reached 32/32 level-4 successes in 42--43 steps, with bilateral acquisition in all episodes and no failure
penalty. At the unseen level 5 it acquired both tails in all 32 episodes but reached 0/32 successes and terminated
with lost grasp in all 32; mean maximum tail separation was about `0.154 m`, below the `0.18 m` success threshold.
The 300-iteration run therefore learned the complete current frontier but had not yet trained on level 5. Its
approximately `0.22` stochastic action standard deviation explains why the online promotion window lagged the
deterministic actor, but earlier low-noise ablations show that globally reducing exploration is not a justified
standalone fix.

The first launch attempt also exposed a launcher mismatch: `--headless` is no longer consumed by the current
training CLI and leaked into Hydra composition, whereas the supported non-rendering form is `--viz none`. Two
subsequent starts encountered the known intermittent native Newton crash in `newton._src.utils.import_usd.parse_usd`;
the identical retry completed normally and no failed attempt performed an RL update. The production continuation
must remain a fresh run because of the changed reward contract, and should be judged at each newly reached
frontier rather than by aggregate replay-level success alone.

The fresh 20,000-iteration production run started successfully with 128 environments on each of eight GPUs. Its
artifacts are under
`logs/rsl_rl/shoelace_dual_franka/2026-09-05_05-25-34_admm_rho400_curriculum19_exclusive_failure_fresh_8gpu_20000`
and its launcher logs are under
`logs/torchrun/shoelace_admm_rho400_curriculum19_exclusive_failure_fresh_8gpu_20000`. At the first saved
checkpoint, `model_50.pt`, the run remained at level 0 as expected: by iteration 55 mean episode length had risen
from about 6 to 68.2, dense task reward from approximately zero to 0.366, mean reward from -12.0 to -5.50, and
unsafe and timeout terminations remained zero. Strict success and acquisition had not yet appeared; the preceding
fresh 300-iteration validation first showed success near iteration 73, so this is a healthy contact-retention
learning phase rather than evidence of a stall.

Strict success first appeared at iteration 89 in the production run and rose to 26.56% by iteration 104, with a
22.66% completed frontier window. Mean reward reached 21.37 while timeout and unsafe termination stayed at zero;
the success and lost-grasp fractions summed to one. This run discovered success 16 iterations later than the
300-iteration validation, but its success rate then grew substantially faster, supporting contact-solver and PPO
trajectory variation rather than a reward regression.

At iteration 115, success and lost-grasp telemetry briefly summed to more than one. This is an intentional
predicate overlap rather than a PPO or scheduler error: `lost_grasp` uses the stricter `0.0025 m` closed-finger
threshold, while success permits retained grasps up to `0.0035 m`. An episode can therefore finish successfully
with a finger position between those values after the loss-confirmation window. The curriculum reads the success
term directly, and the failure reward explicitly excludes concurrent success, so neither the return nor promotion
statistics are penalized. The individual termination plots should not be interpreted as mutually exclusive outcome
fractions.

At maximum level-0 exposure, one window reached 69.77% at iteration 148 and correctly did not promote because the
configured threshold is 70% without rounding. The next qualifying per-rank windows moved the distributed run to
level 1 over iterations 151--152. By iteration 156 all ranks were at level 1, the most recent frontier window was
78.91%, aggregate success was 79.69%, mean reward was 53.07, and timeout and unsafe termination remained zero.
This reproduces the fresh validation's first promotion near iteration 150 and verifies the window scheduler at its
strict boundary.

Level 1 then increased exposure to 35% on a 69.53% frontier window and to 50% on an 83.85% window. It promoted
to level 2 at iteration 193 with an 89.06% qualifying window. At iteration 195 aggregate success was 90.63%,
timeout and unsafe termination were zero, and bilateral acquisition reward became nonzero as the harder reset
began requiring policy-driven acquisition. This event is the expected curriculum transition rather than grasp
instability.

Level 2 raised exposure to 35% with a 92.19% frontier window and to 50% with 91.41%, then promoted to level 3 at
iteration 232 on a 93.75% window. At iteration 235 aggregate success was 88.67%, bilateral acquisition reward was
0.374, and no timeout or unsafe event occurred. The modest aggregate-success decrease reflects the newly sampled
harder reset; the level-specific promotion windows show that level 2 converged rather than regressed.

The first fresh production attempt then exposed the same exploration-driven actor drift seen in earlier contact
frontiers. After entering level 3 at iteration 232, consecutive frontier windows fell from 30.47% to 29.15% and
23.44%. By iteration 301, aggregate success was 3.86%, lost grasp was 94.58%, dense reward was 0.114, and
bilateral acquisition reward remained 0.348. Training was stopped after saving `model_300.pt` rather than allowing
the failed actor to consume the remaining budget.

Fixed level-3 deterministic traces confirmed policy-weight regression. `model_250.pt` acquired both tails in
32/32 episodes and succeeded in 22/32; `model_300.pt` still acquired in 32/32 but lost grasp in all 32. Mean
maximum tail separation fell from approximately `0.193 m` to `0.130 m`, and dense return from 9.58 to 0.91.
Neither policy commanded a gripper open after acquisition. Adaptive learning rate was already at its `1e-5`
floor while value loss grew from about 55 to 195, so the failure was not a learning-rate spike, reset cliff, or
gripper-action bug. Stochastic frontier rollouts with mean action standard deviation near 0.203 drove repeated PPO
updates away from an already useful mean policy.

The evidence-backed consolidation resumed `model_250.pt` at level 3 after changing only its action standard
deviation to 0.1 and clearing that parameter's Adam moments. Actor and critic weights, observation normalizers,
reward, termination, curriculum, and all other optimizer state were retained. The first launch failed before an RL
update with the intermittent native Newton USD-import `SIGSEGV`; an identical retry succeeded. Unlike the original
trajectory, the low-noise continuation recovered aggregate success above 80% and promoted to level 4 at iteration
302 on a 71.875% frontier window, with 87.5% aggregate success and zero timeout. This validates staged exploration:
the original noise is needed to discover the policy from scratch, while lower noise protects contact-sensitive
consolidation once a competent deterministic actor exists.

The 100-update consolidation finished at `model_349.pt`. Online level-4 exposure remained 20%, with a final
31.05% frontier window, 36.33% aggregate success, and 3.91% timeout. Fixed deterministic traces showed that the
mean actor had not forgotten level 3: it improved from 22/32 to 24/32 successes, with acquisition in all 32.
Level 4 also acquired in all 32 but lost grasp before success in every episode. Its mean maximum tail separation
was `0.169 m`, close to the `0.18 m` requirement, and mean dense return was 7.05. Lower exploration therefore
fixed catastrophic old-level forgetting but did not yet solve the newly introduced `0.003625 m` reset. Further
training should continue from `model_349.pt` under the same low-noise contract and be accepted only if fixed
level-4 deterministic success becomes nonzero without reducing level-3 success.

### Checkpoint-continuation stability diagnosis

A fixed-learning-rate control showed that the remaining continuation instability was not explained by the
adaptive schedule alone. Resuming `model_349.pt` at level 3 with a fixed `1e-5` learning rate produced
`model_400.pt`, whose fixed deterministic level-3 trace acquired both tails but lost grasp in all 32 episodes.
The aggregate online success had simultaneously fallen from about 30% to 16%. The run was stopped at iteration
406; its artifacts are under
`logs/rsl_rl/shoelace_dual_franka/2026-09-05_06-33-02_admm_rho400_curriculum19_resume_m349_std0p1_fixedlr1e5_level3_8gpu_100`.

Crossed checkpoints isolated two cumulative failure mechanisms. Keeping the `model_349.pt` actor weights but
replacing only their observation-normalizer buffers with those from `model_400.pt` reduced deterministic level-3
success from 24/32 to 8/32. Keeping the `model_400.pt` actor weights with the older normalizer was worse: all 32
episodes timed out without bilateral acquisition. Over those 51 updates, the actor-normalizer standard-deviation
vector changed by 41% in norm, dominated by the right tail-to-TCP observation, while the actor MLP tensors changed
by only 2--4% in relative norm. Online normalization drift and PPO actor drift therefore both contributed. This
also exposes a distributed-training hazard in the installed RSL-RL implementation: observation-normalizer buffers
are updated from each rank's local rollout but only learnable gradients are all-reduced.

Fixed-level stochastic traces quantified the contact policy's narrow action margin:

| `model_349.pt` level-3 action std | Success | Lost grasp | Timeout |
|---:|---:|---:|---:|
| approximately `0.10` | 3/32 | 22/32 | 7/32 |
| `0.05` | 6/32 | 26/32 | 0/32 |
| `0.02` | 15/32 | 17/32 | 0/32 |
| `0.01` | 15/32 | 17/32 | 0/32 |
| `0.005` | 24/32 | 8/32 | 0/32 |

The same policy at level 4 reached 0/32, 1/32, and 0/32 stochastic successes at standard deviations `0.005`,
`0.01`, and `0.02`, respectively. Noise above approximately `0.005` destroys too many level-3 contacts, while
small noise still occasionally crosses the unseen level-4 boundary. The continuation checkpoint therefore uses
`0.005`, clears stale Adam state, disables entropy pressure, and disables randomized initial episode lengths.

A task-local `FixedObservationStatisticsMLPModel` preserves the checkpoint's exact normalization path while
preventing further statistics drift. Algebraically folding the normalizer into the first layer was rejected:
although the affine outputs matched numerically in a unit calculation, floating-point reordering changed the
contact trajectory enough to reduce fixed level-3 success from 24/32 to 18/32. The fixed-statistics model keeps
the original operation order and is selected only for checkpoint consolidation; fresh training retains the
standard model and online normalization.

The PPO rollout horizon was also raised from 16 to 80 for consolidation. Successful and lost-grasp episodes
usually finish after 60--75 control steps, so the old horizon prevented terminal success and failure impulses from
entering the same GAE return as the early pull action. The longer-horizon run resumes the curriculum at level 4,
which is the frontier that produced `model_349.pt`, and assigns only 20% of slots to it while replaying levels 3,
2, and 1. It uses one PPO epoch per rollout and a fixed `1e-5` learning rate.

This intervention produced the first deterministic level-4 success without forgetting level 3. After 16
long-horizon updates, `model_365.pt` reached 24/32 fixed level-3 successes and 1/32 fixed level-4 successes;
the online level-4 frontier windows rose from zero to 1.58% and then 2.34%. Earlier `model_355.pt` still had
21/32 and 0/32, respectively. The completed validation artifacts are under
`logs/rsl_rl/shoelace_dual_franka/2026-09-05_07-00-42_admm_rho400_curriculum19_resume_m349_fixedstats_std0p005_resetopt_h80_epoch1_level4frac0p2_8gpu_20`
and
`logs/rsl_rl/shoelace_dual_franka/2026-09-05_07-09-19_admm_rho400_curriculum19_resume_m355_fixedstats_h80_epoch1_level4frac0p2_8gpu_45`.
Continuation from `model_365.pt` is being monitored under
`logs/rsl_rl/shoelace_dual_franka/2026-09-05_07-19-17_admm_rho400_curriculum19_resume_m365_fixedstats_h80_epoch1_level4frac0p2_8gpu_35`.

### World-origin precision and contact-release curriculum diagnosis

The apparent gap between 32-environment validation and 128-environment training was traced to world-origin
magnitude rather than reward attribution. The task observations already subtract each environment origin and
Newton creates a separate physics world for every clone, but this contact regime is sensitive to micrometre-scale
coordinate precision. A fixed deterministic `model_370.pt` level-4 rollout changed only `scene.env_spacing` and
produced the following success counts:

| Environment spacing [m] | Success |
|---:|---:|
| `1.50` | 37/128 |
| `1.00` | 51/128 |
| `0.75` | 80/128 |
| `0.50` | 93/128 |
| `0.25` | 122/128 |

The task default is therefore `0.25 m`. This does not introduce inter-environment collisions because the Newton
clones are separate worlds. It instead keeps all local coordinates close enough to the origin for the existing
micrometre-sensitive contact setup. The 128-environment gate is retained for checkpoint selection; the earlier
32-environment result was too weak to reveal this spatial scale effect.

A reset-only sweep at `0.25 m` spacing then isolated a second, genuine contact-release cliff. The same
`model_370.pt` acquired both tails in every episode, but success fell sharply as the initial finger position
crossed approximately `0.00355 m`:

| Initial finger position [m] | Success |
|---:|---:|
| `0.003531250` | 122/128 |
| `0.0035390625` | 114/128 |
| `0.003546875` | 108/128 |
| `0.00355078125` | 74/128 |
| `0.003552734375` | 31/128 |
| `0.0035546875` | 23/128 |
| `0.0035625` | 7/128 |

All failed episodes acquired the tails at about step 7 and then lost contact near steps 60--75 without issuing
an open-gripper action. This rules out reset acquisition, timeout handling, and premature gripper opening as the
primary cause. Seven intermediate reset states were inserted between the former levels 4 and 5, yielding 29
curriculum levels. The settled arm pose is repeated through level 23, followed by the existing five approach
states. The exact inserted finger positions are `0.00353515625`, `0.0035390625`, `0.00354296875`,
`0.003546875`, `0.00355078125`, `0.0035546875`, and `0.00355859375 m`.

With the denser curriculum and `0.25 m` spacing, low-noise continuation from `model_370.pt` rapidly promoted
through levels 4--9. A subsequent `std=0.01` consolidation improved fixed deterministic level-9 success from
69/128 at `model_390.pt` to 77/128 at `model_399.pt`. Continuing from that checkpoint produced the following
fixed-level results without forgetting level 9:

| Checkpoint | Level 9 | Level 10 |
|---|---:|---:|
| `model_410.pt` | 109/128 | 78/128 |
| `model_415.pt` | 114/128 | 83/128 |
| `model_418.pt` | 114/128 | 99/128 |

For `model_418.pt` at level 10, stochastic evaluation reached 99/128 with `std=0.005` and 104/128 with
`std=0.01`; reducing exploration was therefore rejected. The next continuation kept `std=0.01` and promoted all
ranks through levels 10, 11, and 12. Level 13, whose reset jumps from `0.0035625` to `0.00359375 m`, exposed the
next contact cliff: its first complete frontier windows were approximately 48%, 29%, and 37% while acquisition
remained complete and unsafe/time-out events remained zero. The run artifacts are under
`logs/rsl_rl/shoelace_dual_franka/2026-09-05_08-56-18_admm_rho400_curriculum29_spacing0p25_resume_m418_fixedstats_std0p01_h80_epoch1_level10frac0p5_8gpu_20`.

The completed run stopped at level 13 with a final 47.7% frontier window. Deterministic evaluation of
`model_430.pt`, `model_435.pt`, and `model_437.pt` retained 113/128, 119/128, and 117/128 successes at level 12;
their level-13 results were 55/128, 50/128, and 63/128. A reset-only sweep of `model_437.pt` then measured
117/128, 117/128, 114/128, 102/128, 98/128, 83/128, and 59/128 successes at the seven 3.90625-micrometre
increments between `0.0035625` and `0.00359375 m`. The curriculum therefore adds all seven intermediate states
and now contains 36 levels. The former level 13 becomes level 20; continuation starts at the new level 13 so the
policy traverses the measured contact boundary instead of jumping across it.

That 36-level continuation promoted every rank from level 13 through level 20 in 20 PPO updates. The former
level-13 reset reached a 70.0% promotion window, and deterministic `model_456.pt` evaluation retained 114/128
successes at level 19 and 103/128 at level 20. A sweep across the next `0.00359375` to `0.003625 m` gap found
91/128 successes at the first 3.90625-micrometre increment, followed by 74, 68, 49, 33, 31, and 29; the endpoint
reached only 21/128. Acquisition remained complete throughout. Seven more equal increments are therefore added,
bringing the curriculum to 43 levels. The former `0.003625 m` level becomes level 28, and continuation begins at
the new level 21 (`0.00359765625 m`).

Lowering the value-loss coefficient was also considered and rejected. RSL-RL clips actor and critic gradients
separately and the shoelace actor and critic do not share parameters, so changing critic loss weight would not
increase the actor update. Frontier density and deterministic contact traces remain the evidence-based controls
for the next intervention.

Continuation from `model_456.pt` crossed the newly inserted levels 21--23 in 20 updates. Level 21 promoted at
iteration 462 with a 71.09% frontier window, and level 22 recovered from its initial reset-transition dip before
promoting at iteration 468 with the same qualifying rate. The run ended at level 23 with a 68.54% online window,
74.7% aggregate success, 25.4% lost grasp, and no unsafe or time-out events. Fixed deterministic gates showed
continued policy improvement: `model_465.pt`, `model_470.pt`, and `model_475.pt` reached 80/128, 100/128, and
101/128 successes at level 23, respectively. The apparent sub-threshold final online window was therefore
sampling and distributed-window lag rather than a deficient mean policy. The artifacts are under
`logs/rsl_rl/shoelace_dual_franka/2026-09-05_09-33-37_admm_rho400_curriculum43_spacing0p25_resume_m456_fixedstats_std0p01_h80_epoch1_level21frac0p5_8gpu_20`.

The next 20-update continuation from `model_475.pt` promoted level 23 at iteration 477 and level 25 at iteration
489, then ended at level 26. Its final online window was 64.01%, while aggregate success remained 74.38% and
unsafe and time-out events remained zero. Fixed 128-environment deterministic gates again distinguished window
noise from policy capability:

| Checkpoint | Level 25 | Level 26 | Level 27 |
|---|---:|---:|---:|
| `model_485.pt` | 97/128 | 83/128 | not evaluated |
| `model_490.pt` | 94/128 | 85/128 | 82/128 |
| `model_494.pt` | 104/128 | 90/128 | 85/128 |

Every evaluated episode acquired both tails, and every non-success termination was lost grasp; no unsafe or
time-out event occurred. `model_494.pt` therefore improved the mean policy at all three measured levels even
though stochastic frontier windows did not promote every rank. Level 27 deterministic success is already 66.4%,
close enough to the 70% threshold to continue consolidation without inserting another reset state. These run
artifacts are under
`logs/rsl_rl/shoelace_dual_franka/2026-09-05_09-48-54_admm_rho400_curriculum43_spacing0p25_resume_m475_fixedstats_std0p01_h80_epoch1_level23frac0p5_8gpu_20`.

Continuation from `model_494.pt` immediately promoted level 26 with a 71.09% window and level 27 with a 70.99%
window, reaching level 28 at iteration 499. Level 28 remained below the stochastic promotion threshold, with a
final 59.96% frontier window and 64.35% aggregate success, but fixed checkpoint gates showed a real mean-policy
gain rather than a stall. Deterministic level-28 success changed from 77/128 at the input `model_494.pt` to
78/128, 71/128, 76/128, and 85/128 at `model_500.pt`, `model_505.pt`, `model_510.pt`, and `model_513.pt`.
`model_513.pt` also improved level-27 success from 85/128 to 88/128. Every trace acquired both tails and ended
only in success or lost grasp. The final checkpoint is therefore the next continuation point: its 66.4% fixed
level-28 success is close to the promotion threshold, while the current 3.90625-micrometre reset interval is
already sufficiently dense. The artifacts are under
`logs/rsl_rl/shoelace_dual_franka/2026-09-05_10-05-12_admm_rho400_curriculum43_spacing0p25_resume_m494_fixedstats_std0p01_h80_epoch1_level26frac0p5_8gpu_20`.

Continuation from `model_513.pt` promoted level 28 at iteration 522 with a 70.54% window, then exposed the much
larger `0.003625` to `0.00375 m` reset jump at level 29. Its first two complete level-29 windows were 4.41% and
0%, despite bilateral acquisition in every episode, so the run was stopped after `model_525.pt` rather than
training further on a zero-success frontier. Fixed gates confirmed that `model_525.pt` improved level-28 success
to 87/128 without forgetting, while both `model_520.pt` and `model_525.pt` had 0/128 level-29 successes under the
original gripper drive.

Reset-only sweeps then revealed a discontinuity at the initial cable-contact boundary. `model_525.pt` achieved
90/128 successes at `0.003625 m`, 84--90/128 through `0.0036279296875 m`, and 0/128 at
`0.00362890625 m`; every trace acquired both tails at step 8 and continually commanded both grippers closed.
The actor normalizer did not explain this sub-micrometre cliff: the corresponding raw finger-position change was
only about `6e-5` in normalized coordinates and changed the first-layer activation norm by less than `1e-4`.
Solver ablations also rejected contact matching as the remedy. Disabling matching reduced the baseline to 0/128,
sticky matching reduced it to 40/128, and increasing the latest-matching ADMM iteration count from five to ten
left level 29 at 0/128.

The actual bottleneck was the overdamped gripper drive. With the original `stiffness=1000` and `damping=100`, the
tail had eight policy steps to escape before the fingers met the strict acquisition threshold. A drive sweep
showed that reducing damping alone acquired earlier but still produced zero success, whereas sufficient stiffness
both acquired at step 3 and retained the cable. The selected `stiffness=6000`, `damping=60` setting produced
108/128 level-29 successes in the controlled sweep. It retained 128/128 at levels 0, 25, and 28 and 102/128 at
level 20; the original drive reached 128/128, 111/128, 87/128, and 116/128 at the corresponding levels. The
small level-20 regression remains above the curriculum threshold and is outweighed by complete adjacent-frontier
retention. Repeating the gate through the modified task defaults produced 128/128 deterministic successes at
level 28, 111/128 deterministic and 104/128 stochastic successes at level 29, acquisition at step 3, and no
unsafe or time-out events. The stronger, less overdamped drive is therefore used for subsequent training instead
of adding sub-micrometre reset levels.

With the recalibrated drive, a 20-update continuation promoted levels 29, 30, 31, 32, and 33 with qualifying windows
of 78.91%, 100%, 84.50%, 73.44%, and 71.54%. It ended at level 34 with 72.24% aggregate success and no unsafe
or time-out events, but that aggregate metric still contained prior-level episodes. Full 600-step deterministic
traces corrected the interpretation: `model_525.pt`, `model_535.pt`, `model_540.pt`, and `model_544.pt` all had
0/128 acquisition and 128/128 time-outs at the level-34 `0.005 m` reset, while the training checkpoints retained
114--118/128 success at level 33. Both grippers closed to approximately `0.002 m`, but the right tail escaped to
about `0.85 m` before acquisition. The new drive fixed the earlier contact-establishment timing cliff but could
not bridge the entire `0.004` to `0.005 m` opening in one reset transition. The run artifacts are under
`logs/rsl_rl/shoelace_dual_franka/2026-09-05_10-57-41_admm_rho400_curriculum43_spacing0p25_drive6000d60_resume_m525_fixedstats_std0p01_h80_epoch1_level29frac0p5_8gpu_20`.

A reset-only sweep of `model_544.pt` obtained 37/128, 5/128, 1/128, and 29/128 successes at `0.0040625`,
`0.004125`, `0.0041875`, and `0.00425 m`, followed by no acquisition at `0.004375 m` or beyond. Refining the
first interval produced 114/128, 107/128, 94/128, and 94/128 successes at `0.00400390625`, `0.0040078125`,
`0.00401171875`, and `0.004015625 m`; success fell to 47/128 at the next sampled point. These four measured,
high-signal states are inserted after the former level 33, bringing the curriculum to 47 levels and shifting the
five arm-approach states without changing their poses. Continuation starts at the new level 34. Once these states
are consolidated, the next reset sweep should use the improved checkpoint rather than preallocating hundreds of
unverified micrometre-scale levels.

The 20-update continuation from `model_544.pt` validated all four inserted states. It promoted levels 34 and 35
with 77.34% and 70.31% windows, then consolidated level 36 until a 70.98% window promoted nearly every rank at
iteration 558. All ranks reached level 37 on the following update. The final aggregate success was 75.95%, with
24.05% lost grasp and no unsafe or time-out terminations. Fixed level-37 evaluation selected `model_560.pt` over
the adjacent checkpoints: models 555, 560, and 563 produced 90/128, 92/128, and 90/128 deterministic successes,
respectively, and every episode acquired both tails.

The improved checkpoint still had 0/128 acquisition and 128/128 time-outs at the original `0.005 m` endpoint,
so more PPO updates at that reset would again provide no task signal. A new 128-environment reset sweep measured
88/128 successes at `0.004017578125 m`, followed by 68/128, 54/128, 49/128, and 44/128 at
`0.0040185546875`, `0.00401953125`, `0.0040205078125`, and `0.004021484375 m`. Wider sampled points through
`0.004046875 m` retained 39--44/128 successes. Every non-endpoint sweep episode acquired both tails, proving
that these are difficult but learnable retention states rather than the zero-acquisition regime. The nine
measured intermediate resets are therefore inserted before `0.005 m`, bringing the curriculum to 56 levels.
The next continuation resumes `model_560.pt` at level 37 so the actor must consolidate each measured contact
transition before it encounters the unsignalled endpoint.

### Low-noise frontier and retention-credit diagnosis

The 56-level continuation confirmed that action noise, rather than missing acquisition, limited promotion at the
new contact frontier. With the checkpoint's approximately `0.01` standard deviation, level 37 promoted but level
38 stalled; deterministic level-38 success nevertheless rose from 88/128 at the input `model_560.pt` to 86/128
at `model_579.pt`, within rollout variation rather than a clear policy gain. Controlled stochastic gates of the
input checkpoint produced 77/128 and 70/128 successes at level 38 with standard deviations `0.005` and `0.01`.
At level 39, standard deviations `0.001`, `0.0025`, and `0.005` produced 64/128, 67/128, and 59/128 successes.
Every episode acquired both tails, so the degradation was post-acquisition contact loss.

A `std=0.005` continuation promoted level 38 but did not improve the level-39 mean policy. A lower-noise
`std=0.0025` run with the prior fixed `1e-5` learning rate briefly improved deterministic level-39 success to
75/128 at `model_565.pt`, then fell to 49/128 four updates later. Because Gaussian PPO gradients scale inversely
with action variance, the same learning rate was no longer conservative at the smaller standard deviation.
Reducing the learning rate to `2.5e-6` kept surrogate loss near `0.003--0.009` and produced the best local
checkpoint, `model_574.pt`, with 90/128, 78/128, and 63/128 single-run deterministic successes at levels 38,
39, and 40. These artifacts are under the `2026-09-05_12-24-22`, `2026-09-05_12-35-49`, and adjacent
low-noise run directories in `logs/rsl_rl/shoelace_dual_franka`.

Further continuation exposed how narrow that optimum was. One ordinary four-minibatch PPO iteration changed the
actor MLP tensors by only `2.77e-5` in relative L2 norm, but reduced the next fixed level-39 gate from 78/128 to
64/128. On observations visited by the parent policy, the update's mean Gaussian KL was approximately `0.0148`,
above the configured `0.01` target because the fixed learning-rate schedule does not enforce that target. Some
single-action differences reached `0.00185`, roughly 74% of the `0.0025` exploration standard deviation.
Replacing four minibatches with one full-batch optimizer step reduced the relative actor change to `6.7e-6`, but
a GPU-swapped, three-seed comparison still measured 392/768 parent successes versus 353/768 child successes at
level 39. Smaller optimizer steps alone therefore preserve the contact trajectory better but do not supply the
missing corrective objective.

The stochastic 70% promotion gate was independently too strict at this boundary. Changing only the gate to 50%
promoted level 39 on a 51.5% window and exposed level 40, where windows then settled near 40--42%. A three-seed
level-40 gate found 167/384 successes for the parent and 163/384 for the final continuation checkpoint, with
complete acquisition and only lost-grasp failures. Lowering the gate prevents needless overtraining of a usable
frontier, but it does not improve post-acquisition retention by itself.

State traces also exposed a tempting dense-credit hypothesis. Successful level-39 episodes kept the right tail
approximately `0.0078 m` from its TCP at maximum separation, whereas failures averaged `0.026 m` there and
reached `0.050 m` immediately before termination. The policy continued commanding both grippers closed. After
bilateral acquisition, task and directional-pull progress are enabled only while both strict-grasp predicates
remain true, so an incipient slip removes dense task credit before the failure impulse arrives after the six-step
release-confirmation window.

A signed, bounded potential difference of the weaker tail's smooth tail-to-TCP score was tested rather than
accepted from source inspection alone. The fresh eight-GPU run
`2026-09-05_13-29-48_admm_rho400_curriculum56_spacing0p25_drive6000d60_retention1_promo0p5_fresh_8gpu_20000`
reproduced the earlier standalone-retention failure documented above. By iteration 48, mean episode length had
grown from 8 to 193 steps but success remained zero, lost grasp remained 100%, and dense task reward was
`-0.135`. The known-good fresh baseline already had positive `0.163` dense reward and a 54-step mean episode at
iteration 39, before discovering success near iteration 89. The new policy was learning to delay release and
hold statically instead of completing the geometric task. The run was stopped before its next checkpoint, and
the retention potential was removed from both source and configuration.

The known healthy reset-relative dense reward therefore remains unchanged. The independently validated 50%
promotion gate is retained: it moves past frontier states that already supply complete acquisition and useful
success trajectories without pretending that the rejected retention term solved them. Subsequent continuation
uses only pre-retention checkpoints and changes curriculum gating separately from the reward.

### Promotion-gate ablation at the low-noise frontier

A ten-update continuation from `model_583.pt` changed only the promotion and fraction-increase gates from 50%
to 40%. All eight ranks promoted level 40 on a 46.09% completed frontier window and level 41 on a 41.09%
window, reaching level 42 with a final 40.79% window. Aggregate success was 46.14%; acquisition remained
complete, every failure was lost grasp, and unsafe and time-out terminations remained zero. The artifacts are
under
`logs/rsl_rl/shoelace_dual_franka/2026-09-05_13-39-28_admm_rho400_curriculum56_spacing0p25_drive6000d60_resume_promo0p5_m583_fixedstats_std0p0025_h80_epoch1_promo0p4_lr2p5e6_level40frac0p5_8gpu_10`.

GPU-swapped deterministic gates separated curriculum movement from actor improvement. At level 42, three seeds
and both GPU assignments produced 318/768 successes for the input `model_583.pt` and 320/768 for
`model_591.pt`. At level 41, the corresponding two-GPU swapped comparison produced 102/256 and 98/256.
These differences are within sampling variation: the lower gate successfully avoids overtraining adjacent reset
states that the policy can already solve about 40% of the time, but the ten PPO updates did not measurably
improve or forget the mean policy. The 40% gate is therefore kept as an experimental continuation override and
is not lowered further or promoted to the source default yet.

The evaluation trace utility now asserts the constructed environment's runtime solver. Its earlier startup
summary displayed `CouplerProxyCfg` because that summary was rendered before `ShoelaceEnv` converted the proxy
template; every repeated gate reported and asserted `CouplerAdmmCfg` after construction. The header discrepancy
did not invalidate the ADMM evaluations. The next controlled continuation starts from `model_591.pt` at level
42 and keeps every optimizer, rollout, exploration, and curriculum parameter unchanged.

That continuation promoted every rank from level 42 to level 43 on a 43.75% frontier window, then completed at
level 43 with a 32.81% final window. All eight ranks exited normally after ten updates. A three-seed, GPU-swapped
deterministic comparison measured 261/768 input `model_591.pt` successes and 293/768 `model_596.pt` successes at
level 43; five of six paired gates improved, while acquisition remained complete and all failures were lost
grasp. The update therefore moved the mean policy in the desired direction, but not far enough to establish a
robust 40% frontier margin.

A two-assignment reset-only sweep of `model_596.pt` then measured 106/256, 98/256, 113/256, 104/256, 82/256,
92/256, and 88/256 successes at gripper positions `0.004021484375`, `0.0040224609375`, `0.0040234375`,
`0.0040244140625`, `0.004025390625`, `0.0040263671875`, and `0.00402734375 m`. The first four positions
retained approximately 40--44% success before the latter three fell to 32--36%. The three quarter-step positions
between the former levels 43 and 44 are therefore inserted, producing 59 reset levels and preserving a uniform
`0.9765625` micrometre transition across this measured contact boundary. Subsequent continuation resumes
`model_596.pt` at the unchanged physical reset that remains level 43.

The first 59-level continuation did not promote online: its level-43 windows ranged from 29.34% to 38.83% and
ended at 37.06%. Nevertheless, fixed gates selected `model_604.pt` rather than the final checkpoint. A
three-seed, GPU-swapped comparison measured 252/768 successes for the input `model_596.pt` and 296/768 for
`model_604.pt`, an absolute gain of 5.73 percentage points with improvement in five of six paired gates. All
episodes again acquired both tails and ended only in success or lost grasp. The next short continuation therefore
started from `model_604.pt` at level 43. Its final completed frontier window reached 40.62%, promoting the
rank-zero curriculum to the first inserted reset state at level 44. All eight workers exited normally after ten
updates, unsafe and time-out terminations remained zero, and checkpoints `model_604.pt` through `model_613.pt`
were written. This is the first online evidence that the denser curriculum crosses the former level-43-to-44
contact boundary and exercises the newly inserted transition instead of jumping directly to the harder former
level 44.

The online promotion did not by itself prove a better deterministic actor. At the new level 44, a four-seed,
GPU-swapped comparison measured 379/1024 successes for the input `model_604.pt` and 365/1024 for
`model_612.pt`; the two GPU assignments favored opposite checkpoints. Both actors acquired every episode and
all failures were lost grasp, but neither established a 40% mean-policy margin. The next controlled ablation
therefore restarts the stronger input `model_604.pt` directly at level 44 and raises only the current-level sample
fraction from 50% to 75%. This gives the new frontier more learning signal while retaining 25% replay, without
changing the reward, termination, optimizer, exploration noise, or 40% promotion gate.

The 75%-frontier ablation promoted through the first two inserted levels, 44 and 45, on 40.00% and approximately 43.5%
windows, then stopped at level 46 as its complete windows declined from 39.53% to 30.62%. Fixed-policy gates
showed that the movement was curriculum sampling rather than actor improvement: a four-seed, GPU-swapped
comparison at level 46 measured 359/1024 successes for the input `model_604.pt` and 351/1024 for the apparent
single-seed peak `model_609.pt`. All episodes acquired both tails, and all failures were lost grasp. Across the
ten updates, actor MLP parameters moved only `7.7e-5` in relative L2 norm while critic parameters moved
`1.1e-3`; deterministic performance did not improve.

The remaining credit gap inside `reset_relative_dense_reward` motivated one more controlled negative-result
test. After bilateral acquisition, the term disables task-potential credit immediately when either strict grasp
becomes false, although lost-grasp termination waits six control steps. An experimental version preserved only
negative task-potential differences during that interval and continued to suppress positive ungrasped progress.
A fixed `model_604.pt` level-46 gate left successful dense return unchanged at approximately 10.50 while reducing
lost-grasp dense return from 9.41 to 8.40 and its final-16-step component from 2.13 to 1.15.

Fresh training disproved the intervention. At iteration 10 it had a 47.83-step mean episode, `-0.0723` dense
reward, zero acquisition, and 100% lost-grasp termination. The rejected retention run had 46.50 steps and
`-0.0908` dense reward at the same iteration, whereas the healthy baseline had 6.04 steps and positive `0.0005`
dense reward. Negative-only shaping still taught the actor to avoid motion that might incur regression rather
than to complete the pull. The run was stopped at iteration 12 and the shaping change was removed; the healthy
reset-relative reward remains the source contract.

A broader consolidation ablation restarted `model_604.pt` at level 46 with action standard deviation `0.01`,
a fresh Adam state, fixed `1e-5` learning rate, and 75% frontier exposure. Online windows promoted through level
48, but fixed-policy validation rejected the apparent progress. A four-seed, GPU-swapped level-48 comparison
measured 365/1024 successes for the input actor and 347/1024 for `model_611.pt`. The candidate actor moved
`4.46e-4` in relative L2 norm and changed actions by approximately `9.9e-4` RMS, so the failure was an
incorrect/noisy update direction rather than a zero-sized update.

Reset-only gates then exposed the next curriculum discontinuity. The input actor retained bilateral acquisition
through `0.0043203125 m` and reached 29/128 deterministic successes there. Increasing the initial finger
position by only `7.8125 micrometres` to `0.004328125 m` produced 128/128 time-outs and zero acquisitions over
600 steps. Its mean dense return was approximately `-1.01`; the lack of a time-out failure impulse was not more
rewarding than the approximately `+7.4` total return of an acquired level-48 episode ending in lost grasp.
Stochastic evaluation at the new reset supplied a learnable bridge: raising action standard deviation from
`0.0025` to `0.05` increased acquisition from 34/128 to 53/128 and produced 23/128 successes. At `0.005 m`, by
contrast, even `std=0.1` acquired only 5/128 episodes and produced no success. Six measured reset states from
`0.004125` through `0.004328125 m` are therefore inserted before `0.005 m`, increasing the curriculum to 65
levels. The reward and termination contract remained unchanged for the resulting acquisition experiments.

The higher-exploration hypothesis did not survive fixed-policy validation. A ten-update continuation with
`std=0.05`, a fresh Adam state, `1e-5` learning rate, and 75% frontier exposure ended with approximately 12.1%
aggregate success and 59% time-out termination. Its deterministic checkpoints all had zero acquisition at level
55, and the final checkpoints also destroyed acquisition at the adjacent level 54. These artifacts are under
`logs/rsl_rl/shoelace_dual_franka/2026-09-05_16-31-19_admm_rho400_curriculum65_resume_m604_std0p05_resetadam_h80_epoch1_promo0p3_lr1e5_level55frac0p75_8gpu_10`.

Action traces localized the cliff to the binary left-gripper command. At level 54 the input actor's first left
gripper output was approximately `-0.00073`, selecting close. The `7.8125 micrometre` reset change at level 55
moved it to approximately `+0.00092`, selecting open; feedback through the `last_action` observation then kept
both grippers open. A causal output-row intervention confirmed this diagnosis. Subtracting only `0.001` from the
left-gripper output bias changed level-55 acquisition from 0/128 to 128/128. In a four-seed, GPU-swapped comparison,
the `-0.001` and `-0.002` variants produced 227/1024 and 211/1024 successes, respectively, and both acquired in
every episode.

Uniform Gaussian noise remained incompatible with that narrow binary margin. For the `-0.001` variant, action
standard deviations from `0.0025` through `0.02` acquired only approximately 42--45% of episodes. Giving the arms
`0.01` standard deviation and the grippers `2.5e-5` instead restored 256/256 acquisitions and 61/256 successes.
However, two ordinary Gaussian PPO updates still reduced a GPU-swapped deterministic gate from 221/1024 to
195/1024 successes. The manual bias and anisotropic noise are therefore diagnostic warm starts, not an accepted
training solution.

The task now contains an experimental hybrid RSL-RL distribution that models the twelve arm dimensions as
Gaussian and the two sign-thresholded gripper dimensions as Bernoulli. Deterministic outputs retain the original
MLP values so existing `last_action` observations and exported-policy behavior are unchanged. The sampled
gripper sign has a Bernoulli likelihood, removing the inverse-Gaussian-variance gradient at the binary boundary.
Existing Gaussian checkpoints require an explicitly calibrated Bernoulli logit scale; this changes the stochastic
policy contract and must not be treated as an ordinary resume.

Controlled hybrid-distribution tests separated numerical stability from learning progress. Without an output-bias
warm start, scale 100 produced 110/256 acquisitions and 22/256 successes at level 55. With the `-0.005` warm start,
scale 1000 produced 251/256 acquisitions and 50/256 successes. Surrogate losses remained near zero during PPO.
Nevertheless, training the full shared actor caused deterministic acquisition to fall from 128/128 at `model_606.pt`
to 0/128 from `model_607.pt` onward: continuous-arm gradients could still move the shared representation across
the gripper sign boundary.

A final controlled candidate froze the checkpoint actor's hidden backbone and trained only its last action layer
and distribution parameters. This prevented the acquisition collapse through four updates. Its four-seed,
GPU-swapped level-55 gate measured 203/1024 input successes and 208/1024 `model_607.pt` successes, with complete
acquisition for both. The 0.49 percentage-point difference is not a measurable policy improvement, so no long run
was launched. The experiment artifacts are under
`logs/rsl_rl/shoelace_dual_franka/2026-09-05_18-11-04_admm_rho400_curriculum65_bias5_hybrid1000_frozenbackbone_resume_m604_h80_epoch1_promo0p3_lr2p5e6_level55frac0p75_8gpu_4`.
The current verified conclusion is that binary-gripper distribution and shared-backbone interference are real
bugs, while the remaining post-acquisition improvement still needs a stronger training signal or a separately
parameterized gripper head.

An independently parameterized arm/gripper actor then isolated the two action families without changing the
checkpoint's deterministic function: copying the old MLP into both branches gave bitwise-identical outputs. The
fully trainable split actor delayed but did not eliminate acquisition collapse. Its first left-gripper output moved
from `-0.00408` to `-0.00215`, `-0.00192`, and `-0.00064` before crossing to `+0.00024` at `model_607.pt`; that
checkpoint changed from complete acquisition to 0/128 acquisition and 128/128 time-outs. This proves that direct
gripper-policy gradients, not only shared arm gradients, can select the open-gripper exploit at the reset frontier.
The rejected run is under
`logs/rsl_rl/shoelace_dual_franka/2026-09-05_18-33-12_admm_rho400_curriculum65_bias5_hybrid1000_splitbackbone_resume_m604_h80_epoch1_promo0p3_lr2p5e6_level55frac0p75_8gpu_4`.

Freezing only the split gripper branch preserved its first-step outputs exactly while leaving the full arm MLP and
continuous-action variance trainable. Every screened checkpoint through eight updates retained 128/128 acquisition.
However, a four-seed, two-round GPU-swapped gate measured 191/1024 input successes and 187/1024 `model_611.pt`
successes, or `-0.39` percentage points with an approximate 95% interval of `[-3.75, +2.97]` points. Thus the
`2.5e-6` arm-only update was safe but did not produce measurable learning. Its artifacts are under
`logs/rsl_rl/shoelace_dual_franka/2026-09-05_18-46-39_admm_rho400_curriculum65_bias5_hybrid1000_frozengripper_split_resume_m604_h80_epoch1_promo0p3_lr2p5e6_level55frac0p75_8gpu_8`.
The next controlled test keeps the gripper branch frozen and increases only the arm learning rate; no training
process is active after the reported gate.

Raising only the arm learning rate fourfold to `1e-5` made the first surrogate loss `+0.0281`, after which it
settled between approximately `+0.0016` and `+0.0040`. Online success reached 22.8%, but the fixed-policy screen
again rejected apparent progress: the input actor produced 31/128 successes, while `model_608.pt` through
`model_611.pt` produced 25, 27, 31, and 24. Every policy retained 128/128 acquisition and bitwise-identical initial
gripper outputs. Consequently, the remaining failure is not merely an undersized PPO step. The sparse, noisy
post-acquisition success signal does not provide a reliably improving arm gradient even after acquisition is
structurally protected. The rejected higher-rate run is under
`logs/rsl_rl/shoelace_dual_franka/2026-09-05_18-59-52_admm_rho400_curriculum65_bias5_hybrid1000_frozengripper_split_resume_m604_h80_epoch1_promo0p3_lr1e5_level55frac0p75_8gpu_8`.
Successful stochastic rollouts did not yield a useful arm-only self-imitation target. The collection contained
205 successful episodes and 12,630 transitions, but the successful action residuals had no broad constant
direction: 11 of 12 arm dimensions had absolute t-statistics below 1.6. Offline behavior cloning also increased
held-out action error rather than reducing it. This rules out a simple mean-residual or unweighted cloning update;
the rollout set is retained only as diagnostic evidence.

Inspection of the reset-relative implementation then found a reward-accounting error. A newly reset environment
did not seed its previous task, approach, and acquisition potentials from the actual reset state, so the first
post-reset action was credited for the reset geometry itself. The reward term now samples those baselines in its
reset callback. A regression test fails under the old behavior and passes with the fix. The correction is real but
not sufficient by itself: a two-round, four-seed, GPU-swapped level-55 gate measured 204/1024 successes for the
input actor and 211/1024 for an eight-update frozen-gripper candidate, a `+0.68` percentage-point difference with
an approximate 95% interval of `[-2.80, +4.17]` points. The rejected run is under
`logs/rsl_rl/shoelace_dual_franka/2026-09-05_19-23-40_admm_rho400_curriculum65_resetbaseline_bias5_hybrid1000_frozengripper_split_warmstart_m604_h80_epoch1_promo0p3_lr1e5_level55frac0p75_8gpu_8`.

Post-frontier screens exposed a second discontinuity in the binary gripper policy. The biased input checkpoint had
zero acquisition and 128/128 time-outs at every level from 56 through 63; its first left/right gripper outputs
moved from approximately `(0.13, 0.15)` at level 56 to strongly positive open commands at later levels. Reducing
the Bernoulli logit scale restored stochastic close exploration at level 56, but independent decisions at every
step destroyed grasp retention. At scale 10, level 55 produced 20/128 successes while level 56 acquired only
16/128; at scale 20 those values were 22/128 and 7/128. A trainable scale-10 split actor then learned the open
action and produced zero deterministic acquisition at both levels 55 and 56, confirming an objective-level
incentive rather than merely insufficient exploration.

The incentive came from the time-limit contract. The task was configured as an infinite-horizon environment, so
RSL-RL bootstrapped the critic at the 20-second task time limit, while the failure reward excluded time-outs.
Opening the gripper and waiting therefore avoided the failure impulse and retained a positive terminal bootstrap.
The task is now finite-horizon and explicitly includes `time_out` in the exclusive failure event, while excluding
concurrent success. Existing callers of the general event-reward term keep the old default of masking time-outs.
Targeted tests cover both the task configuration and the opt-in time-out event count.

A causal gripper-head intervention verifies that the corrected frontier is physically solvable. Subtracting 0.30
from the two gripper output biases produced 119/128 deterministic successes at level 55 and 77/128 at level 56.
With stochastic scale-20 gripper decisions, level 56 produced 68 successes, 57 lost grasps, and only 3 time-outs;
level 57 acquired 50/128 episodes but did not yet succeed. Under the corrected reward contract, level-57 lost-grasp
episodes returned approximately `+3.22`, whereas time-outs returned approximately `-13.32`, reversing the former
undiscounted open-gripper return.

A 12-update, eight-GPU controlled continuation started at level 56 with a fresh Adam state, fixed `2.5e-6`
learning rate, and the split hybrid actor. It initially promoted online to level 57 with a 61.0% completed-window
success rate and zero time-outs, but later regressed to 18.7% online success and 59.3% time-outs. Deterministic
validation rejected every candidate. The input, `model_606.pt`, `model_610.pt`, and `model_615.pt` produced
respectively 118, 113, 124, and 120 successes out of 128 at level 55, but only 74, 72, 64, and 66 at level 56.
All screened trained checkpoints had zero acquisition and 128/128 time-outs at level 57. The run is under
`logs/rsl_rl/shoelace_dual_franka/2026-09-05_20-19-53_admm_rho400_curriculum65_finitehorizon_timeoutfail_resetbaseline_biasminus0p30_hybrid20_split_warmstart_m604_h80_epoch1_promo0p3_lr2p5e6_level56frac0p75_8gpu_12`.

An eight-GPU advantage probe explains why the finite-horizon correction alone did not prevent this regression.
Across 81,920 first-rollout transitions, the left gripper's aggregate Bernoulli output gradient was `-0.0165`, so
gradient descent increased the open logit. At the first reset step, open samples had mean normalized advantage
`+0.281` while close samples had `-0.849`, producing a much stronger `-0.603` output gradient. Once a sampled open
action entered the `last_action` observation, the next policy output saturated on open and its later Bernoulli
gradient became nearly zero. No time-out occurred in the 80-step rollout; the failure impulse arrives at step 600,
far outside the effective `gamma * lambda` credit horizon. The next intervention must therefore make failed
near-tail acquisition terminal within one rollout, rather than further increasing the distant time-out penalty.

The resulting `missed_grasp` term latches when both tails first enter the `0.012 m` acquisition range and emits an
exclusive failure if bilateral acquisition is still absent eight control steps later. A deterministic level-57
open policy now terminates at step 8 with a `-12` failure instead of waiting until step 600. In an otherwise
identical eight-GPU probe, the left-gripper first-step gradient reversed from `-0.603` to `+0.845`, and its aggregate
gradient reversed from `-0.0165` to `+0.0302`, both now pushing toward close. Four PPO updates moved the level-57
first gripper outputs from `(0.0184, -0.00884)` to `(-0.00554, -0.01581)` and changed deterministic bilateral
acquisition from 0/128 to 128/128. Level 56 retained 128/128 acquisition and 68/128 success. This is the first
validated policy crossing of the former acquisition frontier.

Six additional updates kept level-57 acquisition complete, but did not yet produce repeatable untying: one
checkpoint reached 1/128 deterministic success and the next returned to zero. A reset-position sweep with the
early continuation checkpoint separated this post-acquisition failure from the gripper decision. At `0.005125 m`
and `0.00515625 m`, deterministic success was 81/128 and 64/128, while every tested reset from `0.0051875 m`
through `0.00525 m` produced 0/128 success despite complete acquisition. The curriculum therefore adds only the
verified bridge states `0.0050625`, `0.005125`, and `0.00515625 m` before `0.006 m`, increasing its total size from
65 to 68 levels. The first zero-success state remains outside the curriculum until the bridge policy improves.

A six-update, eight-GPU continuation from the early bridge checkpoint validated the 68-level curriculum. Online
frontier windows promoted from new level 57 through level 59 with success rates of approximately 33.9%, 52.3%,
and 59.7%, then reached old `0.006 m` at new level 60. Fixed-level deterministic gates confirmed that this was
real policy coverage rather than only stochastic curriculum movement. The final two checkpoints produced 56/128
and 58/128 successes at level 57, 71/128 and 79/128 at level 58, and 65/128 and 62/128 at level 59. Both produced
0/128 successes at level 60. Every screened episode at all four levels acquired both tails; level-60 failures were
acquired-then-lost rather than missed acquisition or time-out. The verified stable frontier has therefore advanced
from old level 56 to new level 59, while `0.006 m` is the next measured post-acquisition curriculum boundary. Run
artifacts are under
`logs/rsl_rl/shoelace_dual_franka/2026-09-05_21-18-34_curriculum68_missedgrasp8_resume_m608_level57frac0p75_8gpu_6`.

Re-measuring the next boundary with the final actor produced a monotonic local transition: reset positions
`0.00515625`, `0.0051640625`, `0.005171875`, `0.0051796875`, and `0.0051875 m` yielded 53, 41, 15, 1, and 0
successes out of 128, respectively, with bilateral acquisition in every episode. The last point above the 30%
promotion gate is therefore `0.0051640625 m`. Two bridge states at `0.00516015625` and `0.0051640625 m` are added,
bringing the curriculum to 70 levels. The first below-gate point remains excluded until the new frontier improves.

The first six-update continuation on the 70-level curriculum promoted through levels 60 and 61. Deterministic
checks of `model_618.pt` yielded 49/128 successes at level 60, 40/128 at level 61, and 0/128 at level 62; every
episode acquired both tails. Eight more updates at level 61 produced `model_625.pt`. A four-seed deterministic
comparison measured 192/512 successes (37.5%) for `model_625.pt`, versus 149/512 (29.1%) for its input
`model_618.pt`, an 8.4 percentage-point improvement. The accepted checkpoint is
`logs/rsl_rl/shoelace_dual_franka/2026-09-05_21-43-02_curriculum70_missedgrasp8_resume_m618_level61frac0p75_8gpu_8_retry1/model_625.pt`.

A second eight-update continuation from that checkpoint briefly promoted online sampling to level 62, but did not
consolidate the harder state. `model_630.pt` produced only 1/128 deterministic level-62 successes and the final
`model_632.pt` returned to 0/128, while its level-61 result regressed to 39/128. The later checkpoints are therefore
rejected rather than selected from mixed-level online reward. Those artifacts are under
`logs/rsl_rl/shoelace_dual_franka/2026-09-05_21-52-41_curriculum70_missedgrasp8_resume_m625_level61frac0p75_8gpu_8`.

Re-measuring the accepted actor's frontier at seed 55 yielded 45, 41, 38, 26, 25, 0, 0, and 0 successes out of
128 at reset positions `0.0051640625`, `0.005166015625`, `0.00516796875`, `0.005169921875`, `0.005171875`,
`0.00517578125`, `0.0051875`, and `0.006 m`, respectively. Four-seed gates then measured 173/512 (33.8%) at
`0.005166015625 m` and 143/512 (27.9%) at `0.00516796875 m`. Only the first point clears the 30% frontier gate, so
it is inserted before `0.006 m`, bringing the curriculum to 71 levels. The below-gate point remains excluded.

An eight-update, eight-GPU continuation trained the new level 62 from the accepted `model_625.pt`. All online
frontier episodes continued to acquire both tails, but the completed-window success rates oscillated between
16.4% and 25.1% and never reached the 30% promotion threshold. The final `model_632.pt` reproduced the input
actor's 192/512 level-61 successes and measured 172/512 at level 62, versus 173/512 for the input, so it is
rejected. The run is under
`logs/rsl_rl/shoelace_dual_franka/2026-09-05_22-16-06_curriculum71_missedgrasp8_resume_m625_level62frac0p75_8gpu_8_retry4`.

The intermediate `model_627.pt` showed a small but consistent level-62 gain across two independent four-seed
gates: 357/1024 successes versus 324/1024 for the input, or +3.2 percentage points. The approximate confidence
interval still includes zero, and a finer physical-frontier check did not justify another curriculum insertion.
At reset positions `0.0051669921875` and `0.00516796875 m`, `model_627.pt` reached 149/512 (29.1%) and 151/512
(29.5%) successes, respectively, both just below the 30% gate. No continuation checkpoint is therefore accepted
as a new stable frontier, and no training process remains active after these evaluations.

Trajectory accounting localizes the remaining gap after bilateral acquisition. Lost-grasp episodes still earned
positive dense task credit during their final 16-step credit horizon because the task potential measures throat
occupancy, tail-to-knot distances, and tail separation but has no continuous tail-to-TCP retention margin. The
hard grasp mask only removes shaping after a tail is already outside the strict grasp region, and the failure
impulse arrives after release confirmation. The next controlled ablation should therefore add retention awareness
inside the existing reset-relative dense task term and validate it against fixed checkpoint gates before another
long curriculum run; a separate end-effector tracking objective is not indicated by this evidence.

A retention-aware composite reward ablation multiplied task progress by the weakest continuous tail-to-TCP and
finger-aperture retention margin. Fixed-policy replay confirmed that it corrected the accounting problem: the
dense return of an acquired-then-lost trajectory fell from 8.347 to 6.670 and its final 16-step dense return fell
from 2.812 to 1.319, while a successful trajectory remained effectively unchanged at 11.022 instead of 11.014.
However, neither an eight-update policy-only warm start nor a critic-only burn-in followed by eight policy updates
improved the accepted actor. Their online frontier rates remained below 24%, and the first warm-start candidate
measured 144/512 level-62 successes versus 169/512 for its input. The reward modification is therefore rejected
and was removed from the task source. The two policy runs are under
`logs/rsl_rl/shoelace_dual_franka/2026-09-05_22-45-34_curriculum71_retentioncomposite_policywarmstart_m625_level62frac0p75_8gpu_8`
and
`logs/rsl_rl/shoelace_dual_franka/2026-09-05_23-05-26_curriculum71_retentioncomposite_aftercritic_m625_level62frac0p75_8gpu_8`.

The next ablation tested whether action clipping suppressed useful arm exploration. At level 62, the accepted
actor's Gaussian means lay outside the executable range on approximately 20.3% of left and 61.7% of right wrist-x
rotation samples. Sampling around a straight-through clipped arm mean increased the fraction of nonzero executed
right-wrist residuals from 38.6% to 69.4%, but a fixed four-seed comparison before training produced 129/512
successes versus 132/512 for the original distribution. An eight-update, eight-GPU continuation likewise stayed
between approximately 12.4% and 21.5% online frontier success. A deterministic seed-56 screen of its eight
checkpoints produced 34, 42, 32, 29, 39, 29, 34, and 37 successes out of 128; the accepted input checkpoint
repeated at 37--39/128. This is not a robust improvement, so the bounded sampling prototype is also rejected and
was not added to source. Its artifacts are under
`logs/rsl_rl/shoelace_dual_franka/2026-09-05_23-19-42_curriculum71_clipcenter_m625_level62frac0p75_8gpu_8`.
The accepted frontier therefore remains level 62 with the earlier `model_625.pt`, and no training process is
active after these gates.

### World-origin sensitivity and gripper preload repair

The remaining level-62 failures were traced to physical contact margin rather than a missing task reward. At an
environment spacing of `0.25 m`, environments near the stage origin almost always lost the tails while outer rows
often succeeded. Broadcasting the exact same outer-environment action to all 128 environments still produced
65 successes, whereas broadcasting a center-environment action produced none. The split therefore persisted after
removing policy-observation differences. Disabling either ADMM Baumgarte stabilization or contact matching caused
all 128 trajectories to lose grasp, so those mechanisms are required by the learned motion rather than the source
of the spatial split.

A direct one-contact Newton ADMM reproduction translated otherwise identical rigid-particle systems without
changing their relative geometry. After 120 steps, translating the system by up to `1.375 m` changed its relative
trajectory by as much as `9.02e-6 m` on both CPU and GPU. Inspection of Newton's contact kernels confirmed that the
soft-contact body position is authored and consumed in body-local coordinates; the evidence does not support a
world/local-frame mix-up. Instead, ordinary single-precision cancellation is comparable to the `1--8 um` reset
increments near the existing curriculum frontier.

The authored cable radius is `0.0015 m`, but the previous closed command was `0.002 m` per Panda finger. Its
nominal two-finger aperture was consequently `0.004 m`, leaving `0.001 m` clearance around the `0.003 m` cable and
making retention depend on micrometre-scale numerical bias. A fixed-policy sweep at level 62 and zero environment
spacing isolated the contact margin:

| Configuration | Successes | Lost grasps |
| --- | ---: | ---: |
| Previous `0.002 m` close target | 9/128 | 119/128 |
| Soft-contact friction `2` | 13/128 | 115/128 |
| Soft-contact friction `10` | 13/128 | 115/128 |
| `0.0015 m` close target | 128/128 | 0/128 |
| `0.0010 m` close target | 45/128 | 83/128 |
| `0.0005 m` close target | 0/128 | 128/128 |
| ADMM Baumgarte `0.75` | 128/128 | 0/128 |
| Ten ADMM iterations | 128/128 | 0/128 |

The non-monotonic close-target result rules out a generic "more closure is better" explanation. The selected
`0.0015 m` action target matches the cable radius, adds physical preload without over-compression, and avoids the
global solver-cost and behavior changes of the two ADMM alternatives. It produced 128/128 deterministic successes
at each of `0`, `0.25`, and `0.5 m` environment spacing. The reset curriculum remains anchored at its original
`0.002 m` closed state, so all 71 absolute reset positions and saved curriculum levels retain their meaning.

Frontier checks with the selected target produced 125/128 successes at level 0, 122/128 at level 55, 128/128 at
level 62, and 24/128 at the next `0.006 m` reset state (level 63). Every level-63 trajectory acquired both tails;
the remaining failures were post-acquisition releases. At levels 64 and 65 (`0.008` and `0.010 m`), no trajectory
acquired both tails, so those remain suitable later curriculum stages. Because the action-to-physics contract has
changed, the next policy-training result must come from a fresh run rather than resuming the old actor.

After landing the separate action target in the task configuration, a no-override source replay at seed 73
reproduced the diagnostic result: all 128 level-62 episodes succeeded, while level 63 produced 23 successes and
105 acquired-then-lost episodes. The configuration contract regression test fails with the former `0.002 m`
action target and passes with `0.0015 m`, while independently retaining the `0.002 m` reset anchor.

### Fresh hybrid-policy failure and separation-progress deadline

A fresh eight-GPU run tested the physical preload together with the Bernoulli gripper policy. The run is stored at
`logs/rsl_rl/shoelace_dual_franka/2026-09-06_00-27-12_admm_rho400_close1p5_hybrid1_curriculum71_fresh_8gpu_300`,
with per-rank output under
`logs/torchrun/shoelace_admm_rho400_close1p5_hybrid1_curriculum71_fresh_8gpu_300`. All ranks remained finite and
throughput was approximately 2160 environment steps per second, but success stayed at zero through iteration 121,
so the planned 300-iteration validation was stopped early. The deterministic gripper-open probability fell from
approximately 48--49% at initialization to 3.0--3.6% by iteration 100, confirming that the hybrid policy learned
the close decision.

The remaining failure was geometric rather than a gripper or distributed-training fault. At level 0, `model_60.pt`
acquired both tails and produced meaningful outward motion, reaching mean maximum tail separation `0.168 m`.
By `model_120.pt`, 91/128 episodes instead lost grasp and 37/128 timed out, with mean maximum separation reduced
to `0.142 m`. Both tail-to-knot distances nevertheless reached their individual dense-reward targets, while
tail-separation progress reached only about 0.44 at 32 acquired steps and then collapsed. The actor had learned to
move both tails away from the knot without maintaining the two-sided pull needed to untie it.

Fixed-policy calibration identified a causal separator that uses task geometry rather than end-effector tracking.
The accepted `model_625.pt` reached at least 0.601 reset-relative tail-separation progress after its 32-step
post-acquisition deadline in every successful level-63 trajectory in a 128-environment gate. In the same no-deadline
gate, 59/103 failed trajectories later fell below 0.5. The regressed fresh `model_120.pt` averaged 0.441 at the
deadline. An initial 32-step deadline retained a measured success margin while moving the regressed actor's failure
from a mean of roughly 506 steps to 32.0 steps. Source gates preserved 120/128 level-0 and 128/128 level-62
successes from the accepted actor; at level 63 they produced 19 successes, 45 grasp losses, and 64 earlier
insufficient-separation failures. The intermediate fresh `model_60.pt` remained above the deadline initially, then
74/128 regressing trajectories received the new failure before their eventual grasp loss.

Fresh training showed that 32 steps was nevertheless too early for exploration. The eight-GPU run at
`logs/rsl_rl/shoelace_dual_franka/2026-09-06_01-16-43_admm_rho400_close1p5_sep50deadline32_hybrid1_curriculum71_fresh_8gpu_300_retry1`
improved the iteration-39 dense reward to `+0.0358` and held mean episode length to 31.7 steps, compared with
`-0.0614` and 143.2 steps for the static-hold run. It still produced no success, and by iteration 60 every episode
ended at the 32-step separation boundary. The run was stopped because the repair had become an exploration wall.
The final deadline is therefore 64 acquired control steps: later than the accepted policy's 47--52-step success
window, but early enough to catch the regressed actor after its separation collapses. A final 128-environment gate
terminated all `model_120.pt` trajectories at a mean of 64.03 steps, preserved 128/128 level-62 successes from
`model_625.pt` at a mean of 49.16 steps, and preserved 21/128 level-63 successes without firing the new term before
the remaining 107 grasp losses. The task now emits the exclusive insufficient-separation failure at or after this
64-step deadline whenever reset-relative tail-separation progress is below 0.5.

### Fresh-policy stabilization and complete startup episodes

The 64-step deadline allowed a fresh eight-GPU policy to discover success, but ordinary PPO did not retain the
solution. The run at
`logs/rsl_rl/shoelace_dual_franka/2026-09-06_01-28-32_admm_rho400_close1p5_sep50deadline64_hybrid1_curriculum71_fresh_8gpu_300`
first succeeded at iteration 53, reached approximately 25--28% online success around iterations 67--71, then fell
to zero by iteration 95. It was stopped at iteration 101. Fixed level-0 gates selected `model_60.pt`: four seeds
produced 449/512 deterministic successes, whereas `model_70.pt` produced 343/512 and checkpoints 80 through 100
had already collapsed. With arm standard deviation fixed at 0.10, five additional seeds produced 287/640
successes (44.8%); the same actor's learned arm standard deviations were approximately 0.28--0.31. The missing
promotion was therefore caused by exploration noise and policy regression rather than a missing success signal.

A first stabilization continuation fixed the arm standard deviation at 0.10, froze observation statistics, reset
Adam, used one PPO epoch with a fixed `3e-5` learning rate and zero entropy coefficient, and lowered the
experimental promotion gate from 50% to 40%. All ranks reached level 1, but the shared actor still forgot level 0:
at seed 101 the input actor produced 111/128 level-0 successes, while `model_80.pt` and `model_90.pt` produced only
51/128 and 66/128. The run was stopped at iteration 92 and is stored under
`logs/rsl_rl/shoelace_dual_franka/2026-09-06_01-58-41_fresh64_m60_std010_fixednorm_lr3e5_epoch1_thresh40_8gpu_40_retry1`.

This continuation also exposed a curriculum-accounting bug at startup. RSL-RL's default
`init_at_random_ep_len=True` randomized the time-limit counter without advancing the physical shoelace state. The
first frontier window consequently contained artificial short time-outs and reported only 19.85% success; eight
policy updates occurred before complete trajectories produced the valid 42.97% window. Shoelace training now
starts with complete episodes. The task-specific regression test first failed with the inherited `True` value and
passes with `init_at_random_ep_len=False`.

The corrected stabilization run additionally froze the actor's hidden backbone, used a fresh Adam state and a
fixed `2.5e-6` learning rate, and retained one PPO epoch, zero entropy coefficient, fixed observation statistics,
and arm standard deviation 0.10. Without randomized startup ages, every rank reached level 1 at iteration 70,
eight updates earlier than the shared-actor run, and aggregate success stayed near 39--47% through iteration 83
with no unsafe termination or time-out. A five-seed level-1 gate measured 349/640 successes for `model_83.pt`
versus 288/640 for its input, an absolute gain of 9.53 percentage points; a conservative unpaired normal
approximation placed the 95% interval at approximately +4.08 to +14.98 points. A paired level-0 gate remained
unchanged at 112/128 versus 110/128. The accepted stabilized checkpoint is
`logs/rsl_rl/shoelace_dual_franka/2026-09-06_02-11-14_fresh64_m60_std010_frozenbackbone_lr2p5e6_epoch1_norand_thresh40_8gpu_24_retry1/model_83.pt`.

The accepted mean policy is not yet robust to the training distribution. Four stochastic level-1 gates with arm
standard deviation 0.10 produced only 113/512 successes (22.1%). Reducing the arm standard deviation to 0.01
improved five-seed stochastic success to 197/640 (30.8%), while every episode still acquired both tails. A
16-update continuation then concentrated 75% of resets on level 1 with 25% level-0 replay and a 30% promotion gate.
Its completed online frontier windows reached only 21.6% and 26.6%, so it did not promote. Fixed screening selected
the intermediate `model_94.pt`, but a five-seed comparison measured only 364/640 successes versus 334/640 for its
input, a +4.69-point difference whose approximate 95% interval still included zero. This low-noise continuation is
rejected and remains under
`logs/rsl_rl/shoelace_dual_franka/2026-09-06_02-30-52_fresh64_m83_level1frac75_std001_frozenbackbone_lr2p5e6_epoch1_norand_thresh30_8gpu_16`.
The accepted frontier therefore remains `model_83.pt` at level 1. No training process remains active after these
gates.

Tracing the hybrid actor then isolated the remaining stochastic failure to the Bernoulli gripper decisions rather
than the arm trajectory. At the former gripper logit scale of 1, the right gripper still had approximately 3.8%
open probability at the first step; rare open samples later in the episode could destroy an otherwise stable
grasp. At level 1 with arm standard deviation 0.01 and seed 124, logit scales 1, 1.5, 2, and 3 produced 33, 60, 63,
and 65 successes out of 128, respectively. Five scale-2 seeds produced 341/640 successes (53.3%), with essentially
no post-acquisition open samples. This retains the actor's deterministic action boundary while making its sampled
gripper behavior consistent with the required persistent hold. The task now uses scale 2 by default.

An eight-GPU scale-2 continuation from the accepted `model_83.pt` concentrated 75% of resets on level 1, retained
25% level-0 replay, froze the actor backbone and observation statistics, used arm standard deviation 0.01, and
started every environment with a complete episode. Its first populated promotion window reached 40.62% success at
iteration 99 and all eight ranks advanced from level 1 to level 2. At the final iteration 102, aggregate episode
terminations were 57.37% success, 40.28% lost grasp, and 2.34% insufficient separation, with no unsafe termination
or time-out. The run completed normally under
`logs/rsl_rl/shoelace_dual_franka/2026-09-06_02-48-05_fresh64_m83_level1frac75_std001_scale2_frozenbackbone_lr2p5e6_epoch1_norand_thresh40_8gpu_20`.
Because only the final few iterations sampled level 2, this establishes a qualified level-1 promotion but not yet
a retained level-2 policy. The accepted checkpoint remains `model_83.pt` until a fixed multi-seed gate shows that
one of the new checkpoints preserves level 0 and level 1 while improving level 2.

### Contact-margin bridge and critic warm-up

Direct continuation at the former level-2 reset (`0.003 m` driven finger position) failed before a clean
promotion window could form. A fixed-policy reset sweep showed that this was a sharp contact-margin transition,
not missing task progress: stochastic `model_83.pt` trajectories all acquired both tails at step 3, but success
fell from 192/384 at `0.0029375 m` to 76/384 at `0.0029453125 m` across seeds 158--160. The calibrated
`0.0029375 m` state is therefore inserted between `0.0025` and `0.003 m`, bringing the curriculum to 72 levels.
The former `0.003 m` state is now level 3. The nearby below-gate point remains excluded because its success was
non-monotonic and did not meet the 50% diagnostic target.

Several low-rate actor continuations on the new bridge still degraded before their first valid frontier window.
This suggested that a cold value estimate, rather than an absent reward or grasp signal, could be producing poor
early advantages. A task-local fully frozen actor was added to test that hypothesis without changing policy
outputs or observation statistics. The 16-update critic-only run at
`logs/rsl_rl/shoelace_dual_franka/2026-09-06_03-37-21_curriculum72_m83_level3_criticwarmup_lr3e4_8gpu_16_retry1`
left every actor tensor bit-identical and changed only the critic. On a common level-3 trajectory, overall critic
mean absolute error improved from 13.21 to 12.26 and lost-grasp error improved from 4.62 to 2.64, but success error
worsened from 35.43 to 37.12 and explained variance remained near zero. The critic had mostly learned a failure
baseline, so critic warm-up alone was not an adequate repair.

An actor continuation from that critic at
`logs/rsl_rl/shoelace_dual_franka/2026-09-06_03-46-47_curriculum72_criticwarmup_m98_level3frac100_std001_scale2_frozenbackbone_lr2p5e6_8gpu_12`
avoided the earlier immediate collapse, but a paired stochastic multi-seed gate measured 76/384 successes for its
selected checkpoint versus 74/384 for the input. The change was rejected as noise-level. One earlier critic-only
launch failed during native Newton USD parsing on rank 4 and produced no PPO updates; the successful retry above
is the run used for the analysis.

### Contact-stable continuous action scale

Action tracing isolated a stronger causal variable: the learned arm residuals were too large for marginal cable
contact. Holding the checkpoint, reset, gripper policy, standard deviation, and terminations fixed, a post-policy
arm scale of 0.7 raised stochastic success at the `0.0029453125 m` diagnostic reset from 97/512 to 267/512 across
four paired seeds. A retention matrix at levels 0--3 produced 108/65/51/25 successes at scale 1 and
88/58/97/64 at scale 0.7. The smaller motion trades some excess margin on the easiest states for much stronger
retention at the frontier.

The shoelace hybrid distribution now applies this factor to both the continuous arm means and their Gaussian
noise, while leaving both Bernoulli gripper dimensions unchanged. Applying the factor inside the distribution
keeps PPO likelihoods and deterministic export consistent with executed actions. The source-path validation at
`0.0029453125 m` improved from 117/512 to 253/512, and at the actual `0.003 m` level-3 reset from 125/512 to
277/512, across paired seeds 168--175. Existing checkpoints can recover the former output contract with
`agent.actor.distribution_cfg.arm_action_scale=1.0`.

The scale-0.7 critic warm-up at
`logs/rsl_rl/shoelace_dual_franka/2026-09-06_04-22-51_curriculum72_m83_scale070_level2frac75_criticwarmup_lr3e4_8gpu_16`
completed normally with an unchanged actor and approximately 73.4% aggregate success. A subsequent 24-update
actor run under
`logs/rsl_rl/shoelace_dual_franka/2026-09-06_04-26-42_curriculum72_scale070_aftercritic_m98_level2frac75_std001_frozenbackbone_lr2p5e6_thresh40_8gpu_24`
held aggregate success near 66--67%. Paired level-2 gates measured 200/256 successes for `model_121.pt` versus
197/256 for its input; level-3 gates measured 119/256 versus 123/256. The actor was effectively retained rather
than measurably improved.

A 32-update continuation then used 100% level-2 samples so a complete frontier window would form before policy
drift. It is stored at
`logs/rsl_rl/shoelace_dual_franka/2026-09-06_04-35-00_curriculum72_scale070_m121_level2frac100_std001_frozenbackbone_lr2p5e6_thresh40_8gpu_32`.
The first window reached 71.09% success and all eight ranks promoted to level 3 at iterations 134--135. The run
finished normally at `model_152.pt`, with no unsafe termination or time-out. A new four-seed pure level-3 gate
measured 251/512 successes for `model_152.pt` versus 257/512 for `model_121.pt`; two seeds improved and two
regressed. The 1.17-point aggregate difference is not credible evidence of either learning or collapse, so the
result is accepted as a retained policy plus a validated level-2 promotion, not as actor improvement.

The next continuation started directly from `model_152.pt` with 100% level-3 resets. Three ordinary eight-rank
startup attempts failed before PPO while different ranks concurrently parsed USD in Newton; their native stacks
ended in `newton._src.utils.import_usd.parse_usd`. Staggering worker startup by five seconds per local rank avoided
that parser race without changing the environment or training configuration. The valid run is stored at
`logs/rsl_rl/shoelace_dual_franka/2026-09-06_04-58-23_curriculum72_scale070_m152_level3frac100_std001_frozenbackbone_lr2p5e6_thresh40_admm_8gpu_32_retry7_stagger5`.
Its pure level-3 window reached 53.12% success and all ranks promoted to level 4 at iteration 166. Fixed four-seed
retention gates measured 241/512 level-3 successes for `model_183.pt` versus 244/512 for its input, and 406/512
level-4 successes versus 409/512. Each aggregate changed by only three episodes, so `model_183.pt` is accepted as
the next retained continuation point. The unexpectedly easier `0.0035 m` level 4 also confirms that micrometre
contact difficulty is locally non-monotonic; promotion decisions continue to require a populated per-level window
rather than reset position alone.

A 32-update pure level-4 continuation from `model_183.pt` is stored at
`logs/rsl_rl/shoelace_dual_franka/2026-09-06_05-09-40_curriculum72_scale070_m183_level4frac100_std001_frozenbackbone_lr2p5e6_thresh40_admm_8gpu_32_stagger5`.
Its completed frontier window reached 80.47% success and all eight ranks promoted to level 5 at iteration 197.
At the final iteration, which sampled level 5, aggregate terminations were 76.17% success, 17.04% lost grasp, and
6.79% insufficient separation, with no unsafe termination or time-out. A paired four-seed stochastic level-5 gate
measured 386/512 successes for `model_214.pt` versus 389/512 for `model_183.pt`. The three-episode difference is
again noise-level, so `model_214.pt` is retained as a validated level-4 promotion rather than evidence of actor
improvement.

The subsequent 32-update pure level-5 run is stored at
`logs/rsl_rl/shoelace_dual_franka/2026-09-06_05-20-45_curriculum72_scale070_m214_level5frac100_std001_frozenbackbone_lr2p5e6_thresh40_admm_8gpu_32_stagger5`.
Its level-5 window reached 78.91% success and all ranks promoted to level 6 at iteration 228. The final online
level-6 mixture reached 83.98% success, 12.89% lost grasp, and 3.12% insufficient separation, again without an
unsafe termination or time-out. Fixed level-6 gates over seeds 222--225 measured 405/512 successes for the input
`model_214.pt`, compared with 391/512, 375/512, and 389/512 for `model_228.pt`, `model_236.pt`, and
`model_245.pt`. The trained output layer moved by at most approximately `1e-4`, but each sampled checkpoint
increased lost-grasp relative to the input. The actor updates are therefore rejected, while the level-5 promotion
is retained. Curriculum frontier search continues from `model_214.pt` with the entire actor frozen; this preserves
the best measured policy while identifying the first reset level that actually requires new policy learning.

### Frozen-policy frontier search

The first frozen-policy search ran for 300 updates under
`logs/rsl_rl/shoelace_dual_franka/2026-09-06_05-33-57_curriculum72_scale070_m214_level6frac100_std001_frozenpolicy_lr3e4_thresh40_admm_8gpu_300_stagger5`.
It used 100% frontier resets, a fresh critic optimizer at `3e-4`, and the same stochastic action contract as the
accepted policy. The actor in `model_513.pt` is tensor-for-tensor identical to `model_214.pt`; all six critic MLP
tensors changed and remained finite. The run completed 4,915,200 transitions without an unsafe termination,
time-out, native Newton failure, or distributed failure.

| Promoted from level | Completed window success | All ranks at next level by iteration |
|---:|---:|---:|
| 6 | 74.42% | 227 |
| 7 | 75.78% | 252 |
| 8 | 57.03% | 276 |
| 9 | 51.16% | 299 |
| 10 | 57.03% | 321 |
| 11 | 53.12% | 343 |
| 12 | 51.56% | 365 |
| 13 | 54.69% | 387 |
| 14 | 51.56% | 409 |
| 15 | 62.50% | 431 |
| 16 | 58.91% | 454 |
| 17 | 62.50% | 477 |
| 18 | 57.36% | 499 |

The unchanged policy therefore crossed every tested level from 6 through 18; the prior trainable output-head
continuations were unnecessary and introduced contact-regressing drift. The run ended while collecting the first
pure level-19 window, where online success was 63.53%, lost grasp was 35.69%, and insufficient separation was
0.78%. A subsequent frozen-policy segment resumes at level 19 to continue locating the first genuinely unsolved
reset rather than treating the run-length boundary as a curriculum failure.

### Second frozen-policy search and contact bridge

The next frozen-policy segment resumed at level 19 under
`logs/rsl_rl/shoelace_dual_franka/2026-09-06_06-12-11_curriculum72_scale070_m513_level19frac100_std001_frozenpolicy_lr3e4_thresh40_admm_8gpu_300_stagger5`.
It crossed every tested reset through level 29 without changing the actor:

| Promoted from level | Completed window success | All ranks at next level by iteration |
|---:|---:|---:|
| 19 | 69.53% | 528 |
| 20 | 62.79% | 552 |
| 21 | 57.81% | 576 |
| 22 | 47.29% | 598 |
| 23 | 51.94% | 621 |
| 24 | 53.91% | 644 |
| 25 | 56.25% | 667 |
| 26 | 52.34% | 689 |
| 27 | 60.94% | 713 |
| 28 | 64.84% | 736 |
| 29 | 62.31% | 759 |

Level 30 then produced a complete 23.26% window, with approximately 29.9% online success, 62.3% lost grasp,
and 8.6% insufficient separation. Training was intentionally stopped at iteration 782 rather than spending actor
updates at a now-measured frontier. The last periodic checkpoint, `model_780.pt`, retains an actor that is
tensor-for-tensor identical to `model_214.pt`; all six critic MLP tensors changed and remained finite.

A four-seed fixed-policy comparison reproduced the discontinuity. The `0.003625 m` level-29 reset achieved
322/512 successes (62.9%), while the `0.003750 m` level-30 reset achieved 167/512 (32.6%). Every episode at both
resets acquired both tails at approximately policy step 3, but level 30 increased lost-grasp from 186 to 306 and
insufficient-separation from 4 to 39. The frontier is therefore post-acquisition contact retention rather than a
missing acquisition or success signal.

An eight-point fixed-policy sweep showed that this contact boundary is strongly non-monotonic. Success at reset
positions `0.003625`, `0.003640625`, `0.003656250`, `0.003671875`, `0.003687500`, `0.003703125`, `0.003718750`,
and `0.003750 m` was respectively 56.2%, 39.1%, 29.7%, 32.0%, 42.2%, 48.4%, 48.4%, and 34.4% for a common
128-environment seed. A four-seed confirmation of `0.003718750 m` measured 262/512 successes (51.2%), with every
seed above the 40% promotion threshold; including the sweep seed gives 324/640 (50.6%). This closest stable bridge
is inserted before `0.003750 m`, increasing the curriculum to 73 levels.

An eight-GPU frozen-policy validation then started at the new level 30 under
`logs/rsl_rl/shoelace_dual_franka/2026-09-06_07-00-47_curriculum73_scale070_m780_level30frac100_std001_frozenpolicy_lr3e4_thresh40_admm_8gpu_64_stagger5`.
The bridge produced a complete 46.09% window and promoted every rank to level 31 by iteration 795. The shifted
`0.003750 m` frontier then produced a 29.69% complete window, with approximately 35.3% online success, 57.5%
lost grasp, and 8.1% insufficient separation. The run was intentionally stopped after iteration 823; its last
periodic checkpoint is `model_820.pt`. That checkpoint's actor remains tensor-for-tensor identical to
`model_214.pt`, and its six updated critic tensors are finite. This confirms that the added state bridges the
known contact discontinuity without hiding it, and localizes subsequent actor learning to level 31.

A trainable-output-head continuation from `model_820.pt` then tested whether PPO could improve that measured
frontier. It ran under
`logs/rsl_rl/shoelace_dual_franka/2026-09-06_07-09-34_curriculum73_scale070_m820_level31frac100_std001_frozenbackbone_lr2p5e6_thresh40_admm_8gpu_300_stagger5`
with a fresh Adam state, fixed `2.5e-6` learning rate, and the actor backbone and observation statistics frozen.
Its complete level-31 windows oscillated around 30--37% before falling to approximately 26%, so it was stopped
after iteration 877. A common-seed screen found fewer successes for every sampled candidate than the input except
for the near-tied `model_850.pt`. A final paired gate over seeds 268--271 measured 189/512 successes for the input
and 163/512 for `model_850.pt`; insufficient-separation failures also increased from 33 to 54. All actor updates
are therefore rejected, and `model_820.pt` remains the accepted checkpoint.

Controlled fixed-policy tests next isolated solver position correction as the missing retention margin. Reducing
the arm-action scale below 0.7 converted failures to insufficient separation, while larger scales increased grasp
loss; 0.7 remained the best sampled value. Increasing ADMM interface iterations from 5 to 10 was also strongly
harmful, reducing a four-seed gate from 178/512 successes to zero. In contrast, raising ADMM Baumgarte correction
from 0.5 to 0.75 produced 74, 76, 74, and 77 successes out of 128 for the same four seeds: 301/512 (58.8%) versus
178/512 (34.8%) at the old value. Lost-grasp fell from 299 to 125, while insufficient-separation rose from 35 to
86. The net 24-point cross-seed improvement justifies using 0.75 as the task's ADMM default before resuming the
frozen-policy curriculum search.

### Baumgarte-0.75 frontier search

The frozen actor resumed from `model_820.pt` with Baumgarte correction 0.75 under
`logs/rsl_rl/shoelace_dual_franka/2026-09-06_07-40-37_curriculum73_scale070_m820_level31frac100_std001_frozenpolicy_lr3e4_thresh40_admm_b075_8gpu_300_stagger5`.
Level 31 completed a 57.03% window and promoted every rank to level 32 by iteration 835; level 32 completed a
58.59% window and promoted every rank to level 33 by iteration 861. The unchanged actor then produced a 12.50%
complete window at level 33, with online outcomes around 24% success, 51% lost grasp, and 26% insufficient
separation. The run was stopped after iteration 878, and its last periodic checkpoint is `model_875.pt`.

The level-33 reset is `0.003875 m`. Fixed-policy sweeps at eight positions from `0.0038203125` through
`0.003875 m` produced 25.0--31.3% success for one common seed. A second sub-micron sweep from
`0.0038134766` through `0.0038203125 m` produced 23.4--32.0% success. None cleared the 40% threshold, so another
reset bridge would not be supported by the measured data; level 33 is the first frontier that requires policy or
contact-control improvement rather than additional curriculum density.

### Baumgarte-0.75 output-head gate

A trainable-output-head continuation from `model_875.pt` ran under
`logs/rsl_rl/shoelace_dual_franka/2026-09-06_07-57-01_curriculum73_scale070_b075_m875_level33frac100_std001_frozenbackbone_lr2p5e6_thresh40_admm_8gpu_300_stagger5`.
It used a fresh Adam state at fixed `2.5e-6`, a frozen actor backbone and observation statistics, and 100% level-33
resets. Complete windows rose from approximately 16.8% to peaks around 35.7% but never cleared 40%; the run was
stopped after approximately 67 updates. A common-seed checkpoint screen selected `model_940.pt` as the apparent
best candidate at 46/128 successes versus 31/128 for the input. A four-new-seed gate reversed that apparent gain:
the input produced 142/512 successes (27.7%), 248 lost grasps, and 122 insufficient-separation failures, while
`model_940.pt` produced 131/512 successes (25.6%), 247 lost grasps, and 134 insufficient-separation failures.
The actor update is therefore rejected and `model_875.pt` remains the accepted continuation checkpoint.

A fixed-policy deadline sweep then checked whether the 64-step separation-progress termination hid delayed
successes under the stronger Baumgarte correction. For one seed, deadlines 64, 80, 96, 128, 160, 256, 400, and
600 produced respectively 28, 35, 34, 33, 37, 32, 29, and 27 successes out of 128. Deadlines of 256 or more
converted every insufficient-separation outcome to lost grasp without improving success. A four-new-seed 64/80
comparison measured 121/512 (23.6%) versus 136/512 (26.6%) successes, but the gain was concentrated in one seed
and two seeds slightly regressed. This is insufficient evidence to change the default deadline: a small delayed
success region exists, but the dominant level-33 failure remains post-acquisition contact retention.

### Long-horizon and arm-bias diagnosis

The level-33 successes terminate at approximately policy step 196, while the current 16-step PPO rollout gives
the critic only a short GAE credit path. A controlled 12-update continuation therefore changed only the rollout
horizon to 80 under
`logs/rsl_rl/shoelace_dual_franka/2026-09-06_08-26-47_curriculum73_scale070_b075_m875_level33frac100_std001_frozenbackbone_lr2p5e6_thresh40_admm_h80_8gpu_12`.
The complete online window peaked at 33.6%, and a common-seed screen selected `model_880.pt` at 44/128 successes
versus 31/128 for the input. A four-new-seed gate rejected that apparent gain: the input produced 149/512
successes (29.1%), while `model_880.pt` produced 146/512 (28.5%). A longer rollout made online accounting smoother
but did not supply a reliably improving output-head gradient.

A causal sweep then changed one arm output bias at a time without training. At level 33, subtracting 0.01 from the
left `dx` output improved a four-seed gate from 136/512 to 171/512 successes. A magnitude sweep localized a second
stable point at `-0.03`; its four-new-seed gate improved every seed and increased success from 119/512 (23.2%) to
209/512 (40.8%), while reducing lost grasp from 259 to 221 and insufficient separation from 134 to 82. The
executed action change is only `-0.021` after the 0.7 arm scale, or approximately `-105 micrometres` per policy
step at the configured Cartesian translation scale. A level-32 safety gate remained well above the promotion
threshold at 277/512 successes (54.1%), versus 294/512 (57.4%) for the input.

The frozen-policy validation of this warm start ran under
`logs/rsl_rl/shoelace_dual_franka/2026-09-06_09-13-37_curriculum73_scale070_b075_m875_ldxminus003_level33frac100_std001_frozenpolicy_lr3e4_thresh40_admm_8gpu_120_stagger5`.
Level 33 promoted on a 42.09% completed window and every rank reached level 34 by iteration 900. Level 34 promoted
on a 41.09% window and every rank reached level 35 by iteration 930. The first level-35 windows then measured
approximately 20.3--26.6%, so the run was stopped after iteration 960. The actor in `model_960.pt` is bitwise
identical to the validated bias warm start, all six critic tensors changed and remained finite, and no unsafe or
time-out termination occurred.

A second per-dimension arm-bias sweep at level 35 produced several single-seed candidates near 39%, but a
same-seed combination test rejected every additional bias: the unmodified warm start produced 34/128 successes,
while the best single candidate produced 33/128 and all combinations were lower. A reset-only sweep instead
localized the next contact discontinuity. Positions `0.0039453125`, `0.003953125`, `0.0039609375`,
`0.00396875`, `0.0039765625`, `0.003984375`, and `0.0039921875 m` produced respectively 60, 50, 57, 58, 59, 45,
and 37 successes out of 128 for a common seed. Four-seed confirmation measured 221/512 successes (43.2%) at
`0.0039765625 m`, compared with 205/512 (40.0%) at `0.00396875 m`. The closer and more robust position is inserted
before `0.004 m`, increasing the curriculum to 74 levels; the former level 35 becomes level 36.

A frozen-policy online validation of the 74-level curriculum ran under
`logs/rsl_rl/shoelace_dual_franka/2026-09-06_09-49-14_curriculum74_scale070_b075_m960_level35frac100_std001_frozenpolicy_lr3e4_thresh40_admm_8gpu_64_stagger5`.
The new level 35 first produced complete success windows of 38.76%, 38.46%, 39.33%, 39.84%, and 38.28%, then
cleared the 40% experimental promotion threshold with a 41.41% window; every rank reached level 36 by iteration
1016. The run ended at iteration 1023, only eight updates after the transition, so it did not collect a complete
level-36 window. The final actor remained bitwise identical to `model_960.pt`; all six updated critic tensors were
finite, and no unsafe or time-out termination occurred. This validates the inserted reset as a traversable bridge
without attributing the transition to policy learning, while leaving the `0.004 m` frontier unresolved.

A dedicated frozen-policy level-36 validation then ran under
`logs/rsl_rl/shoelace_dual_franka/2026-09-06_10-01-37_curriculum74_scale070_b075_m960_level36frac100_std001_frozenpolicy_lr3e4_thresh40_admm_8gpu_48_stagger5`.
Its first two complete windows measured 28.91% and 24.22% success, both clearly below the 40% experimental
threshold. At the second window the aggregate rates were approximately 24.1% success, 37.4% lost grasp, and 40.0%
insufficient separation, with zero missed-grasp, unsafe, and time-out terminations. The run was intentionally
stopped after iteration 985; its actor was still bitwise identical to the input and all critic tensors were finite.
The remaining boundary is therefore post-acquisition retention and pull progress at `0.004 m`, not initial grasp
acquisition or numerical instability.

The first output-head continuation at level 36 unintentionally restored the optimizer from the critic-only input
checkpoint. RSL-RL's default `runner.load()` contract restores actor, critic, optimizer, and iteration together;
the saved optimizer param group retained its `3e-4` learning rate and overrode the requested `2.5e-6`. Under
`logs/rsl_rl/shoelace_dual_franka/2026-09-06_10-07-14_curriculum74_scale070_b075_m960_level36frac100_std001_frozenbackbone_lr2p5e6_thresh40_admm_8gpu_300_stagger5`,
the first update moved individual output rows by approximately `0.005--0.011` in L2 norm. The second complete
window collapsed to 1.56% success with approximately 86.7% lost grasp, so the run was stopped after iteration
984. A same-seed fixed-policy screen measured 31/128 successes for the input and only 7/128, 1/128, and 4/128 for
`model_965.pt`, `model_970.pt`, and `model_975.pt`; every actor update from this run is rejected.

The corrected warm-start artifact keeps the input actor and critic bitwise unchanged, clears the Adam state, and
sets the saved optimizer param-group rate to `2.5e-6`. The corrected 300-update run is under
`logs/rsl_rl/shoelace_dual_franka/2026-09-06_10-18-15_curriculum74_scale070_b075_m960freshadam_level36frac100_std001_frozenbackbone_lr2p5e6_thresh40_admm_8gpu_300_stagger5`.
Its first update moved the complete output weight matrix by only `2.67e-4` in L2 norm, with a `1.00e-5` maximum
element change, and its first complete level-36 window measured 25.78% success. This is consistent with the frozen
baseline rather than the immediate optimizer-restoration collapse; later checkpoints still require fixed-policy
gating before acceptance.

The corrected run was stopped after iteration 1101 once approximately 2.33 million environment steps had sampled
level 36. Complete windows rose as high as 35.66% but did not promote and ended at 33.59%; the late oscillation
made the peak window insufficient evidence by itself. A common-seed checkpoint screen measured 35/128 successes
for the input, 37/128 for `model_1040.pt`, 45/128 for `model_1070.pt`, and 40/128 for `model_1100.pt`. Four new
level-36 seeds then confirmed `model_1070.pt`: it improved every paired seed and increased success from 115/512
(22.46%) to 158/512 (30.86%), while reducing lost grasp from 210 to 191 and insufficient separation from 187 to
163. An adjacent level-35 safety gate also improved from 204/512 (39.84%) to 227/512 (44.34%); three seeds
improved, one fell by only 2/128, and every candidate seed remained above 40%. `model_1070.pt` is therefore the
first accepted PPO actor improvement at this contact frontier. It still lacks a level-36 promotion margin, so the
next continuation restarts its Adam state instead of continuing the later optimizer trajectory that lost the
fixed-policy gain.

A second fresh-Adam output-head continuation from the accepted `model_1070.pt` ran under
`logs/rsl_rl/shoelace_dual_franka/2026-09-06_10-50-27_curriculum74_scale070_b075_m1070freshadam2_level36frac100_std001_frozenbackbone_lr2p5e6_thresh40_admm_8gpu_200_stagger5`.
It again used fixed `2.5e-6` learning rate, frozen actor backbone and observation statistics, one PPO epoch, and
100% frontier resets. Level 36 initially fluctuated between approximately 32% and 35% success, then completed a
41.41% window and promoted at iteration 1147. Level 37 opened at 28.12%, recovered through 32--38% windows, and
completed a 42.19% window to promote at iteration 1240. The run ended normally at iteration 1269 on level 38;
its first complete level-38 window measured 31.25%, while the final rolling success was approximately 41.7% with
33.2% lost grasp, 29.1% insufficient separation, and no unsafe or time-out terminations.

A common-seed fixed-policy screen separated actor improvement from curriculum-state promotion. At levels 37 and
38 respectively, the input produced 34/128 and 29/128 successes, `model_1145.pt` produced 43/128 and 39/128,
`model_1240.pt` produced 45/128 and 43/128, and `model_1269.pt` produced 67/128 and 45/128. Four new level-38
seeds then confirmed the final checkpoint: every paired seed improved, from 33, 32, 27, and 38 successes for the
input to 43, 45, 44, and 39 for `model_1269.pt`, or 130/512 (25.39%) to 171/512 (33.40%). The gain converted
insufficient-separation outcomes from 196 to 136, despite lost-grasp increasing from 186 to 205. A level-36
retention gate also improved every seed, from 158/512 (30.86%) to 208/512 (40.63%); lost grasp fell from 191 to
183 and insufficient separation from 163 to 121. `model_1269.pt` is therefore the new accepted actor, with
paired fixed-policy evidence of improvement across levels 36--38 rather than a lucky online promotion window.

A third fresh-Adam output-head continuation from the accepted `model_1269.pt` ran under
`logs/rsl_rl/shoelace_dual_franka/2026-09-06_11-30-39_curriculum74_scale070_b075_m1269freshadam3_level38frac100_std001_frozenbackbone_lr2p5e6_thresh40_admm_8gpu_200_stagger5_retry1`.
The first launch incorrectly passed the unregistered `--save_interval 5` CLI option and failed during Hydra
serialization before PPO started; the correct override is `agent.save_interval=5`. The corrected launch then hit
a native Newton allocator failure on rank 7 before PPO started, while its retry initialized normally. Runtime
checkpoint inspection confirmed the intended fixed `2.5e-6` learning rate. With the actor backbone and observation
statistics frozen, one PPO epoch, and 100% frontier resets, the retry advanced from level 38 to level 45 in 200
updates. Its completed promotion windows for levels 38--44 were 43.41%, 42.75%, 43.41%, 40.62%, 48.46%,
41.41%, and 40.62%. It ended normally at iteration 1468 with approximately 40.4% rolling success, 36.8% lost
grasp, 23.9% insufficient separation, and no unsafe or time-out terminations.

A same-seed screen at levels 44 and 45 selected the final checkpoint over the input and intermediate checkpoints.
At level 45, `model_1269.pt`, `model_1365.pt`, `model_1420.pt`, and `model_1468.pt` produced 48/128, 55/128,
64/128, and 70/128 successes respectively; at level 44, the input and final checkpoint produced 50/128 and
63/128. Four new level-45 seeds then improved every paired result from 48, 45, 42, and 42 successes for the input
to 66, 57, 68, and 58 for `model_1468.pt`, or 177/512 (34.57%) to 249/512 (48.63%). Lost grasp fell from 168
to 144 and insufficient separation from 167 to 119. The level-38 retention gate also improved from 171/512
(33.40%) to 214/512 (41.80%); three seeds improved, one fell by only 3/128, lost grasp fell from 205 to 166,
and insufficient separation fell from 136 to 132. `model_1468.pt` is therefore the new accepted actor and the
next continuation starts at the measured level-45 frontier with another fresh Adam state.

A fourth fresh-Adam output-head continuation from the accepted `model_1468.pt` ran under
`logs/rsl_rl/shoelace_dual_franka/2026-09-06_12-07-38_curriculum74_scale070_b075_m1468freshadam4_level45frac100_std001_frozenbackbone_lr2p5e6_thresh40_admm_8gpu_200_stagger5`.
All eight staggered ranks synchronized and the run completed normally at iteration 1667. Runtime checkpoint
inspection again confirmed the fixed `2.5e-6` learning rate and that only the actor output layer and standard
deviation were trainable. The curriculum advanced from level 45 through level 53 in 200 updates. Completed
promotion windows for levels 45--52 were approximately 40.62%, 41.09%, 40.62%, 48.44%, 40.77%, 42.97%,
51.16%, and 40.62%. Level 50 temporarily fell to 33.59% before recovering without a curriculum change. The
level-53 window ended at 31.78%, with final rolling rates of approximately 26.0% success, 33.9% lost grasp, and
42.4% insufficient separation; unsafe and time-out terminations remained zero.

A same-seed screen at levels 52 and 53 prevented accepting the final online state without fixed-policy evidence.
At level 52, the input and `model_1667.pt` produced 44/128 and 61/128 successes. At level 53, the input produced
37/128 while the screened stage checkpoints ranged from 29/128 to 36/128 and the final checkpoint produced
32/128. Eight new paired seeds showed that this apparent regression was seed noise: `model_1667.pt` improved six
of eight seeds and increased aggregate level-53 success from 248/1024 (24.22%) to 267/1024 (26.07%), while lost
grasp fell from 339 to 328 and insufficient separation fell from 437 to 429. A level-45 retention gate was exactly
neutral at 249/512 successes for both actors. A prospective two-seed level-54 probe remained below the promotion
threshold and favored the input 24/256 to 15/256, so level 54 is not yet claimed as solved. Actor-output
interpolations at 25%, 50%, and 75% of the stage update did not dominate the endpoints across levels 52--54 and
were rejected. One screen process encountered the known native USD allocator failure before loading a policy;
its isolated retry completed normally and only the retry was counted. `model_1667.pt` is accepted as the next
frontier actor, with a modest measured level-53 gain and no level-45 retention loss; the next continuation starts
at level 53 with a fresh Adam state.

The fifth fresh-Adam output-head continuation ran under
`logs/rsl_rl/shoelace_dual_franka/2026-09-06_12-51-58_curriculum74_scale070_b075_m1667freshadam5_level53frac100_std001_frozenbackbone_lr2p5e6_thresh40_admm_8gpu_200_stagger5`.
It was stopped after 41 updates because three completed level-53 windows remained at 25.78%, 23.44%, and 24.03%
success. A fixed four-seed gate rejected the apparent online peak at `model_1700.pt`: it reduced success from
136/512 (26.56%) to 124/512 (24.22%). Extending the separation deadline from 64 through 256 steps did not expose
delayed successes; it primarily converted insufficient-separation failures into later lost grasps. A causal
`+0.01` left-`dx` output-bias screen looked balanced on one seed, but its four-new-seed gate improved only from
129/512 to 132/512 and only two seeds improved. The output bias is therefore rejected and `model_1667.pt` remains
the accepted actor.

A 12-update representation-plasticity probe then changed only the actor class from the frozen-backbone model to
the fully trainable fixed-observation-statistics model. It ran under
`logs/rsl_rl/shoelace_dual_franka/2026-09-06_13-19-27_curriculum74_scale070_b075_m1667_fullactor_level53frac100_std001_lr2p5e6_thresh40_admm_8gpu_12_stagger5`.
All 196,608 transitions completed normally and every fixed screen retained 100% acquisition, but checkpoint
quality oscillated instead of improving. The only same-seed local peak, `model_1672.pt`, reduced four-new-seed
level-53 success from 140/512 (27.34%) to 125/512 (24.41%). Full-actor plasticity alone is therefore rejected.
One concurrent gate process failed during native USD import with `MemoryError`; its isolated retry completed and
only that result was counted.

### Grasp-aligned dense potential

An outcome trace exposed a reward-contract mismatch rather than a missing success signal. At level 53, the
accepted actor's successful, lost-grasp, and insufficient-separation trajectories reached mean maximum geometric
task progress of approximately 0.992, 0.731, and 0.745. Some failures reached 1.0 geometric progress without
satisfying success because the dense potential omitted the required tail-to-TCP grasp margin. More importantly,
the dense task term became inactive as soon as either strict grasp became false while its previous geometric
potential continued updating. Progress lost during the six-step release-confirmation interval was consequently
hidden from the policy. Successful trajectories kept maximum left and right grasp distances near 0.0062 m and
0.0078 m, whereas both failure classes contained much larger tail-to-TCP excursions.

The reset-relative task potential now includes a continuous bilateral retention factor. It remains one while both
tail-to-TCP distances are within the 0.015 m success margin, falls to zero at the 0.020 m release margin, and
multiplies the existing geometric progress. After bilateral acquisition, its finite-difference rate remains active
during incipient slip, so losing an accumulated potential produces immediate negative dense credit without adding
an alive bonus or reference-trajectory reward. Potential recovery on regrasp receives the matching positive
change, preserving the telescoping progress contract.

The focused regression fails under the former masking behavior, and the complete shoelace unit suite passes with
78 tests. A 128-environment Newton rollout held `model_1667.pt` fixed under the corrected reward: behavior remained
31 successes, 43 lost grasps, 54 insufficient-separation failures, and 100% acquisition for seed 430. Mean dense
return remained 4.70 for successes but fell to 0.70 and 0.74 for the two failure classes, compared with roughly
2.0--2.2 before the correction. Because this changes the optimized return, the first subsequent policy training
started from a fresh actor, critic, optimizer, and observation statistics rather than restoring the old-reward
training state. Later actor-only transfer experiments are documented below and deliberately do not restore the
stale critic or optimizer.

### Fresh corrected-reward run and low-noise stabilization

The first fresh run under the grasp-aligned dense potential is stored under
`logs/rsl_rl/shoelace_dual_franka/2026-09-06_13-40-22_curriculum74_retentionpotential_scale070_b075_hybrid2_fresh_8gpu_300_stagger5_retry1`.
It used eight GPUs with 128 environments per rank and first produced success at iteration 81. Online stochastic
success reached 13.4% at iteration 125 before falling to zero near iteration 140, so the run was stopped after
iteration 142. Fixed-policy evaluation showed that this online collapse did not reflect the mean actor: across
five seeds, deterministic `model_140.pt` produced 100/640 successes (15.62%). With action-standard-deviation
parameters fixed to 0.10, four seeds produced 157/512 successes (30.66%), establishing that the actor still
contained useful behavior but that its learned exploration noise had become too large.

A 24-update stabilization from `model_140.pt` therefore fixed the standard-deviation parameters at 0.10, froze
the actor backbone and observation statistics, reset Adam, trained only the actor output layer at a fixed
`2.5e-6` learning rate for one PPO epoch, and retained the ADMM Baumgarte value of 0.75. The successful run is
stored under
`logs/rsl_rl/shoelace_dual_franka/2026-09-06_14-26-16_curriculum74_retentionpotential_scale070_b075_m140_std010_frozenbackbone_freshadam_lr2p5e6_level0_thresh40_admm_8gpu_24_stagger5_retry2`.
It completed normally at iteration 163 with 31--32% online success, approximately 52% lost grasp, 16--17%
insufficient separation, and no unsafe or time-out termination. It remained below the 40% promotion threshold,
so every reset stayed at level 0.

A common-seed checkpoint screen selected `model_160.pt`, which improved deterministic success from 17/128 to
31/128 relative to the input. Four new paired seeds confirmed the gain: the input produced 19, 15, 21, and 23
successes, while `model_160.pt` produced 26, 28, 31, and 23, improving the aggregate from 78/512 (15.23%) to
108/512 (21.09%). The update is therefore retained as a reproducible mean-policy improvement without forgetting,
but level 0 is not yet solved and no curriculum promotion is claimed.

A second 24-update output-head segment restarted Adam from the accepted `model_160.pt` while retaining the 0.10
standard-deviation parameters. It completed under
`logs/rsl_rl/shoelace_dual_franka/2026-09-06_14-44-33_curriculum74_retentionpotential_scale070_b075_m160_std010_frozenbackbone_freshadam2_lr2p5e6_level0_thresh40_admm_8gpu_24_stagger5`.
Its online window ended at 29.23%, and the only common-seed local peak, `model_178.pt`, failed every new paired
seed: deterministic success fell from 115/512 (22.46%) for the input to 69/512 (13.48%). The entire second
segment is rejected and `model_160.pt` remains the accepted actor.

A uniform stochastic-standard-deviation sweep then isolated excess exploration noise. On one seed, parameter
values 0.04, 0.05, 0.06, 0.07, 0.075, 0.08, 0.085, 0.09, 0.10, 0.11, and 0.12 produced respectively 48, 55,
52, 59, 51, 47, 46, 51, 38, 35, and 37 successes out of 128. Four new paired seeds confirmed 0.07 over 0.10:
success improved on every seed from 169/512 (33.01%) to 201/512 (39.26%), while lost grasp fell from 269 to 227
and insufficient separation rose from 74 to 84. With the 0.7 arm-action scale, 0.07 corresponds to approximately
0.049 executed arm-action standard deviation. This noise setting is accepted for subsequent sampling without
changing the deterministic actor or lowering the 40% curriculum threshold.

A 0.07-noise output-head continuation from `model_160.pt` ran under
`logs/rsl_rl/shoelace_dual_franka/2026-09-06_15-08-45_curriculum74_retentionpotential_scale070_b075_m160_std007_frozenbackbone_freshadam3_lr2p5e6_level0_thresh40_admm_8gpu_64_stagger5`.
It was stopped after iteration 197 when its first two complete windows fell from 31.01% to 27.91%. Every screened
checkpoint through `model_188.pt` was below the input on a common deterministic seed. The run is rejected; lower
noise improved the fixed actor's success sampling but did not correct the PPO update direction.

A critic-only diagnostic then froze every actor parameter, retained 0.07 noise, increased the rollout horizon
from 16 to 80, and trained the critic for 12 updates at `3e-4` under
`logs/rsl_rl/shoelace_dual_franka/2026-09-06_15-19-35_curriculum74_retentionpotential_scale070_b075_m160_std007_frozenpolicy_criticwarmup_lr3e4_h80_level0_admm_8gpu_12_stagger5`.
Actor comparison found only the intentional standard-deviation change; all MLP tensors remained bitwise equal.
On a held-out rollout, aggregate critic MAE improved from 12.10 to 11.53 and explained variance from -0.10 to
0.055, but successful-trajectory MAE changed only from 24.07 to 23.91. The critic continued to predict a negative
mean value for trajectories whose realized discounted return was positive. This is a modest calibration gain,
not evidence that critic warm-up alone resolves the high-variance success credit problem.

Finally, a four-update horizon-320 output-head gate ensured that every policy update saw complete success and
failure episodes. Its first startup attempt failed inside Newton USD parsing before PPO and was automatically
retried; the valid retry completed 1,310,720 transitions under
`logs/rsl_rl/shoelace_dual_franka/2026-09-06_15-33-38_curriculum74_retentionpotential_scale070_b075_m171_std007_frozenbackbone_h320_freshadam_lr2p5e6_level0_admm_8gpu_4_stagger5`.
`model_172.pt` improved a common-seed deterministic screen from 20/128 to 31/128, but four new paired seeds
produced exactly 97/512 successes for both input and candidate; only one seed improved and three regressed. The
candidate and the hypothesis that rollout truncation is the sole remaining cause are rejected. The accepted
state remains the `model_160.pt` mean actor with fixed 0.07 sampling noise; further work must improve the
state-conditioned post-acquisition update rather than repeat short output-head PPO, unweighted self-imitation, or
critic-only warm-up.

A corrected-reward fixed-policy sweep also rechecked whether the 64-step separation-progress deadline was hiding
late successes at level 0. Deadlines of 64, 80, 96, 128, 160, 192, 256, and 400 acquired control steps produced
55, 52, 52, 55, 54, 50, 53, and 47 successes out of 128. Longer deadlines mainly converted
insufficient-separation outcomes into lost grasps, so the 64-step default is retained.

A causal screen then added positive and negative 0.01 offsets to each arm-action output bias of `model_160.pt`.
The strongest single-round result was a negative left-`dz` offset with 41/128 successes, compared with 22/128
for the concurrently evaluated input and 19/128 for the opposite offset. A new-seed magnitude sweep from -0.003
through -0.020 found every negative offset above its 17/128 concurrent input and selected -0.015 at 35/128 for
confirmation. Four new paired seeds rejected that candidate: aggregate deterministic success fell from 106/512
for the unmodified input to 96/512, with only one of four seeds improving. The left-`dz` bias and the broader
single-seed arm-bias hypothesis are therefore rejected; neither changes the accepted actor or training config.

### Corrected-reward credit and reset-frontier diagnosis

The stochastic success traces used above initially saved only policy-mean actions. A corrected trace then saved
the actual sampled actions for eight seeds, covering 1,024 level-0 episodes at standard deviation 0.07. It
produced 368 successes, 533 lost grasps, and 123 insufficient-separation failures. Successful-minus-failed action
residual means were approximately 0.001 and changed sign across seeds. A cross-seed ridge model from policy hidden
features to successful residuals obtained approximately -0.1% to -0.2% explained variance. Successful exploration
noise is therefore not a reproducible state-conditioned action target, so sampled-action cloning is rejected.

Credit-horizon checks also rejected a simple rollout-length explanation. Level-0 acquisition occurs at reset,
while successful episodes typically terminate around 260 policy steps. With `gamma=0.99` and `lambda=0.95`, the
terminal contribution to the first-step generalized advantage is only about `1.2e-7`. Setting `lambda=1.0` and
using 320-step rollouts put complete episodes in each update, but a four-update output-head gate changed a common
seed only from 30/128 to 31/128 at its best checkpoint and degraded at later checkpoints. Long-horizon credit is
necessary for this task but was not sufficient to repair the fresh actor.

A fixed-level sweep then exposed a reset-contact cliff rather than missing exploration. For the fresh
`model_160.pt` actor with standard deviation 0.07, level 0 produced 47/128 successes, level 1 produced none, level
2 produced 1/128, and levels 4, 8, 12, 16, and 24 produced none, although every level eventually acquired both
tails. The only change between levels 0 and 1 is the reset finger position, from 0.0020 m to 0.0025 m. A 0.1 mm
sweep gave 41, 17, 5, 0, 0, 0, 4, and 0 successes at positions 2.0 through 2.8 mm. A finer 12.5 micrometre sweep
between 2.0000 and 2.0875 mm remained strongly non-monotonic, with counts from 2/128 to 35/128. Neither denser
reset spacing nor a monotonic aperture curriculum alone can smooth this contact manifold.

At level 1, deterministic evaluation and stochastic standard deviations from 0.01 through 0.07 all produced zero
success. Waiting with zero arm actions for 1 through 64 steps also produced zero success and only converted lost
grasps to insufficient-separation failures. The 0.5 mm observation change is approximately 0.009 stored standard
deviations, and initial normalized observations otherwise match level 0. Exploration magnitude, preload waiting,
and observation normalization are therefore rejected as primary causes.

The accepted old-reward actor provided the causal control. Under the corrected reward and current physics,
`model_1667.pt` produced 15/128 deterministic successes at level 1 and 71/128 at level 2, compared with 0/128 for
the fresh actor at both levels on that screen. A second sweep produced 30, 21, 78, 96, 104, 110, and 109 successes
at levels 0, 1, 2, 4, 5, 6, and 7. Thus the current reset states are physically solvable; the fresh run had learned
a narrow, level-0-specific two-arm controller. Reward changes invalidate the old value function and Adam moments,
but they do not erase measured actor behavior. This evidence motivated an actor-only transfer: copy the old actor
exactly, use a corrected-reward critic, clear optimizer state, and validate every subsequent actor change under the
current reward.

### Actor-only transfer and corrected-reward frontier

The transfer checkpoint combines the actor from old-reward `model_1667.pt` with the corrected-reward critic from
the horizon-320 diagnostic, fixes the action standard deviation at 0.01, and clears Adam. Tensor comparison found
zero actor difference from the old checkpoint and zero critic difference from the corrected checkpoint. A
12-update frozen-policy probe is stored under
`logs/rsl_rl/shoelace_dual_franka/2026-09-06_17-25-36_curriculum74_retentionpotential_b075_oldactor1667_correctedcritic171_std001_frozenpolicy_h80_lam100_criticwarmup_level2_thresh40_admm_8gpu_12`.
All eight ranks promoted from level 2 to level 3 on a 48.44% window. Actor tensors remained bitwise unchanged in
every saved checkpoint, proving that the corrected reward and actor-only transfer form a viable curriculum start.

A 24-update output-head continuation then ran under
`logs/rsl_rl/shoelace_dual_franka/2026-09-06_17-34-58_curriculum74_retentionpotential_b075_oldactor1667_correctedcritic11_std001_outputhead_h80_lam100_freshadam_lr2p5e6_level3_thresh40_admm_8gpu_24`.
It used 80-step rollouts, `lambda=1.0`, fixed observation statistics and standard deviation, one PPO epoch, and a
fresh fixed `2.5e-6` Adam optimizer. Every rank promoted level 3 to level 4. The first level-4 window measured
67.42% and raised exposure from 20% toward 35%; the next completed window measured 76.56%. The run ended normally
at iteration 34 with 35% level-4 exposure.

Fixed-policy replay separated curriculum progression from actor learning. At level 4, the input and final
`model_34.pt` each produced 111/128 deterministic successes; intermediate checkpoints produced 104--109. On a
new common seed, input versus final success counts were 37 versus 39 at level 0, 24 versus 23 at level 1, 79
versus 87 at level 2, and 87 versus 87 at level 3. At levels 5 through 8, input totals were 113, 109, 105, and
116, while final totals were 110, 108, 113, and 113. The output-head update therefore caused no material retention
loss and modestly improved level 2, but it did not explain the level-4 promotion.

A wider fixed-level scan located the corrected-reward actor frontier. `model_34.pt` produced 115/128 successes at
levels 9 and 12, 113 at level 16, 111 at level 24, 72 at level 32, 63 at level 40, and 73 at level 48. A finer
new-seed scan produced 59, 59, 68, and 65 successes at levels 49--52, followed by 30 at level 53, 13 at level 54,
and 8 at level 56. Level 52 is therefore the last measured reset above the 40% experimental promotion threshold;
level 53 is the first clear learning frontier.

At level 53, action noise from 0.01 through 0.05 reduced success relative to deterministic evaluation. A lower
noise screen found one 43/128 result at standard deviation 0.005, but a four-seed confirmation showed that 0.003
and 0.005 were statistically tied at 126/512 and 128/512 successes. Standard deviation 0.005 is retained because
it reduced lost grasps from 171 to 154, shifting failures toward the grasp-retaining insufficient-separation class
that preserves the dense pull signal.

The level-4 trace also exposed insufficient critic calibration. From the start to `model_34.pt`, held-out critic
explained variance improved only from 0.144 to 0.177 and mean absolute error from 25.22 to 24.32. The final critic
predicted a mean value of 0.33 for trajectories whose realized discounted return averaged 23.48. The output-head
run's shared `2.5e-6` optimizer rate was appropriate for the actor but too small to fit the critic. A staged
frontier run therefore starts with the actor fully frozen, standard deviation 0.005, a fresh optimizer, and critic
learning rate `3e-4` at level 52. Only after held-out value fit improves will the actor output layer be unfrozen at
the level-53 frontier.

### Level-53 critic calibration and weakest-margin diagnosis

The frozen-policy stage promoted from level 52 to level 53 and was extended with 24 updates sampled entirely from
level 53. Its accepted `model_80.pt` retained the actor tensors exactly while improving held-out critic explained
variance from 0.003 for `model_34.pt` to 0.306. Overall mean absolute error fell from 10.78 to 10.00, successful-
trajectory error fell from 35.23 to 25.36, and the mean successful-state prediction rose from -4.56 to 5.31 for a
realized mean return of 30.67. The critic remained conservatively biased, but it was materially better calibrated
than the critic produced by the shared low actor learning rate.

A 24-update level-53 output-head continuation then used the calibrated critic, fixed standard deviation 0.005,
80-step rollouts, `lambda=1.0`, and fresh Adam at `2.5e-6`. Its fixed-seed checkpoint screen showed no monotonic
improvement. Four new paired seeds changed aggregate success only from 133/512 for the input actor to 136/512 for
the best candidate, with per-seed changes of -5, +1, +1, and +6. This is too small relative to the measured Newton
contact variance, so the actor update was rejected; the accepted state remains the `model_34.pt` actor paired with
the level-53-calibrated `model_80.pt` critic.

Outcome traces across four further seeds contained 129 successes, 170 lost grasps, and 213 insufficient-separation
failures. At 32 steps after acquisition, normalized success-trajectory components averaged approximately 0.17 for
throat clearance, 0.69 for left-tail distance, 0.03 for right-tail distance, and 0.40 for tail separation; failure
trajectories were nearly identical at that horizon. The right tail is therefore the consistent incomplete margin,
not a missing success event or a general failure to separate the tails.

A fixed-trajectory counterfactual on those 512 episodes compared weakest-margin weights from 0 through 1. The
former 0.25 blend allocated only about 40--42% of the local task-potential gradient to the right-tail component.
A 0.75 blend allocates about 71--75% to it while preserving roughly 6% for each other component. It also increased
the approximate discounted task-potential gap between success and failure from about 0.14 to 0.16. A pure 1.0
soft-min was rejected because it left effectively zero gradient for the already-higher tail-distance and separation
components. The default is therefore changed to 0.75 and must start with a frozen-actor critic recalibration because
the reward return contract changed.

### Soft-min recalibration, gradient-noise diagnosis, and level-53 densification

The 0.75-soft-min critic recalibration ran for 24 frozen-actor updates under
`logs/rsl_rl/shoelace_dual_franka/2026-09-06_19-34-22_curriculum74_retentionpotential_softmin075_b075_m80_std0005_frozenpolicy_h80_lam100_criticlr3e4_freshadam_level53frac100_locked_admm_8gpu_24`.
The accepted `model_23.pt` retained the actor exactly. Across four held-out seeds it improved explained variance by
0.069 and reduced overall, successful-trajectory, and failed-trajectory value MAE by 0.76, 0.81, and 0.74
respectively relative to the input critic.

Neither of the subsequent 80-step actor gates produced a reproducible improvement. The output-head gate's best
screened checkpoint reduced four-seed success from 123/512 to 113/512. A split actor that kept the complete gripper
branch fixed while training the arm backbone reduced success from 122/512 to 116/512 at its first checkpoint and to
115/512 at its best later screen. Both actor changes were rejected. Successful, lost-grasp, and insufficient-
separation episodes averaged approximately 166, 163, and 165 steps, so an 80-step rollout does not contain their
terminal outcome from a synchronized reset.

A four-update, 240-step split-arm gate then put complete episodes into each PPO batch under
`logs/rsl_rl/shoelace_dual_franka/2026-09-06_20-56-37_curriculum74_retentionpotential_softmin075_b075_m23_std0005_frozengripper_split_h240_lam100_freshadam_lr2p5e6_level53frac100_locked_admm_8gpu_4_stagger10_retry2`.
Its `model_2.pt` improved a selection seed from 29/128 to 39/128 successes, but eight prospective paired seeds
changed aggregate success only from 248/1024 to 252/1024 and increased lost grasps from 345 to 367. The checkpoint
and the hypothesis that rollout truncation was the sole remaining cause were rejected.

A diagnostic PPO implementation measured the per-environment score-function gradient before updating the actor.
With 1,024 environments, a 240-step horizon, and arm standard deviation 0.005, every arm-output gradient had an
absolute signal-to-standard-error ratio below 1 and only 49--52% of environments agreed on its sign. Increasing
the standard deviation to 0.02 or 0.05 reduced the standard error but left at most two dimensions above a ratio of
1, while the apparent directions changed between probes. Outcome variation at this frontier is therefore not
reliably correlated with Gaussian action perturbations; longer rollouts and larger uniform exploration alone make
PPO follow contact noise rather than a repeatable arm-control improvement.

The remaining discontinuity was localized to the reset curriculum. The former levels 52 and 53 used identical
settled arm poses and changed only the driven finger position from 4.046875 to 4.125000 mm, while fixed-policy
success fell from roughly 51% to 23%. A seed-561 reset-only sweep across this 78.125 micrometre interval produced
59, 61, 60, 51, 43, 52, 40, and 29 successes out of 128. Six measured intermediate finger positions are therefore
inserted, reducing the largest step to approximately 11.16 micrometres and increasing the curriculum from 74 to
80 levels. The five later Cartesian-approach states retain their original poses and order. Because numeric indices
after the insertion shift by six, checkpoint continuations must select their starting level from the authored
finger position rather than reuse an old numeric level blindly.

### Terminal-credit rejection and ADMM proximal calibration

Eight stochastic level-59 traces covered 1,024 complete episodes from the accepted actor and critic: 275 ended in
success, 347 in lost grasp, and 402 in insufficient separation. A time-dependent leave-one-out baseline was used
to recompute the score-function gradient while varying the success/loss impulses from the configured `60/-12`
through zero. Removing both impulses did not improve the gradient signal-to-noise ratio or cross-seed agreement;
the dense-only direction was likewise inconsistent. Terminal reward scale is therefore not the cause of the
failed actor updates, and the configured success and failure weights are retained.

The same traces also rejected two observation and credit-horizon hypotheses. Adding the critic-only throat-density
feature to the offline outcome classifier increased its AUC from approximately 0.85 to 0.95, but score gradients
conditioned on throat-progress bins still changed direction across seeds and had no repeatable actionable arm
dimension. The feature remains privileged instead of being exposed to the actor. The accepted critic reached
approximately 0.38 explained variance overall and 0.70 late in an episode, but almost zero during the first half,
when the eventual contact outcome was not yet predictable. Counterfactual GAE and real four-rank, 256-environment
probes found no stable improvement from `lambda=0.95` over `1.0`; neither a critic-only extension nor shorter credit
horizon repairs the action-gradient noise.

Contact parameters were then screened with the policy fixed. Raising the Newton soft-contact friction from 0.5 to
16 increased success only from 104/512 to 118/512 across four seeds, regressed one seed, and increased lost grasps,
so the extreme friction value was rejected. ADMM proximal regularization had a much larger but non-monotonic
effect. `gamma=1e-4` improved level 59 from 117/512 to 272/512 successes, yet degraded level 52 from 262/512 to
136/512, moving rather than removing the reset-contact cliff. Values from zero through `7.5e-5` were consequently
screened at both levels before changing the task.

`gamma=7.5e-5` passed the two-sided four-seed gate: level-52 success was 75, 77, 80, and 73 out of 128, totaling
305/512 (59.57%), while level-59 success was 65, 59, 62, and 68, totaling 254/512 (49.61%). Both exceed the 40%
experimental promotion threshold and improve the corresponding zero-gamma baselines of 51.17% and 22.85%.
A fixed-seed scan then located the next frontier: levels 60--67 produced respectively 32, 40, 28, 29, 24, 0, 0,
and 28 successes out of 128. Uniformly interpolating the gripper aperture between levels 59 and 60 had already
shown declining success, so continuous reset jitter would dilute rather than bridge this signal. The shoelace ADMM
default now uses `gamma=7.5e-5`; level 59 is the last demonstrated promotion-capable reset and level 60 is the next
policy-learning target. The complete 78-test shoelace unit suite passes under the new default.

Because the solver change alters the transition distribution, the old value function was not passed directly to
an actor update. The accepted actor was frozen for eight 240-step critic updates at level 59, covering 1,966,080
transitions with all eight ranks. The first launch failed in native Newton USD parsing on rank 1 before PPO; an
unchanged staggered retry completed normally. Four held-out seeds compared the final critic with its input on the
same trajectories: mean explained variance improved from 0.335 to 0.353, overall MAE from 16.42 to 16.10, and
successful-trajectory MAE from 25.73 to 23.97. Every actor tensor remained bitwise unchanged. The final critic was
therefore paired with that actor in a new checkpoint with empty Adam state, fixed `0.005` standard deviation, and
`2.5e-6` output-head learning rate for the next guarded policy-learning stage.

The guarded stage started at level 59 with 100% frontier exposure and a 40% promotion threshold. Its first command
was rejected before PPO because the overridden promotion rate was below the unchanged fraction-increase rate; the
corrected command set both to 0.4 and started cleanly. Every rank promoted to level 60 by iteration 8, after a
47.66% complete level-59 window. The first three pure level-60 updates then declined from 26.71% through 23.32% to
21.63% success, so the nominal 300-update run was stopped at `model_10.pt` instead of being allowed to drift.
Four prospective paired seeds confirmed no actor gain: level-60 success changed only from 134/512 to 136/512,
while the level-59 safety gate regressed from 257/512 to 246/512. The candidate is rejected. Proximal calibration
therefore improves the physical curriculum boundary and supplies a clean level-60 target, but does not by itself
make the existing output-head PPO update learn that target.

### Arm-command EMA and action-history consistency

Level 60 failures were next tested against the hypothesis that small frame-to-frame arm-command reversals disturb
the marginal finger/cable contact. A same-seed sweep of the previous-command EMA weight selected `beta=0.25`
(`alpha=0.75` newest-command weight): success rose from 34/128 without filtering to 61/128. Stronger smoothing was
not monotonic and eventually suppressed all success, so only the mild filter was retained for validation.

Four prospective level-60 seeds compared the unfiltered actor with the mild EMA. Success increased from 150/512
(29.30%) to 187/512 (36.52%), lost grasp decreased from 215 to 201, and insufficient separation decreased from
147 to 124. The level-59 safety gate remained above the 40% experimental threshold at 230/512 (44.92%) versus
251/512 (49.02%) unfiltered. A second level-52 safety gate then improved from 315/512 (61.52%) unfiltered to
327/512 (63.87%) with EMA, while lost grasp decreased from 120 to 117 and insufficient separation from 77 to 68.

The action-history observation was a decisive part of the intervention. An initial gate suggested that feeding the
actor raw commands while executing filtered commands collapsed success, but it overlapped the level-52 GPU gate.
An exclusive eight-GPU rerun therefore paired four new seeds at level 60. Observing executed EMA commands produced
55, 43, 58, and 43 successes, totaling 199/512 (38.87%). Observing raw commands produced zero successes, two lost
grasps, and 510 insufficient-separation failures. The result establishes a state-semantics bug rather than a
scheduling artifact: the actor's feedback loop depends on seeing the action that actually reached the controller.

The task now implements the filter as a local differential-IK action term instead of modifying the environment
step. It preserves the raw action for logging and action-rate rewards, publishes the filtered arm command for the
14-dimensional action-history observation, maintains independent per-environment history, and makes the first
post-reset command exact. The policy and action dimensions are unchanged, but the transition and observation
contracts have changed; the accepted actor may initialize the next stage only after critic recalibration under the
new contract, with a fresh optimizer rather than a direct training resume.

An eight-seed production gate then removed the temporary external filter and exercised the new action term directly
with 128 environments per GPU. The seeds produced 50, 45, 46, 50, 41, 51, 38, and 52 successes, totaling 373/1024
(36.43%), with 386 lost grasps and 265 insufficient-separation failures. One rank failed during native Newton scene
initialization with an allocator error before policy rollout; an unchanged isolated retry supplied the eighth
result. The production success rate is consistent with the preceding EMA gates and validates dynamic action-term
loading, per-environment reset behavior, and executed-action observation wiring. It remains below the 40% promotion
gate, so the next actor stage first requires critic recalibration under this changed transition contract.

The recalibration gate froze the actor for eight 240-step updates at level 60, collecting 1,966,080 transitions
across eight ranks. Online success windows stabilized near 34--36%, unsafe remained zero, and the actor state was
bitwise unchanged. Models after one, four, six, and eight updates were compared with the input critic on paired
held-out trajectories. All four moved the same tradeoff: explained variance and successful-trajectory MAE improved,
but overall and both failure-class MAEs worsened. For example, the six-update candidate changed mean explained
variance from 0.3581 to 0.3647 and success MAE from 24.16 to 23.15, while overall MAE increased from 13.35 to 13.42,
lost-grasp MAE from 6.88 to 7.61, and insufficient-separation MAE from 6.64 to 7.38. One held-out rank exited during
native scene initialization and completed unchanged in isolation. Because no checkpoint improved the complete
value-fit contract, the critic-only continuation is rejected and the input critic is retained with fresh optimizer
state for the guarded EMA actor update.

The guarded output-head update then ran four complete 240-step batches at fixed level 60 with learning rate
`2.5e-6`, fixed `0.005` standard deviation, and fresh Adam. Its online success window moved from 29.55% through
31.33% to 40.48%, but prospective evaluation showed that the apparent final crossing was batch noise: model 10
reduced success from 199/512 to 190/512 and increased lost grasps from 201 to 220. The earliest checkpoint was also
tested to exclude excessive update count as the cause. After one update, model 7 changed success from 189/512 to
188/512 and lost grasps from 200 to 208. Both candidates and the complete output-head branch are rejected. Mild
command smoothing improves the physical transition distribution, but it still does not create a repeatable PPO
score gradient at the remaining aperture discontinuity.

The remaining 4.125-to-4.250 mm reset-aperture interval was therefore rescanned with the production EMA action
term. Eight evenly spaced positions produced 66, 52, 47, 41, 61, 50, 49, and 45 successes out of 128. Unlike the
unfiltered scan, every point retained a dense mixture of successes and both failure classes instead of an abrupt
drop. Two additional seeds targeted four weak or structurally important interior points. Across three seeds,
4.161, 4.179, 4.214, and 4.232 mm produced respectively 153/384 (39.84%), 132/384 (34.38%), 157/384 (40.89%), and
149/384 (38.80%) successes. These are viable learning distributions even where they fall just below the promotion
threshold.

Six measured interior apertures are consequently inserted, reducing the maximum reset step from 125 to about
17.86 micrometres and increasing the curriculum from 80 to 86 levels. Old levels 0--59 retain their indices, the
inserted apertures occupy new levels 60--65, old level 60 becomes new level 66, and the five Cartesian-approach
levels shift from 75--79 to 81--85. The associated arm reset poses remain identical throughout the insertion.
Checkpoint continuations must therefore map levels 60 and above by authored gripper position, not by their former
numeric index.

The remapped source configuration passed a four-seed gate without position overrides: new level 60 produced
209/512 successes (40.82%) and new level 61 produced 223/512 (43.55%). A 300-update curriculum run was then
started from the unchanged accepted actor with 75% maximum frontier exposure, 25% replay, and a 40% promotion
threshold. Its first launch hit a native Newton segmentation fault on rank 7 before PPO; the unchanged retry ran
normally. All ranks advanced from level 60 to 61 and then to 62 within eight updates, while unsafe remained zero.
At level 62, however, four consecutive 50%-exposure windows were 37.73%, 32.56%, 39.48%, and 37.79%, so the run
was stopped rather than allowed to drift for its nominal duration.

Prospective evaluation compared model 14, captured as all ranks entered level 62, with model 18 after the four
frontier updates. Model 18 increased success from 186/512 to 195/512, but also increased lost grasp from 164 to
172. A second four-seed gate against the original accepted actor rejected the apparent gain: model 18 produced
178/512 successes versus 189/512 for the input and increased insufficient-separation failures from 135 to 155.
The curriculum densification is retained because it demonstrated two real promotions, while the trained actor and
its optimizer are rejected. New level 62 remains the first learning frontier.

### Level-62 gradient and controllability diagnosis

A pre-update score-function probe then measured the policy signal at pure new level 62 with the accepted actor,
1,024 environments across eight GPUs, and a complete 240-step rollout. Nine of the twelve continuous arm
dimensions had absolute gradient signal-to-noise ratio below one; the maximum was 1.65 and per-environment gradient
signs stayed between 47.95% and 52.05% positive. A second probe increased the batch eightfold to 8,192 environments
and 1,966,080 transitions. Only one arm dimension reached absolute SNR above two (`2.44`), eleven remained below
two, and only half of the gradient signs agreed with the smaller probe. The larger scene reached 89% occupancy in
Newton's ADMM-internal contact-reduction table but emitted no overflow or failure warning. It is useful as a
diagnostic batch, but production runs retain 128 environments per rank to preserve collision-capacity headroom.

Two counterfactual gates checked whether the weak score estimate still pointed toward an actionable policy change.
First, splitting the 4.161-to-4.179 mm reset interval at 4.170 mm reduced paired success from 187/512 at level 62 to
170/512, so another numeric midpoint is not added to the non-monotonic contact curriculum. Second, adding one
executed-action standard deviation (`0.0035`) with either sign to each of the four largest apparent gradient
dimensions changed paired successes by only zero to three episodes out of 128. Three of the four measured
directions opposed the large-batch gradient sign. The policy is therefore locally outcome-insensitive at this
reset, rather than merely trained with too few samples or too small a learning rate.

Finally, the earlier world-origin sensitivity was rechecked under the current EMA and ADMM settings. Co-locating
all environments at zero spacing reduced success from 206/512 at the default 0.25 m spacing to 124/512 and
increased both failure classes. Zero spacing is rejected, and the authored scene spacing is retained. The default
spacing result also places the accepted actor's level-62 success distribution directly around the experimental
40% promotion boundary, motivating a frozen-actor frontier search before any further actor update.

The first frozen-actor search temporarily lowered the diagnostic promotion threshold to 35%, used pure-frontier
resets, and collected 1,638,400 transitions over twenty 80-step updates. The unchanged actor advanced every rank
from level 62 through levels 63 and 64; by the final update, the distributed mean frontier was 65.7125 as ranks
began entering level 66. Completed distributed windows near the transitions were approximately 38.9--41.4%.
Every one of the actor's eleven state tensors remained bit-identical to the input, while six critic tensors
changed by at most `2.25e-4`. Thus levels 62--64 were false learning frontiers created by placing a hard 40%
threshold on a high-variance numerical-contact distribution. The lower threshold remains an experimental search
setting rather than a task default until the search locates the first state that genuinely requires a policy
change.

A second twenty-update frozen search restarted every rank at level 65. It promoted all ranks through levels 65
and 66, then remained at level 67 for the final nine updates. Once level-67 episodes populated the windows,
distributed success fell from 29.5% through 27.3%, 23.9%, 21.9%, and 19.4% to a final 19.8%; lost grasp remained
approximately 48--55%. The actor was again bit-identical to the accepted input, while six critic tensors changed
by at most `2.48e-4`. New level 67 (`0.00428125 m` driven finger position) is therefore the first demonstrated
frontier below the 35% search gate. The preceding levels are retained as measured physical transitions, but should
not consume actor updates.

### Level-67 acquisition-control diagnosis

The aperture interval from level 66 (`0.00425 m`) to level 67 (`0.00428125 m`) was first sampled at eight evenly
spaced positions. Success counts were 50, 46, 39, 46, 30, 33, 35, and 25 out of 128. The gradual but non-monotonic
decline did not expose another reset midpoint that stayed reliably above the diagnostic 35% gate. A distributed
level-67 gradient probe was attempted three times, but different ranks exited during native Newton/USD startup
before collecting a rollout. Eight independent single-GPU traces therefore supplied the same 1,024-episode batch
without a distributed initialization barrier. They produced 223 successes, 476 lost grasps, and 325
insufficient-separation failures. Only right-arm `dy` reached absolute score-gradient SNR above two, and its signs
agreed on only six of eight seeds. A paired one-standard-deviation intervention then contradicted that gradient:
negative `dy` produced 138/512 successes while positive `dy` produced 117/512. No actor update was accepted.

The decisive check removed exploration entirely. The deterministic mean policy produced 29/128 successes, almost
identical to the stochastic 21.78% aggregate, proving that the contact outcome distribution was not primarily
controlled by the `0.005` policy standard deviation. Time-resolved residual analysis then isolated an early
right-arm local-`x` signal: successful trajectories had a positive residual during steps 0--15 on seven or eight
seeds, while later action differences were consequences of the already-diverged cable state. Applying `+0.0035`
only during those steps improved four paired deterministic seeds from 95/512 to 130/512 successes. A same-seed
dose scan showed that the effect was actionable rather than correlational: the first-16-step success count moved
from 17/128 at `-0.007`, through 28/128 with no bias, to 41/128 at `+0.007` and 45/128 at `+0.056`. Increasing the
state-independent policy standard deviation to either `0.02` or `0.08` did not reproduce the gain.

Duration scans localized the useful intervention to bilateral acquisition. A `+0.056` local-`x` correction for
only the first four or eight steps produced 49/128 and 47/128 successes; extending it through 16, 24, 32, 48, or
all 240 steps produced 32, 27, 37, 29, and 16 successes because the same inward correction later opposed untying.
A simpler controller-neutral intervention, holding both Cartesian arm commands at zero while still executing the
gripper actions, peaked at the same boundary: three and four warm-up steps produced 41/128 and 39/128 successes,
whereas twelve steps produced no successes. This matches the measured bilateral-acquisition time of step three.

The task-local EMA arm action consequently gained a configurable three-control-step post-reset warm-up. It retains
the raw policy input for accounting and observation contracts, executes zero Cartesian commands for the warm-up,
and then enters the existing EMA from zero. A cross-level deterministic gate found only small retention changes at
levels 60 and 66 (47 to 40 and 54 to 50 successes), while level 67 improved from 25 to 55 and level 68 from 32 to
43. The source implementation then produced 384/1,024 level-67 successes across eight new seeds, or 37.50%, versus
223/1,024 before the warm-up; lost grasps fell from 476 to 357. This clears the 35% diagnostic promotion gate
without changing the success definition, grasp thresholds, reward scale, actor weights, or exploration noise.

The scripted open-loop demo exposed a separate diagnostic-only reset bug during this work. It disabled the old
success, unsafe, and lost-grasp terms but left later missed-grasp, insufficient-separation, and time-out terms
active. Its time-out reset invoked the curriculum after the success term had been removed and raised `KeyError`.
The finite-step demo now disables every automatic termination consistently. With that fixed, the constant
reference pull did not solve level 67: it retained the left tail but lost the right, ending with 71 throat segments
and 161.8 mm tail separation. Slower pulls, pure lateral directions, increased upward components, stronger closure,
and longer closure all performed worse. These controls independently support acquisition timing, rather than pull
speed or success thresholds, as the repaired bottleneck.

### Post-warm-up contact-mode calibration

A frozen-policy curriculum run with the three-step arm warm-up promoted level 67 after a 38.28% completed window,
but level-68 windows then ranged from 21.88% to 33.36%. Timing sweeps showed that extending the warm-up from three
to six steps improved levels 69 and 70 but did not produce a single success at levels 71 or 72. A level-71 aperture
scan was strongly non-monotonic: nearby driven finger positions produced 65, 67, 0, 0, 64, 24, 0, and 0 successes
out of 128. The remaining frontier was therefore a discrete contact-mode transition rather than insufficient
curriculum density or a single closure-time setting.

Tail positions in the right-hand frame exposed a false acquisition. At the common reset pose the right tail began
near `[-4.08, -2.51, 3.92] mm`; after closure it settled near `[0.26, -0.33, 0.73] mm` at level 70 but was ejected
to `[3.86, -0.17, 5.81] mm` at level 71. Both endpoints remained inside the 12 mm spherical acquisition threshold,
so the state machine credited proximity even though only the first geometry retained useful finger contact.
Right-arm action biases, direct hand-coordinate corrections, lower actuator velocity limits, and explicit slower
gripper ramps either left success at zero or missed acquisition. These interventions were rejected.

Jacobian-calibrated reset poses instead placed the tail into stable contact before the policy acted. The first pose
covered levels 68--77: it produced 331/512 level-71 successes across four prospective three-step-warm-up seeds and
53.9--64.8% success at every level from 72 through 77. It changed level-67 success only from 46/128 to 42/128, so
the original settled pose remains in use through that boundary. A second, narrow pose produced 296/512 level-78
successes across four seeds but zero at either adjacent contact mode. A third pose produced 487/512 level-79
successes across four seeds and 150/256 at level 80, while failing at levels 77 and 78. The production reset table
therefore uses the original pose through level 67, the first calibrated pose for levels 68--77, the second for
level 78, and the third for levels 79--80. This changes reset geometry only; gripper positions, level indices,
policy weights, rewards, and success thresholds remain unchanged.

A source-only regression gate then removed every temporary reset override and replayed the same deterministic
checkpoint directly from the production table. Levels 67, 68, 77, 78, 79, and 80 produced respectively 55, 55,
86, 67, 118, and 70 successes out of 128. Every environment acquired both tails and every episode reached an
exclusive terminal outcome. All six levels therefore clear the 35% diagnostic promotion threshold under the
authored configuration.

The next unresolved boundary is the Cartesian-approach phase beginning at level 81. A fixed-policy 128-environment
probe at the existing level-81 pose produced no successes, 52 lost grasps, and 76 insufficient-separation failures.
Fading the third contact pose into levels 81--84 also produced no successes and suppressed acquisition; levels 82
and 84 had no acquisition under either pose family. One level-83 current-pose process exited during native Newton
cleanup with `free(): invalid next size`; its adjusted-pose counterpart completed with zero acquisition. The
approach poses remain unchanged until a source-level gate demonstrates a physical bridge from level 80.

Separating the two arm trajectories showed that the left arm could move through 25% of the former first approach
interval while retaining 65/128 successes. The right arm was the discontinuous side: it retained 68/128 successes
at 3.125% of the interval, but 6.25% produced 127 lost grasps. Eight subdivisions placed the last three-step-
warm-up point above the diagnostic threshold at 3.90625% with 52/128 successes; the next 0.1 mm TCP displacement
dropped to 9/128. This cliff occurs while arm commands are still suppressed, so it cannot be repaired by policy
learning under a fixed three-step warm-up.

Disabling warm-up only for approach resets changed that causal result. At 4.296875% of the former right-arm
interval, zero warm-up produced 110/128 successes versus 20, 20, and 14 with one, two, and three warm-up steps.
At 6.25%, zero warm-up produced 61/128 successes while either one or two held steps produced none. The original
first approach endpoint remained unsolved with zero warm-up, so removing the hold is necessary but does not skip
the reach curriculum.

With immediate arm control, a 6.640625% reset produced 30/128 successes on a prospective seed. Applying a right-x
action bias only over the first four control steps increased success monotonically to 52, 82, and 117 out of 128
at biases of -0.05, -0.10, and -0.20, while lost grasp fell from 95 to 69, 41, and 7. A +0.05 right-z bias also
produced 52/128 successes. This is the first post-contact frontier with demonstrated action controllability, in
contrast to the earlier reset-only contact cliffs.

The curriculum consequently adds three right-only bridge poses at 4.296875%, 6.25%, and 6.640625% of the former
level-80-to-81 right-arm interval. The left arm remains at its contact pose for those levels. Arm-command warm-up
is selected per environment: levels 0--80 retain three held commands and the new bridge plus all later approach
levels use zero. The curriculum now contains 89 levels; old levels 81--85 shift to 84--88. The frozen actor should
advance through new levels 81 and 82, while new level 83 is the intended policy-learning frontier.

A production-runtime gate then exercised the authored schedule without any reset or warm-up overrides. Levels 80,
81, 82, and 83 produced 71, 109, 56, and 25 successes out of 128 respectively; all environments acquired both
tails and every episode completed. This simultaneously validates the 3-to-0 per-level warm-up transition and
places the frozen actor above the 35% diagnostic gate through level 82 but below it at the controllable level-83
frontier.

An eight-rank frozen-policy curriculum run then started at level 80 and collected 1,638,400 transitions over
twenty 80-step updates. It promoted level 80 after a 53.91% completed window, level 81 after 84.38%, and level 82
after a 39.06% transition window. Once every environment reached level 83, completed-window success stabilized
between 9.38% and 22.14% and ended at 21.08%; the last instantaneous episode mix was 19.82% success, 77.86% lost
grasp, and 3.10% insufficient separation, with zero unsafe terminations. All eleven actor tensors remained bitwise
identical to the input. Six of ten critic tensors changed, by at most `2.49e-4`. This distributed gate confirms
new level 83 as the first policy-learning target after the reset and timing repairs.

### Level-83 causal action correction and distillation

Output-head updates were first restricted to the right-arm local-`x` action identified by the preceding residual
analysis. Three updates formed the best retention trade-off: at stochastic standard deviation `0.10`, it produced
339/1,024 (33.11%) successes over eight held-out level-83 seeds, while remaining above the diagnostic gate at
levels 80--82. An independent 1,024-episode residual trace then exposed a stronger, seed-consistent signal in the
right-arm local-`z` action. Successful trajectories had higher right-`z` residuals during both the first eight and
first sixteen steps on all eight seeds. Applying `+0.05` only during the first four steps increased a four-seed
prospective gate from 173/512 (33.79%) to 218/512 (42.58%), reducing lost grasps from 324 to 274 without changing
acquisition, unsafe, or unfinished counts.

A direct PPO update of only the right-`z` output was rejected. Its score gradient had the opposite sign to the
causal intervention, changed the output bias by `-1e-4`, and reduced held-out success from 186/512 to 181/512.
The approximately 180-step delay between the four acquisition actions and terminal success made the whole-rollout
policy gradient too noisy to assign this local credit reliably. A local behavior-distillation experiment therefore
used the unchanged policy plus the causal action correction as a temporary teacher. The observation normalizer and
MLP backbone were frozen, only output row 9 was fitted, and steps after the intervention were zero-delta anchors.
No teacher bias, auxiliary reward, or controller was present at inference. The first `+0.05` candidate reproduced
the teacher result at level 83, improving 184/512 to 217/512 successes in a policy-only gate, but its margin over
the production 50% threshold remained insufficient.

The teacher dose was subsequently calibrated before another fit. Over two-seed screens, first-four-step corrections
of `+0.05`, `+0.075`, `+0.10`, and `+0.15` produced respectively 120, 130, 145, and 179 successes out of 256.
An extension produced 175/256 at `+0.15`, 203/256 at `+0.25`, and 218/256 at `+0.30`; one `+0.20` rank failed during
native scene initialization and was excluded rather than counted as rollout data. Four prospective paired seeds
then measured 351/512 (68.55%) at `+0.15` and 454/512 (88.67%) at `+0.30`. The increase consistently converted
lost grasps into successes without increasing insufficient-separation or unsafe terminations.

A duration scan showed that the useful correction was even more local than the original four-step hypothesis.
At `+0.30`, durations of one, two, three, and four steps produced 200, 213, 223, and 226 successes out of 256.
The single first-step correction was therefore selected: it already had a 28-point margin over the production
threshold and could be separated from later observations using the previous-action input. Direct cross-level data
also showed that `+0.15` for four steps did not trade away retained behavior: levels 80--83 changed from
58/128, 89/128, 80/128, and 42/128 to 58/128, 94/128, 100/128, and 86/128 respectively.

The final fit used paired stochastic trajectories from levels 80--83. It targeted an executed right-`z` delta of
`+0.30` at step zero for every level and zero from steps 1--64, with four late anchors per positive row, positive
weight six, and ridge coefficient `0.1`. On held-out environment slots, the first-step delta was `+0.299993` with
`4.5e-5` mean absolute error. The later mean delta was `2.5e-5`, with a 95th-percentile absolute value of `0.002231`.
Only actor output row 9 changed. The resulting checkpoint is
`stabilization_inputs/curriculum89_level83_u3_std010_rightz_distilled030_step0_all80to83_w6_l4_r0.1.pt`.

Policy-only gates contained no runtime action bias. A first cross-level seed changed levels 80--83 from 53, 89,
75, and 47 successes out of 128 to 60, 99, 100, and 93. Four new seeds then measured 263/512 (51.37%) at level 80
and 395/512 (77.15%) at level 83, with 100% bilateral acquisition and no unsafe or unfinished episodes. A standard-
deviation sweep did not explain the residual level-80 variability: deterministic and stochastic values between
`0.025` and `0.10` produced 39.06--49.22% on one seed, while every level-83 condition remained between 71.88% and
85.16%. The selected checkpoint consequently retains `std=0.10`; lowering exploration was not accepted as a fix.

The first production-threshold curriculum gate ran 40 frozen-actor, 80-step updates on eight GPUs from level 80
under ADMM (`gamma=7.5e-5`, Baumgarte `0.75`, latest contact matching). Level 80 promoted on a 56.25% window.
Level 81 then passed its default 20%, 35%, and 50% frontier exposures with approximately 58.98%, 64.06%, and 65.62%
completed-window success before promoting to level 82. Level-82 online success remained approximately 70% with
zero unsafe terminations, but the run ended before its first 128-episode frontier window completed. The final
checkpoint's eleven actor tensors were bit-identical to the distilled input; only six critic MLP tensors changed,
by at most `0.01272`.

The initial launch of this gate exposed a CLI-only failure before simulation. Passing the deprecated `--headless`
flag through the unified launcher left it in Hydra's arguments, which forced config serialization; the recursive
slice converter does not traverse tuples, so the shared `asset_cfgs` tuple raised an `UnsupportedValueType` error.
Removing `--headless` and retaining `--viz none` made all eight ranks start normally. This failure produced no
rollout data and is not attributed to the checkpoint, curriculum, or ADMM solver.

### Level-84 approach-discontinuity diagnosis

A second eight-rank frozen-actor gate resumed from the first gate's `model_66.pt`, repeated level 81 so that the
level-82 exposure schedule restarted at 20%, and collected 4,915,200 transitions over sixty 80-step updates. It
promoted level 81 on a 65.62% window; level 82 passed its 20%, 35%, and 50% exposures at approximately 77.34%,
82.81%, and 80.47%; and level 83 passed the same exposures at 69.53%, 75.78%, and 73.44%. The policy therefore
entered level 84. Its actor remained bit-identical to the distilled input, so these promotions validate the
authored curriculum and not further policy fitting.

A fixed-level level-84 gate then contradicted the mixed replay success shown immediately after promotion. Across
four independent seeds the former level 84 produced 0/512 successes: 403 episodes terminated for insufficient
separation, 90 lost grasp, and 19 missed acquisition. Bilateral acquisition remained 96.3%, usually occurring on
the first control step, but the median right-tail and separation progress components were both zero at step 64.
In comparison, level 83 had 100% acquisition and a median separation component near 0.71 at the same gate. The
new failure was therefore localized to the arm reset transition rather than grasp acquisition.

Extending the post-acquisition separation deadline did not repair the behavior. Paired 256-episode gates with
deadlines of 64, 96, 128, and 160 control steps all produced zero successes. Insufficient-separation terminations
fell from 202 to 122 while lost-grasp terminations rose from 46 to 126, showing that extra time only delayed the
same failed pull. The deadline already counts acquired control steps and was retained unchanged.

The former level-83-to-84 reset moved the right arm directly from 6.640625% to 100% of the first approach interval
while moving the left arm to its first endpoint at the same time. A source-equivalent reset scan held the left arm
at its contact pose and retained the fully open gripper and zero-warm-up approach contract. On a common seed,
right-arm fractions of 7.0%, 7.5%, 8.0%, and 9.0% produced respectively 91, 43, 23, and 34 successes out of 128;
at 10% and beyond, acquisition also collapsed. Prospective gates then measured 360/512 (70.31%) at 7.0%, 264/512
(51.56%) at 7.25%, and 188/512 (36.72%) at 7.5%, all with complete episodes, 100% bilateral acquisition, and no
unsafe termination. One 7.125% seed produced 78/128 and one 7.375% seed produced 62/128; a second 7.125% process
failed during native scene initialization and contributed no rollout data.

The production curriculum consequently adds right-only 7.0%, 7.25%, and 7.5% learning poses as levels 84--86,
leaves the left arm at its retained contact pose, and moves the five former Cartesian-approach levels to 87--91.
Levels 0--80 still use three held arm commands; every open-gripper approach level still uses immediate arm control.
The curriculum now contains 92 levels. The unchanged distilled actor should clear the first two new levels, while
level 86 supplies a populated first PPO frontier instead of the former zero-success discontinuity.

A source-only gate then removed every temporary pose override. Levels 84 and 85 produced 169/256 (66.02%) and
136/256 (53.12%) successes over two seeds, while level 86 produced 199/512 (38.87%) over four seeds. All 1,024
episodes acquired both tails and completed with zero unsafe termination. This reproduced the intervention result
through the production reset table and validated level 86 as the next learning target.

The first follow-up launch exposed a checkpoint-loading pitfall and was stopped after four updates. Although its
saved YAML reported the requested `1e-5` learning rate, the nominally empty optimizer state still contained a
parameter group with `lr=1e-4`; RSL-RL restored that group after constructing the CLI-configured optimizer. The
actor's maximum output-layer change of approximately `1e-4` per update confirmed the effective rate. No checkpoint
from `2026-09-07_12-25-51_curriculum92_outputhead_h240_lr1e5_start83_thresh50_admm_8gpu128_300_seed1050` is used
as a continuation.

The corrected follow-up run is stored under
`logs/rsl_rl/shoelace_dual_franka/2026-09-07_12-35-41_curriculum92_outputhead_h240_lr1e5_start83_thresh50_admm_8gpu128_300_seed1050_retry1`.
It uses eight GPUs with 128 environments per rank, 240-step rollouts, one full-batch PPO epoch, a fixed `1e-5`
learning rate, fixed `std=0.10`, frozen observation statistics and MLP backbone, and a trainable output layer. Its
input combines the bit-identical distilled actor with the level-84 critic-warm-up checkpoint and a fresh optimizer
whose saved `lr` and `initial_lr` are both `1e-5`. After update zero, only the actor output-layer tensors changed,
each by at most `1.0001e-5`; the saved optimizer retained the requested rate.

The run promoted level 84 and entered level 85 at 20% exposure on update 10. Level-85 completed windows initially
measured 46.46% and 43.41%, then recovered sufficiently to enter 35% exposure on update 17. Subsequent reported
frontier windows remained close to but mostly below the 50% gate: 50.22%, 49.61%, 43.91%, 43.86%, and 46.88%.
The run was stopped after the complete `model_22.pt` checkpoint so the apparent online decline could be tested
without spending the remainder of the eight-GPU run on an unverified gradient. At update 22, the aggregate episode
mix was 61.61% success, 32.07% lost grasp, and 6.32% insufficient separation, with no unsafe termination.

A paired fixed-level-85 test then compared the distilled input and `model_22.pt` on seeds 1060--1063. The input
produced 268/512 (52.34%) successes, 215 lost grasps, and 29 insufficient-separation terminations. `model_22.pt`
produced 272/512 (53.12%) successes, 220 lost grasps, and 20 insufficient-separation terminations. Both acquired
both tails in every episode and had no unfinished or unsafe episodes. The 0.78-point difference is not a material
policy improvement, but it rules out the suspected output-head regression: the lower online windows came from the
borderline level-85 distribution and changing frontier exposure, while the PPO updates were effectively neutral.

Representative stochastic rollouts of the unchanged distilled input were also recorded locally for levels 83--86
with Newton GL at 1920x1080 and 30 FPS. Using the same seed, levels 83, 84, and 86 terminated successfully after
178, 183, and 184 steps; level 85 lost its grasp after 116 steps. This one-episode set is qualitative evidence only,
not a replacement for the fixed-level rates above. Kit recording was unavailable in the Newton-only environment,
so the recorder used the repository's `video` extra and the headless Newton GL framebuffer instead.

### Level-86 causal action correction and staged approach bridge

The neutral level-85 PPO continuation motivated a causal analysis of the next frontier rather than another learning-
rate change. Eight independent level-86 datasets contained 1,024 stochastic episodes from the unchanged distilled
actor: 411 succeeded, 588 lost grasp, and 25 ended for insufficient separation. At absolute control step zero, the
successful episodes' sampled-action residual differed most strongly in right translation: right local `z` was
`+0.0544` higher and right local `x` was `-0.0476` lower than lost-grasp episodes. Both signs agreed in all eight
seeds, and the difference largely disappeared after step eight. The action coordinates are expressed in the right
robot root frame. Its 180-degree world-`z` rotation preserves `+z` as world-up and maps local `-x` to world-outward.

A controlled intervention changed only the first action. On a tuning seed, the selected local
`(x, z)=(-0.15, +0.10)` correction produced 105/128 successes, compared with the unmodified actor's approximately
40% multi-seed baseline. Four new paired seeds then measured 198/512 (38.67%) successes and 295 lost grasps for the
input, versus 376/512 (73.44%) successes and 101 lost grasps with the correction. Per-seed gains were 24.22, 36.72,
39.84, and 38.28 percentage points. The configured 5-mm Cartesian action scale makes these action-space offsets
approximately 0.75 mm outward and 0.50 mm upward at the TCP, and the correction is absent after the first control
step. This establishes contact stabilization, not a hand-authored untying trajectory, as its measured effect.

Cross-level paired gates found no retention trade-off. On one common seed, levels 83, 84, and 85 improved from
97, 88, and 74 successes out of 128 to 115, 110, and 105; two level-80 seeds changed from 123/256 to 134/256. The
former level 87 remained 0/256 both with and without the correction, however. Bilateral acquisition remained about
95%, but the right-tail progress component stayed zero. Source inspection localized this second discontinuity to
the reset table: that level simultaneously moved both arms from the retained contact poses to the first original
approach endpoints.

A naive seven-point, dual-arm joint interpolation was rejected after every interpolated level produced zero
success. At the first 1/8 point only 12.5% of episodes acquired both tails, demonstrating that joint-space distance
does not preserve the cable's contact manifold. Separate reset scans instead showed that left-arm motion was much
more tolerant than right-arm motion. With the first-step correction active, left-only fractions of 12.5% and 25%
of the remaining first-endpoint interval produced 85/128 and 75/128 successes. Right-only fractions of 1.0% and
1.25% produced 98/128 and 83/128, while 1.5%, 1.75%, 2.0%, and 4.0% fell to 27, 8, 0, and 0. Measured combinations
of left 25% with right 1.0% and 1.25% produced 72/128 and 64/128; left 50% with right 1.25% retained 45/128.

The production reset table therefore stages five nodes after level 86: `(left, right)` fractions
`(12.5%, 0%)`, `(25%, 0%)`, `(25%, 1%)`, `(25%, 1.25%)`, and `(50%, 1.25%)`. The five former approach levels move
to 92--96, increasing the curriculum from 92 to 97 levels. The gripper remains fully open and the approach command
warm-up remains disabled. A source-only common-seed gate with the runtime correction measured 89, 73, 76, 76, and
66 successes out of 128 across new levels 87--91, with 100% bilateral acquisition at every level.

The intervention was then distilled into the actor output head. Paired current-policy and corrected trajectories
from levels 80, 83, 84, 85, and 86 supplied reset-step targets and steps 1--64 supplied zero-delta anchors. A held-
out environment split selected positive weight 8 and ridge `0.01`. Its executed reset-step mean was
`(-0.149999, +0.099999)` with component MAE about `6e-6` and `4e-6`; later absolute error had a 95th percentile of
about `8.8e-4`. Only output rows 7 and 9 changed, while the critic, observation statistics, backbone, other action
rows, and fixed standard deviation remained unchanged. The checkpoint is
`stabilization_inputs/curriculum97_level86_xz_distilled_xm015_zp010_step0_all80to86_w8_l4_r0.01.pt`.

Policy-only gates contained no runtime correction. One seed produced 69/128 at level 80 and 104, 108, 103, 103,
90, 74, and 70 successes at levels 83--89. Four level-86 seeds totaled 398/512 (77.73%). Level 90 produced 62/128
on one seed, while four level-91 seeds totaled 189/512 (36.91%), making levels 90--91 a populated learning frontier
rather than claiming that distillation already solved the remaining approach. A 60-update, eight-GPU frozen-policy
critic warm-up was subsequently launched from level 86 with 128 environments per rank, an 80-step horizon, and
fresh Adam at `3e-4`; its run directory starts with `2026-09-07_14-29-21_curriculum97_xz_distilled_frozenpolicy`.
