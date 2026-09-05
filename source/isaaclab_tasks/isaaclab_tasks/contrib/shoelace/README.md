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
| PPO rollout horizon | 32 policy steps per environment |
| PPO checkpoint interval | 10 iterations |
| PPO configured maximum | 1,000 iterations |

Setting `env.coupling_mode=admm` switches the runtime to symmetric ADMM two-way coupling. The current
ADMM defaults are iterations `5`, rho `400`, gamma `0`, Baumgarte `0.5`, and rigid-contact matching
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
| 9 | `last_action` | 14 | 1 | Previous raw action supplied to the environment. |

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

### Gripper actions

Each gripper contributes one continuous policy output but is mapped to a binary joint-position target:

| Raw action | Command | Driven finger target |
|---:|---|---:|
| `< 0` | Close | `0.002 m` |
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

The `0.002 m` close target lies inside this strict interval. Acquisition, dense task shaping, and lost-grasp
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
unsafe or lost-grasp termination: -12 once
timeout: 0 terminal reward
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

### `time_out`

The episode truncates after 600 policy steps (`20 s`). Timeout is marked separately from the three normal
terminations and has no explicit terminal reward or penalty.

## Reset and curriculum context

Every reset restores the authored cable pose and zero cable velocity. The 56-level curriculum then applies
calibrated arm and gripper states:

- levels 0--50 keep both arms at the settled grasp pose while the gripper target moves from `0.002` to `0.005 m`;
- the contact-release intervals around `0.0035--0.003625 m` and `0.004--0.004047 m` use measured micrometre-scale
  reset increments;
- levels 51--55 keep the gripper fully open and progressively move the arms through calibrated approach poses
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
  -> success, unsafe/lost-grasp failure, or timeout
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
the improving frontier to finish its windows, rather than reacting to the replay-mixed aggregate success trough,
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

With the retuned drive, a 20-update continuation promoted levels 29, 30, 31, 32, and 33 with qualifying windows
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
