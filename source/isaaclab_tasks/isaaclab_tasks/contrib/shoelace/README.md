# Dual-Franka shoelace task

For multi-GPU training, `--num_envs` is the number of environments on each GPU.
RSL-RL collects 16 steps per environment before each PPO update. With 8 GPUs and
1024 environments per GPU, each iteration collects 131,072 transitions in total.
Use `agent.num_steps_per_env=32` to restore the previous rollout length.
The outer Newton collision pipeline reserves at least `max(1_000_000, 8192 * num_envs)`
triangle pairs per process. The internal ADMM pipeline keeps 1,000,000 triangle pairs
and scales its contact-reduction hashtable factor to reserve at least 2048 slots per
environment. At 1024 environments per GPU, the factor is 2.097152 and the hashtable
has 2,097,152 slots. Larger explicit budgets and factors are preserved.
Override the internal budget with `env.sim.physics.solver_cfg.contact_max_triangle_pairs`
and its table factor with `env.sim.physics.solver_cfg.contact_reduction_hashtable_size_factor`.
These affect memory allocation at startup; they do not change the solver timestep or contact material.
Contact matching remains enabled. Newton 1.6 requires the internal triangle-pair
budget to stay below `2**20` in this mode; increase the hashtable factor for table warnings.

`IsaacContrib-Shoelace-DualFranka` exposes the standalone Newton shoelace scene as a manager-based RL
environment. The initial interface intentionally includes only dual-arm Cartesian actions, binary gripper
actions, proprioceptive and tail observations, compact finger-tail signed-distance history, randomized resets,
dense task and pregrasp progress rewards, grasp retention and success rewards, small arm action penalties,
and episode timeouts.
The compact contact-related policy
input contains:

- four finger-tail signed surface distances, clipped to +/-2 mm and stacked over the current and previous two
  policy steps (12 values, oldest to newest); positive means separation, zero means touching, and negative means
  penetration;
- two positive actual-minus-target gripper closure residuals; and
- two free-tail speed magnitudes relative to the controlling TCPs.

The signed-distance history is derived directly from Newton collision candidates because coupled solvers do not
expose the standard contact-force sensor. A pair without a collision candidate uses the positive 2 mm cap.

The dense reward covers approaching, grasping, and pulling the tails, with independent credit and a bilateral
bonus in each phase. Let `H(a, b) = a * b / (a + b - a * b + 1e-6)` be the Hamacher soft-AND,
`A` the per-arm TCP proximity, `G` the filtered per-arm grasp quality, and `d` each tail's
outward X displacement from the first valid post-reset sample. The left arm pulls toward negative X, and the
right arm toward positive X. Positions are relative to the fixed cable seam midpoint, so translating the shoe
or the whole environment does not create progress. The default is:

```python
A = 1 - tanh(tail_tcp_distance / 0.08)  # metres
approach = 0.5 * A.mean(dim=-1) + 0.5 * H(A[:, 0], A[:, 1])
grasp = 0.3 * G.mean(dim=-1) + 0.7 * H(G[:, 0], G[:, 1])
scale = max((0.18 - initial_x_separation) / 2, 0.01)  # per environment, metres
P = 0.5 * (1 + tanh(d / scale))
per_arm_pull = H(G, P)
bilateral_pull = H(per_arm_pull[:, 0], per_arm_pull[:, 1])
pull = 0.2 * per_arm_pull.mean(dim=-1) + 0.8 * bilateral_pull
potential = 0.3 * approach + 0.3 * grasp + 0.4 * pull
reward_rate = (potential - previous_potential) / step_dt
```

`acquisition_weight=0.6` and `approach_fraction=0.5` divide the potential into 30% approach, 30% grasp, and
40% pull. The manager multiplies the returned rate by `step_dt` and the overall weight (10), giving maximum
progress budgets of 3, 3, and 4. `bilateral_approach_fraction=0.5` and `bilateral_grasp_fraction=0.7` reserve
half the approach budget and 70% of grasp acquisition for cooperation. Independent credit starts either arm;
bringing the other arm into the same phase earns a larger increment. With ideal scores, one approached arm
earns 0.75 of the approach budget and approaching the other adds 2.25. One grasp earns 0.45 of the acquisition
grasp budget; both grasps earn 3. These are potential gains, not rewards paid on every stationary step.

Grasp credit remains independent of TCP proximity. Each grasp requires both finger surfaces near contact,
actual gripper closure, and low tail-TCP slip. The 0.10 s filter smooths contact flicker and also delays the
response to release. Each arm's pull score uses only its own grasp and tail position. Moving the ungrasped
tail cannot earn the other arm's pull credit.

`bilateral_pull_fraction=0.8` reserves 80% of pulling for cooperation and leaves 20% available independently.
The smooth `P=0.5` at the initial tail position preserves outward feedback even below the reset baseline.
It still grants some pull credit on acquisition: with the other hand ungrasped, one perfect stationary grasp
adds 0.20 weighted pull potential, down from 1.40 in the previous defaults. It does not imply successful
pulling. References remain fixed throughout the episode, including releases and regrasping. Each bilateral
fraction can be set to zero to use only the corresponding per-arm mean.

The `dense_task` term differences one potential, without rate clipping or a stage-switch gate. A closed cycle of its
full state has zero undiscounted dense reward; stationary states give zero once the filter settles. Invalid
samples do not advance its state, and the first valid sample after reset gives no dense reward. This is a
progress objective, not a claim of optimal-policy invariance under discounting.

The independent `pregrasp` progress term bridges coarse approach and the first physical grasp. It uses the same
three-segment tail centers and TCP positions as the policy's `tails_to_tcp` observation:

```python
alignment = exp(-(tail_tcp_distance / 0.015)**2)
closure_gate = maximum(1 - (tail_tcp_distance / 0.01)**2, 0)**2
closure = clip((0.01 - actual_finger_position) / (0.01 - 0.001), 0, 1)
pregrasp_potential = (0.75 * alignment + 0.25 * closure_gate * closure).mean(dim=-1)
pregrasp_reward = pregrasp_potential - previous_pregrasp_potential  # manager weight = 1
```

The 15 mm Gaussian width adds fine positional guidance inside the broader 80 mm approach scale. The closure
gate is exactly zero at and beyond 10 mm and rises smoothly toward the TCP, so closing far from the tail adds
no credit. Both terms reward each arm independently: ideal alignment contributes at most 0.375 per arm, and
nearby actual closure adds at most 0.125 per arm. Their combined maximum potential is 1 across both arms,
smaller than the existing physical grasp and pull budgets. A close command without actual joint motion does
not change the closure score. The gate does not shrink as the gripper closes.

This is a TCP-centered positional heuristic, not a test of tail orientation, collision-free capture, or physical
grasping. Empty closure inside the gate can earn a small, bounded progress increment; it earns no repeated
pregrasp reward for holding still and does not count as contact-based grasp, retention, or success. Opening or
moving away removes the corresponding potential. Closed state cycles have zero undiscounted pregrasp return;
discounted optimal-policy invariance is not claimed. Invalid distances or finger positions return zero without
advancing history; the first valid sample after a per-environment reset seeds history without reset credit.

The two components can be ablated without rescaling the other: set
`env.rewards.pregrasp.params.closure_weight=0` for fine positional guidance alone, or set `alignment_weight=0`
for gated closure alone. Set `env.rewards.pregrasp.weight=0` to recover the previous reward objective with all
other terms unchanged. The default combines both components as an initial shaping candidate, not a trained
result. Compare near-tail close-command frequency, contact-based grasp quality, and continuous retention
duration; change one component at a time in follow-up experiments. Observation, action, and PPO interfaces
are unchanged, so checkpoints remain loadable but need re-evaluation under the new objective.

The independent `grasp_hold` term rewards time spent retaining physical grasps:

```python
hold = 0.5 * G.mean(dim=-1) + 0.5 * H(G[:, 0], G[:, 1])
hold_reward = 1.0 * step_dt * hold
```

Its weight is a maximum reward rate of 1.0 per second: ideal bilateral retention pays 1.0 per second and
one-sided retention pays 0.25. Empty closure earns approximately zero. It owns its filter, so disabling or
reordering `dense_task` does not change retention evaluation. Invalid contact, closure, or slip inputs earn
zero and preserve filter history; reset clears only the selected environments. When tuning contact or filter
parameters, update both `dense_task.params` and `grasp_hold.params` to keep their grasp definitions aligned.
Set `env.rewards.grasp_hold.weight=0` for a progress-only ablation.
To restore the previous retention budget, set `env.rewards.grasp_hold.weight=0.2` and
`env.rewards.grasp_hold.params.bilateral_grasp_fraction=0.8`. The current defaults increase ideal bilateral
retention credit fivefold and single-arm credit 12.5-fold, without changing the physical grasp criteria.

The `success` reward pays +5 on the existing geometric success termination, including success on the timeout
step. Timeout alone pays no success reward. `mdp.shoelace_success_reward` divides the success flag by
`step_dt`, so the manager's timestep multiplication leaves +5 per event. The environment then resets.
Ideal retention throughout the current 10-second episode can contribute at most 10; completion earns a
separate event reward. Holding still can therefore compete with task completion. These are acquisition
and retention tuning budgets, not a guarantee that training will learn to pull.
Evaluate simultaneous approach, continuous grasp duration, gripper switching, outward displacement, and
geometric success when comparing runs.

**Migration:** The observation/action interface is unchanged, but the reward objective and defaults changed.
`approach_fraction` now linearly divides acquisition between the cooperative approach and grasp scores;
the previous `H(A, approach_fraction)` saturation was removed. Existing parameter names remain accepted,
but restoring their old values does not reproduce the old reward formula. Re-evaluate or retrain policies,
configure the new bilateral fractions explicitly for custom tasks, and disable `grasp_hold` and `success`
independently for ablations. Historical reward curves are not directly comparable.

Two independent penalties regularize the 12 normalized arm commands before Cartesian scaling. Both select the
`left_arm` and `right_arm` action terms by name and exclude binary gripper commands:

- `arm_action_rate` has weight `-0.001` and sums squared changes from the previous policy step to discourage
  jitter and frequent direction changes. This is a step difference, without division by `step_dt`.
- `arm_action_magnitude` has weight `-0.001` and sums squared commands to discourage unnecessary motion,
  including constant commands that incur no action-change cost.

The reward manager multiplies both penalties by `step_dt`, so the total per-step reward is:

```python
reward = 10.0 * (potential - previous_potential) + pregrasp_reward + hold_reward + 5.0 * success - step_dt * (
    0.001 * (arm_action - previous_arm_action).square().sum(dim=-1)
    + 0.001 * arm_action.square().sum(dim=-1)
)
```

Both progress terms are zero on their first valid samples as described above; other reward terms still apply.
Action history resets to zero per environment, so the first command is compared with zero. The total reward
therefore does not have the dense term's zero-return closed-cycle property. These small initial weights have
not been tuned through training. Adjust them independently with `env.rewards.arm_action_rate.weight` and
`env.rewards.arm_action_magnitude.weight`.

The dense term writes phase metrics to `extras["log"]` on every policy step. RSL-RL prints their rollout
averages in each training iteration and writes the same tags to TensorBoard. Prefer physical displacement
to the grasp-gated score when checking whether a tail actually moved outward:

| Tag under `Metrics/shoelace/` | Interpretation |
| --- | --- |
| `approach_distance_m` | Mean TCP-to-tail distance across both arms [m]; lower is better. |
| `grasp_left`, `grasp_right` | Filtered contact, closure, and low-slip grasp quality for each arm in [0, 1]. |
| `grasp_both` | Hamacher soft-AND of both filtered grasp qualities in [0, 1]. |
| `pull_x_separation_m` | Absolute two-tail X separation [m]; 0.18 m is one necessary success condition. |
| `pull_left_displacement_m`, `pull_right_displacement_m` | Signed outward tail displacement from the episode baseline [m], independent of grasp quality. Positive means outward; negative means inward. |
| `pull_left_score`, `pull_right_score` | Each tail's progress gated by its own filtered grasp, in [0, 1]; not a distance or success rate. |
| `pull_left`, `pull_right` | Deprecated aliases of `pull_left_score` and `pull_right_score`; their values have not changed. |
| `success_rate` | Mean success over each environment's most recent completed episode. |
| `valid_fraction` | Fraction of environments whose checked reward inputs contain no NaN/Inf; normally 1.0. |

Each displacement uses the mean position of the corresponding tail's three free-end segments relative
to the fixed seam midpoint. Left-arm outward motion is negative X; right-arm outward motion is positive X.
The baseline is the first finite reward sample after that environment resets, not the first grasp. Thus
`0.02` means 2 cm farther outward than the baseline, `-0.01` means 1 cm inward, and acquiring a stationary
grasp leaves displacement at zero. Releasing and regrasping do not rebase it. The displacement is neither
clipped nor grasp-gated: passive cable motion also counts, so it does not prove that the gripper caused
the motion. A stationary ideal grasp can still produce a pull **score** near 0.5 because the reward's
internal progress starts at 0.5; the new displacement metrics do not change that reward formula.

`finite` is the internal per-environment numerical mask. It checks tail-to-TCP distances, finger-tail signed
distances, relative speeds, closure fractions, tail X separation, and outward tail offsets. If any checked
value is NaN or infinite, that environment's dense reward is zero and its reward history is preserved.
`valid_fraction = finite.float().mean()` reports the fraction passing this check in the current step.
For example, 1023 valid environments out of 1024 gives approximately 0.9990. This checks the numerical
validity of these reward inputs; it does not measure grasp validity, task success, or physical plausibility.

Phase metrics average over valid environments before automatic reset. When no environment is valid, those
metrics are zero and `valid_fraction=0` identifies that case. `success_rate` uses the actual `success` termination
flag, excludes environments that have not finished an episode, and is zero until the first completion. These
continuous grasp scores are contact-based proxies, not discrete stage-completion labels. For example, high
`pull_x_separation_m` with low `grasp_both` indicates separated tails without strong simultaneous grasps. Metrics are
computed by the dense reward term and require its weight to remain nonzero.
RSL-RL then averages the per-step values over its rollout window. In the current multi-GPU runner these
custom metrics are logged from rank 0's local environments, not reduced across all GPUs. Displacement
curves are therefore mean signed displacements, not per-episode maximum distances; opposite motions
across environments can cancel in the average.

`grasp_both` averages the per-environment joint grasp quality. It cannot be recovered by combining the logged
averages of `grasp_left` and `grasp_right`: different environments may hold different single tails.

Dashboard migration: use `pull_left_displacement_m` and `pull_right_displacement_m` for physical progress,
and `pull_left_score` and `pull_right_score` when inspecting the reward's internal scores. The old
`pull_left` and `pull_right` tags remain available as deprecated score aliases for compatibility; do not
interpret old curves as meters or compare score and displacement values directly. Existing event files
are not rewritten. Reward calculations, including slip-sensitive grasp quality and the bilateral pull
bonus, are unchanged by this logging update.

The reward manager separately logs `Episode_Reward/pregrasp`, `Episode_Reward/grasp_hold`, `Episode_Reward/success`,
`Episode_Reward/arm_action_rate`, and `Episode_Reward/arm_action_magnitude` when episodes reset.
These are episode sums divided by the configured episode duration, not raw event rewards.

Success now uses `mdp.shoelace_success` and requires all of the following:

- no more than 52 free cable segments inside a sphere of radius 0.025 m around the fixed seam midpoint;
- both tail centers at least 0.09 m from that midpoint;
- absolute two-tail X separation of at least 0.18 m; and
- finite cable segment positions and velocities.

The throat and tail-distance criteria follow `reb/newton_shoelace_demo`. The midpoint is the average of the
left cable's last segment and the right cable's first segment. These two fixed anchors are excluded from
the throat count; the remaining 156 segment centers are counted. The region follows shoe translations.
This rejects separation-only success while a dense knot remains in the throat or one tail stays near it.
It is a regional geometry heuristic, not a general topological proof of untying: changes to the cable asset,
segment resolution or knot geometry require revalidating the radius and count threshold. Unlike the old
branch, success does not require both grippers to remain closed; cooperation is rewarded through the bonus.

The previous `mdp.tail_x_separation_success` helper remains available for custom distance-only evaluations.
Custom tasks should switch to `mdp.shoelace_success` with the four thresholds in `TerminationsCfg.success`.
Historical success rates from the distance-only criterion are not comparable to the new rates.

The deprecated `maximum_progress_rate`, `approach_weight`, `grasp_weight`, and `task_weight` parameters
have been removed. If only `maximum_progress_rate` was used, delete that key. Configurations using any
of the three old weights must be migrated before loading:

1. For old weights `a` (approach), `g` (grasp), and `p` (task), set
   `acquisition_weight=(a+g)/(a+g+p)` and `approach_fraction=a/(a+g)`; use zero for the latter if `a+g=0`.
   Omitted old weights defaulted to `a=0.15`, `g=0.35`, and `p=1.0`.
2. Multiply the reward manager's overall term weight by `a+g+p` to preserve the potential scale.
3. Delete all four deprecated parameters. The ignored `maximum_progress_rate` has no replacement.

Pass `cable_cfgs` and `robot_cfgs` by keyword in direct calls; removing the old parameters changed their
positional indices. The migration preserves phase budgets, not historical reward trajectories. Current
task defaults already use `acquisition_weight`, `approach_fraction`, and `bilateral_pull_fraction`.

The task uses the assets in `scripts/demos/shoelace/assets`.

## Reset randomization

Each episode independently samples uniform offsets for every selected environment:

- Each of the seven joints on each arm starts within +/-0.02 rad (about 1.15 degrees) of its nominal
  pregrasp position, clamped to the soft joint limits. Finger positions stay at the default open position.
  Joint velocities stay at their defaults, and arm position targets match the sampled positions.
- The shoe moves by up to +/-0.02 m along each of X and Y; its height and orientation stay unchanged.
  Both settled laces, including their fixed anchor segments, receive the same translation. The shoe collider,
  tongue, visual mesh and pinned lace mesh move with the kinematic shoe, which stays fixed during the episode.

These conservative ranges vary the initial approach without changing the knot geometry. They help diversify
training starts; broader generalization still needs evaluation on held-out starts.

The ranges live in `EventsCfg.reset_left_arm`, `reset_right_arm`, and `reset_shoe` in `shoelace_env_cfg.py`.
To restore the previous deterministic starts, keep `reset_scene` and disable the three randomization terms
before constructing the environment:

```python
cfg.events.reset_left_arm = None
cfg.events.reset_right_arm = None
cfg.events.reset_shoe = None
```

`shoe_asset_cfg()` now returns a `RigidObjectCfg` for the kinematic shoe. The pinned mesh prim moved from
`{ENV_REGEX_NS}/ShoelacePinned` to `{ENV_REGEX_NS}/Shoe/ShoelacePinned`; update custom prim-path lookups accordingly.
The scene entity name `shoelace_pinned_visual` is unchanged.

## Explicit proxy inertia

`ShoelaceEnvCfg.cable_inertia_regularization` defaults to `1e-6` [kg*m^2]. Before Newton
finalization, each dynamic cable segment receives `I_effective = I_geometry + regularization * identity`
after capsule mass/radius correction and anchor pinning. Its inverse inertia is updated too. Masses,
fixed anchors, shoe and robot inertias are unchanged. This is an intentional approximation of rotational
dynamics for the current solver budget, not extra damping or a change to the cable's material stiffness.
The default approximates the previously used effective inertia without depending on Newton's automatic repair.

Override it with `env.cable_inertia_regularization=3e-7` appended to the training command below.
Values must be finite and nonnegative. Zero disables only the task-local addition: Newton's normal validation
still applies and may enlarge very small inertias. Recheck passive settling, contact penetration and scripted
grasp/pull behavior when changing this value; previous settled states and policies may behave differently.
The standalone demos expose the same setting as `--cable_inertia_regularization`.

```bash
uv run isaaclab random_agent --task IsaacContrib-Shoelace-DualFranka --num_envs 4 --device cuda:0
uv run isaaclab train --rl_library rsl_rl --task IsaacContrib-Shoelace-DualFranka \
    --num_envs 4 --max_iterations 1 --device cuda:0
```
