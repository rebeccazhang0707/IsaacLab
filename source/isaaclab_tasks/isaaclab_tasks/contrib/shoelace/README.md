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
a unified dense task reward, small arm action penalties, and episode timeouts. The compact contact-related policy
input contains:

- four finger-tail signed surface distances, clipped to +/-2 mm and stacked over the current and previous two
  policy steps (12 values, oldest to newest); positive means separation, zero means touching, and negative means
  penetration;
- two positive actual-minus-target gripper closure residuals; and
- two free-tail speed magnitudes relative to the controlling TCPs.

The signed-distance history is derived directly from Newton collision candidates because coupled solvers do not
expose the standard contact-force sensor. A pair without a collision candidate uses the positive 2 mm cap.

One reward covers acquiring the tails and pulling each outward with its corresponding grasp. Let `H` be the
Hamacher soft-AND, `A` the per-arm TCP proximity, `G` the filtered per-arm grasp quality, and `d` each tail's
outward X displacement from the first valid post-reset sample. The left arm pulls toward negative X, and the
right arm toward positive X. Positions are relative to the fixed cable seam midpoint, so translating the shoe
or the whole environment does not create progress. The default is:

```python
acquire = H(A, 0.3).mean(dim=-1) + 0.7 * G.mean(dim=-1)
scale = max((0.18 - initial_x_separation) / 2, 0.01)  # per environment, metres
P = 0.5 * (1 + tanh(d / scale))
per_arm_pull = H(G, P)
bilateral_pull = H(per_arm_pull[:, 0], per_arm_pull[:, 1])
pull = 0.8 * per_arm_pull.mean(dim=-1) + 0.2 * bilateral_pull
potential = 0.3 * acquire + 0.7 * pull
reward_rate = (potential - previous_potential) / step_dt
```

`approach_fraction=0.3` provides partial acquisition credit before contact. The remaining acquisition budget
rewards the two grasps independently and additively, without scaling their credit by TCP proximity. Each grasp
requires both finger surfaces near contact, actual gripper closure, and low tail-TCP slip. The 0.10 s filter
smooths contact flicker and also delays the response to release. Each arm's pull score uses only its own
grasp and tail position. Moving the ungrasped tail cannot earn the other arm's pull credit.

At fixed approach, the acquisition grasp contribution to the weighted potential is
`1.05 * (G_left + G_right)`: one perfect grasp contributes 1.05 and two contribute 2.10. Partial grasps earn
proportional credit, and either hand can be acquired first. These are cumulative gains as grasp quality rises,
not a reward paid every step for holding still. Releasing a grasp removes its credit through the same signed
potential difference. Pulling also contributes grasp credit because `P=0.5` at the initial tail position.
This positive, smooth progress value gives feedback while pulling outward even below the reset baseline.
It approaches one as the tail moves outward, rather than declaring the knot untied at a target distance.
`bilateral_pull_fraction=0.2` reserves 20% of the pulling budget for the bilateral bonus; the other 80% is
available independently to the two arms. Both grasping and actual outward motion increase the bilateral
score, while stationary states stop earning reward once filtering settles. Set the fraction to zero to
disable the bonus. References remain fixed throughout the episode, including releases and regrasping.

Previously, the pulling score required both grasps and clamped separation progress to zero below its reset
baseline. The new formula changes intermediate rewards while preserving maximum phase budgets. Current
phase-fraction configurations and policy checkpoints remain loadable; migrate deprecated parameters as
described below. Re-evaluate or retrain policies and compare grasp quality and geometric success rather
than comparing old and new reward curves directly.

`acquisition_weight=0.3` allocates 30% of the potential to acquisition and 70% to pulling. The manager multiplies
the returned rate by `step_dt` and the overall reward weight (10). Only one potential is differenced, without
rate clipping or a stage-switch gate. For this dense term, a closed cycle of its full state has zero undiscounted
total reward; stationary states give zero once the grasp filter settles. Invalid samples do not advance its
state, and the first valid sample after each reset gives no dense reward. These are progress rewards, not a claim
of optimal-policy invariance under discounting.

Two independent penalties regularize the 12 normalized arm commands before Cartesian scaling. Both select the
`left_arm` and `right_arm` action terms by name and exclude binary gripper commands:

- `arm_action_rate` has weight `-0.001` and sums squared changes from the previous policy step to discourage
  jitter and frequent direction changes. This is a step difference, without division by `step_dt`.
- `arm_action_magnitude` has weight `-0.001` and sums squared commands to discourage unnecessary motion,
  including constant commands that incur no action-change cost.

The reward manager multiplies both penalties by `step_dt`, so the total per-step reward is:

```python
reward = 10.0 * (potential - previous_potential) - step_dt * (
    0.001 * (arm_action - previous_arm_action).square().sum(dim=-1)
    + 0.001 * arm_action.square().sum(dim=-1)
)
```

The dense contribution is zero on its first valid sample as described above; action penalties still apply.
Action history resets to zero per environment, so the first command is compared with zero. The total reward
therefore no longer has the dense term's zero-return closed-cycle property. These small initial weights have
not been tuned through training. Adjust them independently with `env.rewards.arm_action_rate.weight` and
`env.rewards.arm_action_magnitude.weight`; set both to zero to restore the previous reward objective.

The dense term writes nine phase metrics to `extras["log"]` on every policy step. RSL-RL prints their rollout
averages in each training iteration and writes the same tags to TensorBoard:

| Tag under `Metrics/shoelace/` | Interpretation |
| --- | --- |
| `approach_distance_m` | Mean TCP-to-tail distance across both arms [m]; lower is better. |
| `grasp_left`, `grasp_right` | Filtered contact, closure, and low-slip grasp quality for each arm in [0, 1]. |
| `grasp_both` | Hamacher soft-AND of both filtered grasp qualities in [0, 1]. |
| `pull_x_separation_m` | Absolute two-tail X separation [m]; 0.18 m is one necessary success condition. |
| `pull_left`, `pull_right` | Each tail's progress gated by its own filtered grasp. |
| `success_rate` | Mean success over each environment's most recent completed episode. |
| `valid_fraction` | Fraction of environments whose checked reward inputs contain no NaN/Inf; normally 1.0. |

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

`grasp_both` averages the per-environment joint grasp quality. It cannot be recovered by combining the logged
averages of `grasp_left` and `grasp_right`: different environments may hold different single tails.

The log was reduced to nine metrics by removing `approach_score`, `grasp_slip_mps`, `pull_progress`, and
`pull_score`, and adding the per-arm `pull_left` and `pull_right` scores. Update dashboards to use the
physical approach distance, per-arm grasp/pull scores, actual separation, and success rate above. These are
diagnostic replacements, not numerically equivalent signals. Reward calculations, including slip-sensitive
grasp quality and the bilateral pull bonus, are unchanged by this logging reduction.

The reward manager separately logs the weighted penalties as `Episode_Reward/arm_action_rate` and
`Episode_Reward/arm_action_magnitude` when episodes reset.

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
