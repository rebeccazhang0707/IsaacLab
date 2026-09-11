# Dual-Franka shoelace task

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

One reward covers two phases: acquiring the tails and pulling them apart while holding both. Let `H` be the
Hamacher soft-AND, `A` the per-arm TCP proximity, `G` the filtered per-arm grasp quality, and `X` the normalized
X-separation progress from the first valid post-reset sample to the 0.18 m success threshold. The default is:

```python
acquire = H(A, 0.3).mean(dim=-1) + 0.7 * G.mean(dim=-1)
pull = H(H(G[:, 0], G[:, 1]), X)
potential = 0.3 * acquire + 0.7 * pull
reward_rate = (potential - previous_potential) / step_dt
```

`approach_fraction=0.3` provides partial acquisition credit before contact. The remaining acquisition budget
rewards the two grasps independently and additively, without scaling their credit by TCP proximity. Each grasp
requires both finger surfaces near contact, actual gripper closure, and low tail-TCP slip. The 0.10 s filter
smooths contact flicker and also delays the response to release. Pulling has no approach floor: its score
combines both grasp qualities with achieved separation.

At fixed approach and zero pulling progress, the default grasp contribution to the weighted potential is
`1.05 * (G_left + G_right)`: one perfect grasp contributes 1.05 and two contribute 2.10. Partial grasps earn
proportional credit, and either hand can be acquired first. These are cumulative gains as grasp quality rises,
not a reward paid every step for holding still. Releasing a grasp removes its credit through the same signed
potential difference.

Previously, grasp quality was inside `H(A, 0.3 + 0.7 * G)`, which reduced grasp gains when the tail center was
offset from the TCP even with good physical contact. The new formula keeps the no-grasp approach potential
and maximum stage budgets, but changes intermediate rewards. Existing configurations and checkpoints remain
loadable; re-evaluate or retrain policies under this objective and compare grasp quality and success rather
than comparing old and new reward curves directly.

`acquisition_weight=0.3` allocates 30% of the potential to acquisition and 70% to pulling. The manager multiplies
the returned rate by `step_dt` and the overall reward weight (10). Only one potential is differenced, without
rate clipping or a stage-switch gate. For this dense term, a closed cycle of its full state has zero undiscounted
total reward; stationary states give zero once the grasp filter settles. Invalid samples do not advance its
state, and the first valid sample after each reset gives no dense reward. These are progress rewards, not a claim
of optimal-policy invariance under discounting.

Two independent penalties regularize the 12 normalized arm commands before Cartesian scaling. Both select the
`left_arm` and `right_arm` action terms by name and exclude binary gripper commands:

- `arm_action_rate` has weight `-0.01` and sums squared changes from the previous policy step to discourage
  jitter and frequent direction changes. This is a step difference, without division by `step_dt`.
- `arm_action_magnitude` has weight `-0.001` and sums squared commands to discourage unnecessary motion,
  including constant commands that incur no action-change cost.

The reward manager multiplies both penalties by `step_dt`, so the total per-step reward is:

```python
reward = 10.0 * (potential - previous_potential) - step_dt * (
    0.01 * (arm_action - previous_arm_action).square().sum(dim=-1)
    + 0.001 * arm_action.square().sum(dim=-1)
)
```

The dense contribution is zero on its first valid sample as described above; action penalties still apply.
Action history resets to zero per environment, so the first command is compared with zero. The total reward
therefore no longer has the dense term's zero-return closed-cycle property. These small initial weights have
not been tuned through training. Adjust them independently with `env.rewards.arm_action_rate.weight` and
`env.rewards.arm_action_magnitude.weight`; set both to zero to restore the previous reward objective.

The dense term also writes phase metrics to `extras["log"]` on every policy step. RSL-RL prints their rollout
averages in each training iteration and writes the same tags to TensorBoard:

| Tag under `Metrics/shoelace/` | Interpretation |
| --- | --- |
| `approach_distance_m` | Mean TCP-to-tail distance across both arms [m]; lower is better. |
| `approach_score` | Mean proximity score in [0, 1]; higher is better. |
| `grasp_left`, `grasp_right` | Filtered contact, closure, and low-slip grasp quality for each arm in [0, 1]. |
| `grasp_both` | Hamacher soft-AND of both filtered grasp qualities in [0, 1]. |
| `grasp_slip_mps` | Mean tail-to-TCP relative speed across both arms [m/s]; lower is better while grasping. |
| `pull_x_separation_m` | Absolute two-tail X separation [m]; success starts at 0.18 m. |
| `pull_progress` | Reset-relative separation progress in [0, 1], independent of grasp quality. |
| `pull_score` | Separation progress combined with both grasps in [0, 1]. |
| `success_rate` | Mean success over each environment's most recent completed episode. |
| `valid_fraction` | Fraction of environments with finite reward inputs in the current step. |

Phase metrics average over valid environments before automatic reset. When no environment is valid, those
metrics are zero and `valid_fraction=0` identifies that case. `success_rate` uses the actual `success` termination
flag, excludes environments that have not finished an episode, and is zero until the first completion. These
continuous grasp scores are contact-based proxies, not discrete stage-completion labels. For example, high
`pull_progress` with low `grasp_both` means the tails separated without both grasps being maintained. Metrics are
computed by the dense reward term and require its weight to remain nonzero.

The reward manager separately logs the weighted penalties as `Episode_Reward/arm_action_rate` and
`Episode_Reward/arm_action_magnitude` when episodes reset.

Success checks only absolute two-tail X-position separation reaching 0.18 m; it does not require grasping or
prove that the knot is topologically untied. The dense pulling score still requires both grasps.

Earlier three-stage configs remain accepted with a deprecation warning. To migrate weights `a` (approach), `g`
(grasp), and `p` (task), set `acquisition_weight=(a+g)/(a+g+p)` and `approach_fraction=a/(a+g)`, and multiply the
manager's overall weight by `a+g+p` to preserve the potential scale. If `a+g` is zero, use zero for
`approach_fraction`. Remove `maximum_progress_rate`, which is now ignored. This preserves stage budgets, not the
old reward trajectory: the acquisition formula and treatment of regressions have changed. The new default
fractions above are independent of that compatibility mapping.

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
