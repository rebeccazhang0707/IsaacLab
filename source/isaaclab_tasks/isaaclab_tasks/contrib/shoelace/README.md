# Dual-Franka shoelace task

`IsaacContrib-Shoelace-DualFranka` exposes the standalone Newton shoelace scene as a manager-based RL
environment. The initial interface intentionally includes only dual-arm Cartesian actions, binary gripper
actions, proprioceptive and tail observations, compact finger-tail signed-distance history, deterministic resets,
a unified dense task reward, and episode timeouts. The compact contact-related policy input contains:

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
acquire = H(A, 0.3 + 0.7 * G).mean(dim=-1)
pull = H(H(G[:, 0], G[:, 1]), X)
potential = 0.3 * acquire + 0.7 * pull
reward_rate = (potential - previous_potential) / step_dt
```

`approach_fraction=0.3` provides partial acquisition credit before contact. Grasping either tail increases
acquisition, so there is no separate grasp bonus. Each grasp requires both finger surfaces near contact, actual
gripper closure, and low tail-TCP slip. The 0.10 s filter smooths contact flicker and also delays the response to
release. Pulling has no approach floor: its score combines both grasp qualities with achieved separation.

`acquisition_weight=0.3` allocates 30% of the potential to acquisition and 70% to pulling. The manager multiplies
the returned rate by `step_dt` and the overall reward weight (10). Only one potential is differenced, without
rate clipping or a stage-switch gate. A closed cycle of the full reward state has zero undiscounted total reward;
stationary states give zero once the grasp filter settles. Invalid samples do not advance the reward state, and
the first valid sample after each reset gives no reward. These are progress rewards, not a claim of optimal-policy
invariance under discounting.

Success checks only absolute two-tail X-position separation reaching 0.18 m; it does not require grasping or
prove that the knot is topologically untied. The dense pulling score still requires both grasps.

Earlier three-stage configs remain accepted with a deprecation warning. To migrate weights `a` (approach), `g`
(grasp), and `p` (task), set `acquisition_weight=(a+g)/(a+g+p)` and `approach_fraction=a/(a+g)`, and multiply the
manager's overall weight by `a+g+p` to preserve the potential scale. If `a+g` is zero, use zero for
`approach_fraction`. Remove `maximum_progress_rate`, which is now ignored. This preserves stage budgets, not the
old reward trajectory: the acquisition formula and treatment of regressions have changed. The new default
fractions above are independent of that compatibility mapping.

The task uses the assets in `scripts/demos/shoelace/assets`.

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
