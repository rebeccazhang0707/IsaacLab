# Dual-Franka multisolver RL example

`IsaacContrib-Shoelace-DualFranka` is a manager-based task for grasping and pulling the two free ends
of a tied shoelace. It couples **MJWarp** (the Franka arms and shoe) and **VBD** (the cable rods) through
**ADMM**. The physics backend is fixed to Newton; there is no physics preset selector for this task.

The example uses the ADMM contact-capacity controls from
[PR #7912](https://github.com/isaac-sim/IsaacLab/pull/7912) and the VBD rigid-solver controls from
[PR #7913](https://github.com/isaac-sim/IsaacLab/pull/7913). Those changes must be present when applying
this task independently. The VBD configuration uses the updated #7913 interface: only
`rigid_avbd_alpha=0.0` and `rigid_body_contact_buffer_size=256` are overridden; hard contacts retain
Newton 1.6's default. The task was smoke-tested with the repository's Newton 1.6.0 dependency.

## Run

From the repository root, with Git LFS assets fetched. For the current kitless OpenUSD runtime,
apply the repository's [documented single-worker workaround](../../../../../docs/source/refs/issues.rst)
before launching Python; remove it after upgrading to a runtime containing the upstream fix:

```bash
export PXR_WORK_THREAD_LIMIT=1

# Small headless PPO smoke test; increase batch size and iterations for training.
uv run isaaclab train --rl_library rsl_rl --task IsaacContrib-Shoelace-DualFranka \
  --num_envs 4 --max_iterations 1 --visualizer none

# Interactive training with the Newton OpenGL visualizer.
uv run isaaclab train --rl_library rsl_rl --task IsaacContrib-Shoelace-DualFranka \
  --num_envs 4 --visualizer newton_gl

# Inspect a saved policy.
uv run isaaclab play --rl_library rsl_rl --task IsaacContrib-Shoelace-DualFranka \
  --num_envs 1 --checkpoint /path/to/model.pt --visualizer newton_gl
```

CUDA is required by the coupled MJWarp/VBD simulation. Newton GL is the documented visualizer for
this example; its environment-browser preview enables the viewer's collision-geometry display.
Enable **Show Collision** in the viewer to inspect the shoe collider and the coupled cable scene.
Headless training needs no visualizer.
The smoke test verifies rollout and PPO updates; it does not establish policy convergence or a success rate.

## Module guide

| Module | Responsibility |
| --- | --- |
| `shoelace_env_cfg.py` | Standard manager-based scene, solver composition, managers, and dependent configuration resolution |
| `shoelace_assets.py`, `shoelace_constants.py` | USD material overrides, common spawners, asset paths, and physical defaults |
| `shoelace_contacts.py` | Scene sensor for signed finger-tail distances, including selective reset and CUDA capture |
| `generate_asset.py`, `asset_authoring.py` | Offline generation of the composite shoe/cable USD and its physics properties |
| `mdp/grasp.py` | Shared contact/closure/slip quality, filtering, and bilateral scoring |
| `mdp/observations.py`, `mdp/events.py` | Policy inputs, startup settled defaults, and reset randomization |
| `mdp/rewards.py`, `mdp/terminations.py` | Independent manager-term histories for progress, retention, and success |
| `agents/rsl_rl_ppo_cfg.py` | PPO runner configuration |
| `data/` | Baked composite USD, source assets, textures, and offline settled poses; see attribution there |

The Gym entry point is `isaaclab.envs:ManagerBasedRLEnv`. The default task has no
model-builder callback. The Newton importer reads asset-local physics properties
before replication; `EventTermCfg` handles settled defaults and randomized resets.
`FingerTailContactSensor` initializes before solver CUDA capture and clears its cache
through the normal scene sensor reset. Robot gravity compensation uses `MujocoRigidBodyCfg`.

To regenerate the asset after changing offline geometry or material calculations:

```bash
PXR_WORK_THREAD_LIMIT=1 uv run python -m isaaclab_tasks.contrib.shoelace.generate_asset
```

`data/shoelace.usda` uses native segment masses (which also determine geometric inertia),
material/contact attributes, and element collision filters. `NewtonCablePropertiesCfg` extends
the existing `SchemaFragment` interface for fixed endpoints, proxy inertia, initial frames,
tail stiffness/damping, and Dahl parameters that preserve the special model behavior. Its paths
and segment indices are internal to the asset. Runtime `finger_mu`, `lace_mu`, `shoe_mu`,
and `cable_inertia_regularization` overrides remain available; they modify USD properties
before import. Contact capacity and ground size resolve from the final `scene.num_envs`
in `validate_config()`, including CLI overrides.

The Isaac Lab USD extension and supported units/topologies are documented in
[Authoring Newton cable assets](../../../../../docs/source/how-to/newton_cable_assets.rst).

Grasp calculations are shared, but reward and termination filters remain independent. Disabling a reward
for an ablation cannot change the success history. The task configures the framework's `VBDSolverCfg`
directly.

## Task contract

The action vector has 14 components, in order: left relative TCP pose (6), left binary gripper (1),
right relative TCP pose (6), and right binary gripper (1). Cartesian action scales are 0.005 m and
0.01 rad. Arm-action penalties select the arm terms by name and exclude gripper commands.

The 64-dimensional policy observation contains both arms' relative joint positions and velocities,
positive gripper closure residuals, tail-to-TCP vectors, three policy steps of signed finger-tail
distance history, tail-to-TCP relative speeds, and the previous action. Tail quantities use robot-arm
order: each robot controls the opposite cable's free end. Contact distance is positive for separation,
negative for penetration, and capped at 0.002 m when no candidate is present.

The policy runs at 30 Hz (`dt=1/120`, `decimation=4`), with ten Newton substeps per physics step and
collision detection every second substep. Episodes last at most 10 s. Reset restores settled cable
poses, perturbs arm joints by up to 0.02 rad, synchronizes their targets, and translates the shoe and
both laces together by up to 0.02 m in X and Y. Partial resets clear only the selected environments'
contact and manager histories.

| Reward | Weight | Behavior |
| --- | ---: | --- |
| Dense task progress | 10 | Acquisition potential changes plus new outward pull records gated by grasp quality |
| Pregrasp progress | 1 | Alignment potential changes and closure credit near a tail |
| Grasp retention | 1 | Filtered grasp quality; full-rate budget of 2 quality-weighted seconds, then 20% rate |
| Success | 5 | One completion bonus |
| Arm action changes | -0.001 | Squared policy-step arm-command changes |
| Arm action magnitude | -0.001 | Squared arm commands |

Progress potentials seed from the first valid sample without reset credit. Pull records advance even
without a grasp, so regrasping cannot earn credit for passive motion or repeating an old excursion.
Invalid samples earn no progress and preserve valid reward history. The grasp estimate combines
both-finger signed distances, actual closure, and low slip; it is not a force-closure measurement.

Success requires both arms to accumulate at least 0.025 m of new outward motion while both raw and
filtered grasps exceed 0.2 at both ends of each credited interval. Then each tail must reach its own
0.09 m outward X boundary, and each cable must leave at most 15 free segments within the 0.025 m
knot-throat radius. One arm cannot compensate for the other. Completed loaded pulls remain recorded
after release; current geometry must still satisfy the completion condition.

`mdp.shoelace_bilateral_pull_success` uses the grasp and geometry parameters in
`TerminationsCfg.success`. The success reward reads this termination result without recomputing it.

The outer collision pipeline scales its capacity with environment count. The ADMM pipeline retains
its small triangle budget for Newton 1.6 contact matching and scales its reduction table instead.
Existing explicit capacity increases are retained. These are public coupler settings; task code does
not modify a coupler's private collision pipeline.

## Validation

```bash
uv run --extra test python -m pytest -q \
  source/isaaclab_tasks/test/contrib/test_shoelace_rewards.py \
  source/isaaclab_tasks/test/contrib/test_shoelace_contact_observations.py \
  source/isaaclab_tasks/test/contrib/test_shoelace_physics.py
```

The focused tests cover reward cycles and record gating, invalid samples, cooperative success,
contact aggregation, action/observation contracts, proxy inertia, capacity overrides, and randomized
partial resets with CUDA-captured coupled steps. The reset/step test skips when CUDA is unavailable.
