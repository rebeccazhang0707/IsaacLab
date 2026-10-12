# Surface-based shoelace grasp MDP

Dual-Franka and G1 Inspire share grasp rewards based on two opposing surface reference points per
hand. Robot-specific configuration supplies four bodies, four body-local offsets and the two ready
apertures. Rewards do not assume a prismatic joint, a particular closing direction or one designated
finger joint. The G1 controls retain two hand binary actions, each moving thumb and index together.

## Geometry and observations

- The capture region is the line segment between the two current surface points. Its distance to
  the three-segment free-tail centre guides approach. A moving midpoint does not reduce capture
  credit when asymmetric closure still brackets the tail.
- All four tail-minus-finger velocity vectors are observed separately in robot root frames.
- Ready-to-closed progress uses actual aperture relative to the ready aperture and cable diameter.
  Remaining outside the capture region removes closure credit; closing an empty hand earns no bonus.
- G1's controller TCP remains the live surface midpoint with wrist orientation. Franka retains its
  existing wrist-offset IK controller. Geometry rewards use its pad-face reference points directly.
- Actor and critic use the same policy observation group. Task progress and loaded-pull history
  remain internal to rewards and success checks.

The surface points approximate the contact patches, and the tail state averages three free segments.
Slip is measured at each reference point against the mean tail velocity, not at resolved contact
points. These quantities do not measure grasp forces or prove physical retention.

## Acquisition and retention

The pregrasp potential has three budgets: capture alignment 0.60, aperture progress 0.25 and soft
contact acquisition 0.15. Contact acquisition uses a 1.5 mm width and
gives each finger independent credit. No-candidate sensor values are excluded; excessive solver
penetration is penalized. Loss of capture/contact removes previously acquired potential, and resets
seed the baseline without credit. Stationary poses pay no repeated pregrasp reward.

Retention, pull eligibility and success share the stricter grasp score. It combines two-finger
contact with a 0.5 mm width, bounded penetration, aperture close to target diameter, tail capture,
reference-point proximity and worst-finger relative speed. Opposing finger velocities cannot cancel.
Relative speed includes all three axes, so ongoing closing motion also reduces the retention score;
the soft acquisition reward remains available while closing. The shared hold filter, hold budget and
loaded-pull requirements remain in effect.

The target diameter defaults to 3 mm, matching the USD cable width. Geometry tolerances are task
parameters in metres: aperture and capture widths default to 2 mm, pad-localization width to 6 mm.
Ready apertures are measured from the imported robot: Franka 19.96 mm per hand, G1 28.81/25.74 mm.
New robots should supply calibrated surface points and measured ready apertures, then pass scripted
grasp/hold/pull checks before learning runs.

## Migration and validation

Actions remain 14-dimensional. Dual-Franka now uses 86 inputs for both actor and critic; G1 uses
112 inputs for both. The runner uses `policy` for both networks. Earlier checkpoints require their
original code/configuration for playback; start fresh policies.

Custom reward configurations replace `open_position`, `closed_position` and reward `robot_cfgs`
with `finger_cfgs`. Set `finger_body_offsets`, `grasp_open_apertures`, `grasp_target_diameter` and
the geometry widths on the environment. Observation `tail_tcp_relative_speed` was replaced in the
policy by `finger_tail_relative_velocity`; `pad_to_tail` is a shared term.
`inspire_pad_to_tail` and `inspire_pregrasp_progress_reward` were consolidated into the common terms.
Metrics now report `capture_distance_m` and `surface_midpoint_distance_m` separately.
The intermediate `tail_direction`, `task_phase_state`, `task_state` group and `tail_tangent_w`
helper were removed, along with `orientation_weight`. Grasp geometry reports `relative_speed`
in place of `tangential_speed`.

Contract tests and finite scene/PPO smoke runs establish consistency of these interfaces. They do
not establish stable physical grasp, successful untying or learning performance; those require
deterministic contact/hold/pull traces and training comparisons.
