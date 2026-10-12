# G1 Inspire shoelace scene

The task `IsaacContrib-Shoelace-G1Inspire` uses Isaac Lab's `G1_INSPIRE_FTP_CFG` with a fixed pelvis,
two Cartesian arm controllers, and one synchronous binary thumb/index command for each hand.
The shoe and pedestal are kinematic. This stage isolates manipulation from balance and walking.

## Scene and controls

- The pedestal is 78 cm high, with a 6×6 cm column and an 8×26 cm top. The sole overhangs the top
  so the hands can approach the laces from the sides without a wide table underneath them.
- The robot pelvis is 79.5 cm above the ground, 38 cm behind the shoe, facing it.
- Initial arm angles were measured with IK over the nominal baked free tails. The configured TCPs
  start approximately 6 cm above those tails. Reset randomization changes the actual clearance.
- The 14 actions, in order, are left relative TCP pose (6), left thumb/index open-close (1),
  right relative TCP pose (6), and right thumb/index open-close (1). Each hand action moves both
  finger chains synchronously; the left and right hands remain independently controlled.
  Negative finger actions close; zero and positive finger actions open.
- Middle, ring and little finger targets remain zero (straight). Contact can still deflect those joints.
  Thumb yaw stays opposed in both open and closed states. Active fingers open to a curved ready pose
  rather than sweeping all the way from straight fingers on every closure.
- Target changes are limited to 1 rad/s at the control timestep. One interpolation fraction per hand
  preserves the authored mimic ratios throughout the movement: index intermediate = proximal,
  thumb intermediate = 1.6 × primary pitch, thumb distal = 2.4 × primary pitch.
  This bounds commanded targets, not measured joint velocity under contact.
- Keep MJWarp's `implicitfast` integrator. Euler became nonfinite in the isolated stock-drive probe.
  Cable dynamics and coupling retain the Franka task's VBD and ADMM settings.

## Nominal pinch calibration

Closed targets were fitted using the imported collision meshes, joint limits and native mimic ratios
around a 2.8 mm diameter target. Forward kinematics were checked against Newton body poses.
These are geometric candidate poses; fitting a small target does not establish physical retention.

| Primary joint [rad] | Left open | Left close | Right open | Right close |
| --- | --- | --- | --- | --- |
| Index proximal | 0.602548 | 0.802548 | 0.653050 | 0.853050 |
| Thumb yaw | 1.191681 | 1.191681 | 1.246639 | 1.246639 |
| Thumb pitch | 0.215063 | 0.265063 | 0.212787 | 0.262787 |

Each TCP position is the live world-space midpoint of two approximate contact-surface centres
near the thumb and index tips. The points are transformed from the thumb distal and index
intermediate body frames at every read. Their local offsets, in metres, are:

| Hand | Thumb distal surface point | Index intermediate surface point |
| --- | --- | --- |
| Left | `(0.019816, -0.013389, -0.004498)` | `(-0.005060, -0.037262, -0.003575)` |
| Right | `(0.014990, 0.014964, -0.007629)` | `(-0.005540, 0.035395, -0.006249)` |

`TCP = (thumb_surface_world + index_surface_world) / 2`. These points approximate the contact
surfaces near the tips; they are distinct from the asset's decorative fingertip spheres.
TCP orientation follows the wrist. Arm IK uses the average linear Jacobian of the two surface
points and the wrist angular Jacobian. Rewards measure capture between these same surface points
and use each point's velocity separately, including the finger body's linear velocity and
angular-velocity cross local-offset contribution.

In the measured ready-to-closed motion, the TCP shifted approximately 9.34 mm in the left wrist
frame and 9.55 mm in the right wrist frame. A fixed wrist offset would miss this movement.

The 112 actor and critic observations include 12 thumb/index joint positions, 12 joint velocities, four
normalized target residuals, and four three-dimensional landmark-to-tail vectors, alongside the
existing arm, cable, contact and action observations. They include twelve separate finger-relative
velocity components. Both networks use the same policy observation group.

Franka and Inspire now share the surface-geometry MDP described in [GRASP_MDP.md](GRASP_MDP.md).
Pregrasp uses the capture region between surfaces, aperture progress and
soft first-finger contact. Retention and success require strict two-finger contact, local capture and
low worst-finger relative speed. Joint angles alone do not establish grasp quality.

## Run and checkpoint compatibility

Run from the repository's existing uv environment:

```bash
uv run --no-sync isaaclab train --rl_library rsl_rl --task IsaacContrib-Shoelace-G1Inspire \
  --num_envs 64 --max_iterations 3 --logger tensorboard env.episode_length_s=0.2
```

The short episode override exercises batch resets during a smoke test. For learning runs, omit it
to retain the normal 10-second episodes. The configured Newton GL camera frames the robot, shoe
and pedestal. Add `--visualizer newton_gl --max_visible_envs 1` to inspect a run.

The previous G1 policy, including `model_7850.pt`, used 14 actions and 64 observations and cannot
resume with this configuration. Replay it using the previous task revision (`7ac982f0b085`) and
start a fresh policy for this action/observation contract. The intermediate four-switch variant
used 16 actions and 104 observations; the first live-TCP version used 14 actions and 102 observations.
Both are incompatible with the current actor/critic inputs of 112/112. The intermediate 118/124
version with tail direction and critic task state is also incompatible. Franka checkpoints use different
joint and TCP semantics.

The shoe assets must exist in `data/` beside this file, including `shoelace.usda` and its variants,
textures and referenced shoe geometry. This checkout's global ignore rules exclude USD text assets.
For local validation, the existing assets were copied from the sibling
`IsaacLab_shoelace_rl_example` worktree; the G1 USD remains the standard Isaac Lab asset.

## Validation status

Paired-action tests cover rate limits, native mimic ratios and isolated partial resets. Shared MDP
contracts cover geometry without joint-state assumptions, uncancelled opposing finger slip,
capture-preserving asymmetric closure, first-finger contact and success history across partial resets.

Earlier four-environment, 96-control-step probes validated the shared surface landmarks and measured
ready apertures with frequent batch resets. G1's ready surface apertures measured 28.81 and 25.74 mm;
Franka's measured 19.96 mm per hand. These surface definitions and ready poses remain unchanged.
Both also completed 64-environment, three-update PPO smoke runs with finite recorded scalars and
model/optimizer tensors and frequent episode resets.

The moving G1 TCP Jacobians were previously checked against finite differences, with maximum
position derivative errors of 0.000168 and 0.000127 metres/radian. Its point definition and arm IK
remain unchanged. Earlier policies used different reward/observation contracts and cannot establish
learning performance for this version. Stable bilateral grasp, hold and pull still require physical
validation before another long untying run. Successful untying and free-base balance are unvalidated.
