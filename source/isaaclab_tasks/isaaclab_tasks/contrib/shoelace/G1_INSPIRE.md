# G1 Inspire shoelace scene

The task `IsaacContrib-Shoelace-G1Inspire` uses a fixed pelvis, two Cartesian arm controllers,
and one binary index-finger command per hand. The shoe is kinematic.

## Symmetry contract

Left and right hands use the same finger targets. The physical pinch boxes and their surface
landmarks mirror across the wrist XZ plane (negate Y), including box orientation. The right TCP
is the same reflection of the left TCP.

The source G1 USD is not an exact mirror: its thumb joint frames and index link offsets differ.
Consequently, the right pads need different body-local offsets and rotations to produce mirrored
geometry in the wrist frame. Simply negating the Y coordinate of each body-local offset would
not achieve this. Visual meshes, source joint frames, masses and inertias are unchanged, so the
visible finger meshes retain the source asset's small asymmetry.

Earlier closures were fitted independently for each hand without a mirror constraint. This
configuration uses the left closure for both hands and compensates the right pad transforms.

## Scene and controls

- A visual-only PackingTable workstation is uniformly scaled to a 78 cm tabletop, with its
  tray moved to the right rear to leave the shoe and hands clear. It uses the same asset as
  the G1 pick-place task. Imported table, crate and tray physics are disabled.
- The shoe remains at 78 cm, directly above the tabletop, with about 1 mm sole clearance.
  The previous shoe fixture and support column, including their collisions, are removed.
  The shoe is kinematic and therefore needs no physical support. The decorative table has
  no collisions; robot poses, shoe resets, actions, rewards and solver settings are unchanged.
- Set `scene.packing_table=None` to omit the decorative workstation.
- The pelvis starts 79.5 cm above the ground and 38 cm behind the shoe.
- The 14 actions are left relative TCP pose (6), left open-close (1), right relative TCP pose (6),
  and right open-close (1). Negative finger actions close; zero and positive actions open.
- Thumb yaw and pitch stay fixed. Only the index proximal joint moves during closure.
- Downstream thumb and index mimic constraints are removed, and those joints are held straight
  with revolute limits of +/-0.001 degrees. Middle, ring and little finger targets remain zero.
- Target changes are limited to 1 rad/s. This bounds commanded targets, not measured joint velocity.
- Newton uses the MJWarp `implicitfast` integrator and the shared cable coupling configuration.

| Joint target [rad], either hand | Open | Closed |
| --- | --- | --- |
| Index proximal | 0.975962774 | 1.225962774 |
| Thumb yaw | 1.233697987 | 1.233697987 |
| Thumb pitch | 0.4999 | 0.4999 |

Terminal thumb/index mesh colliders are replaced by invisible 16 by 10 by 3 mm boxes.
Their nominal surface separation is 19.776 mm open and 2.8 mm closed. These dimensions describe
geometry; they do not establish physical retention of a cable.

Arm IK uses fixed wrist-relative TCP offsets, in metres:

- Left: `(0.193129098, -0.063929696, 0.029342429)`.
- Right: `(0.193129098, 0.063929696, 0.029342429)`.

Each TCP coincides with the closed pad-face midpoint. Rewards use the live pad-surface landmarks
and their velocities, including finger motion. See [GRASP_MDP.md](GRASP_MDP.md).

## Initial state and resets

The left nominal arm pose came from an IK fit above the nominal left free tail. Right joint angles
mirror that pose: shoulder pitch, elbow and wrist pitch retain their signs; shoulder roll/yaw and
wrist roll/yaw change signs. The source shoulder offsets leave approximately 0.01 mm of spatial
mirror error. The baked rope tails are asymmetric, so mirrored hands need not have equal clearance
above their respective tails.

Each reset samples the existing +/-0.02 rad left-arm noise, then mirrors left positions, velocities
and position targets into the right arm. Separate environments retain independent samples.
Partial resets leave the other environments untouched. Shoe randomization is unchanged.

## Checkpoints and validation

The action count remains 14 and actor/critic observations remain 112/112. Nevertheless, the new
finger targets, collision geometry, TCP and reset distribution change the policy's environment.
Train a new policy for this configuration; reproduce earlier evaluations using their original
source revision. Older 64-, 102-, 104- and 118/124-observation checkpoints are also incompatible.

The existing USD spawner regression checks both open and closed poses using independent forward
kinematics: joint targets, initial wrist poses, actual box transforms, surface landmarks, apertures
and TCPs. Runtime validation additionally checks mirrored randomized resets, partial resets and
actual Newton pad poses throughout an unloaded open-close-open sequence.

Those checks establish symmetry and reset behavior. Bilateral capture, retention and pulling still
need physical validation before another long training run. See [README.md](README.md) for task assets
and launch configuration.
