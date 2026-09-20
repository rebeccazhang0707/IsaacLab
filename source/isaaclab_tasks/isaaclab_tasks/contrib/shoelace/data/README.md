# Shoelace task assets

`collider_simplified.usd` is a collision-only derivative of `collider.usd`. Its closed shoe shell was reduced
from 28,106 to 8,000 triangles with quadric decimation while preserving the original scale, winding, and prim
path. The source collider comes from
[`newton/examples/assets/shoelace`](https://github.com/newton-physics/newton/tree/7daf8324dbe1ca6301a3440f4b1204e3158c2369/newton/examples/assets/shoelace)
at Newton commit `7daf8324dbe1ca6301a3440f4b1204e3158c2369`. The visual shoe remains sourced from Newton's
`model.usd`.

The source Newton repository distributes these files under the
[Apache License 2.0](https://github.com/newton-physics/newton/blob/7daf8324dbe1ca6301a3440f4b1204e3158c2369/LICENSE.md).

A copy of the upstream license is included in [LICENSE.md](LICENSE.md).
The USD files, textures, and settled segment poses are bundled as package data so the task also works
from an installed wheel. Fetch Git LFS assets before building a source checkout.

`settled_tail_clear_segment_poses.npz` contains the task's offline gravity-settled initial poses.
It stores 79 segment poses per free cable, in local coordinates with quaternion order x-y-z-w.
Reset translates these poses together with the shoe and preserves their fixed anchors.
