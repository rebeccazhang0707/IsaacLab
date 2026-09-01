# Shoelace 4096-Environment Performance Summary

## Results

All results used a headless simulation with 4096 environments. FPS measures global simulation frames per second; higher is better.

| Version | Configuration | FPS | Observed delta vs. previous | Environment steps/s |
|---|---|---:|---:|---:|
| Single cable | Baseline implementation | 0.5200 | — | 2,130 |
| Two cables | Simulate only the dynamic left and right portions | 0.6030 | +16.0% | 2,470 |
| Two cables, localized contact | Visual-only static span, two localized collision meshes, no contact history, `TRIANGLE_PAIRS_PER_ENV=8192`, `CONTACT_BUFFER=128` | 0.9842 | +63.2% | 4,031 |
| Two cables, shared full collider | One visible and collidable six-sided static tube, no contact history, `TRIANGLE_PAIRS_PER_ENV=8192`, `CONTACT_BUFFER=128` | 0.9642 | -2.0% | 3,949 |

## Observed configuration delta

- Single-cable result to two-cable result: `0.5200 → 0.6030 FPS`, a **16.0%** increase.
- Two-cable result to localized-contact result: `0.6030 → 0.9842 FPS`, a **63.2%** increase.
- Localized contact to shared full collider: `0.9842 → 0.9642 FPS`, a **2.0%** decrease in FPS and a **2.1%** increase in time per frame.
- Overall observed change: `0.5200 → 0.9642 FPS`, an **85.4%** increase in FPS and a **46.1%** reduction in time per frame.

## Measurement notes

- The latest raw result was `0.964163 FPS` and `3,949.211 environment steps/s` over 120 frames. It excluded USD loading, model construction, solver initialization, and CUDA graph capture.
- The latest command was `uv run python scripts/demos/newton_shoelace.py --num_envs 4096 --visualizer none --max_steps 120`; `TRIANGLE_PAIRS_PER_ENV=8192` and `CONTACT_BUFFER=128` are now fixed script configuration.
- The localized-contact baseline used the same command, timing exclusions, and 120-frame window and measured `0.984205 FPS`. The shared full collider contains 3,036 collision triangles per environment instead of 352, but the measured FPS cost was only 2.0% and no buffer overflow warning occurred.
- Both controlled collision-mesh runs were single measurements on an NVIDIA RTX 5880 Ada Generation GPU; no repeat variance is available.
- An earlier optimized run measured `0.969708 FPS`. Its contact history was already automatically disabled at 4096 environments, so the later `0.984205 FPS` result was a remeasurement, not a separate contact-history gain.
- The 30-frame light-contact result of `3.188 FPS` is excluded because it did not cover the more expensive tightening/contact phase.
- The first two values are retained from earlier tests and do not have the same detailed timing metadata. These runs are not a strict one-variable ablation because solver settings, anchor ranges, and static-contact geometry also changed during development; no individual change, including contact-history removal, receives a causal performance attribution.
