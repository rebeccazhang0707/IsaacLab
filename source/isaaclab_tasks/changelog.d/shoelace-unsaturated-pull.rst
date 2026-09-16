Fixed
^^^^^

* Removed premature saturation of dual-Franka shoelace high-water pull rewards at the per-arm
  separation scale. New outward records continued earning grasp-qualified independent credit,
  and bilateral credit extended with the trailing arm's progress. Below-scale shaping, grasp
  gates, passive-motion exclusion, cycle protection, and success conditions remained unchanged.
  Existing checkpoints remained loadable but require re-evaluation and further training under
  the extended objective; use the original code revision to reproduce capped-pull reward curves.
