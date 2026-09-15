Added
^^^^^

* Added ``Metrics/shoelace/pull_left_displacement_m`` and ``pull_right_displacement_m`` for signed
  outward tail displacement from each episode's first valid reward sample [m], independently of grasp
  quality. Added explicit ``pull_left_score`` and ``pull_right_score`` names for the existing
  grasp-gated scores. Reward formulas and weights were left unchanged.

Deprecated
^^^^^^^^^^

* Deprecated the ambiguous ``Metrics/shoelace/pull_left`` and ``pull_right`` names while retaining
  their values as score aliases. Dashboards should use the new displacement tags for physical progress
  or the explicit score tags for reward diagnostics; historical score curves are not distances and
  cannot be compared numerically with the new displacement curves.
