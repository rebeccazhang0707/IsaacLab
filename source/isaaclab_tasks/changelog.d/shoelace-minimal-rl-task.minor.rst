Added
^^^^^

* Added a minimal trainable Newton dual-Franka shoelace task with deterministic resets, task-local finger-tail
  contact observations, a two-phase acquisition-and-pull reward with Hamacher soft-AND, and an X-separation
  success condition.
* Replaced separately clipped phase rewards with one signed potential difference, including contact filtering
  and per-environment reset state. Deprecated the earlier three-stage weight parameters in favor of
  ``acquisition_weight`` and ``approach_fraction``; documented the budget conversion in the task README.
  Accepted but ignored the deprecated ``maximum_progress_rate`` parameter.
