Removed
^^^^^^^

* **Breaking:** Removed the deprecated ``Metrics/shoelace/pull_left`` and ``pull_right`` aliases from
  shoelace training logs. Update dashboards to ``pull_left_score`` and ``pull_right_score`` for the
  unchanged diagnostic scores, or use ``pull_left_displacement_m`` and ``pull_right_displacement_m``
  for physical progress. Reward calculations were unchanged. Existing event files were preserved;
  restart training processes to use the new metric set.
