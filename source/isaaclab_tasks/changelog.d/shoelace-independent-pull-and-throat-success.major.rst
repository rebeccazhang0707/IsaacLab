Changed
^^^^^^^

* Changed shoelace pulling progress to use each tail's own outward displacement and grasp quality, with
  a smaller bilateral bonus. Removed the hard reset-baseline dead zone and retained signed potential
  differences and fixed episode references across regrasping. Re-evaluate or retrain policies and use
  ``bilateral_pull_fraction`` to adjust the collaboration bonus; migrate deprecated parameters below.
* Changed the default shoelace success term to require knot-throat clearance, both tails away from the
  fixed anchors, finite cable state, and X separation. Kept the distance-only helper available; migrate
  custom task configurations to ``shoelace_success`` using the thresholds in ``TerminationsCfg.success``.
  Revalidate throat thresholds when changing cable geometry or segment counts, and do not compare the
  previous distance-only success rates directly with the new rates.
* Reduced shoelace phase logging to nine metrics, replacing aggregate pull progress/score with per-arm
  pull scores and removing ``approach_score`` and ``grasp_slip_mps``. Update dashboards to use physical
  approach distance, per-arm grasp/pull scores, simultaneous grasp quality, actual separation, success
  rate, and valid-input fraction; these metrics are diagnostic replacements, not numerical aliases.

Removed
^^^^^^^

* **Breaking:** Removed the deprecated ``maximum_progress_rate``, ``approach_weight``, ``grasp_weight``,
  and ``task_weight`` parameters from the shoelace dense reward, along with their compatibility logic.
  For old budgets ``a``, ``g``, and ``p``, set ``acquisition_weight=(a+g)/(a+g+p)`` and
  ``approach_fraction=a/(a+g)`` (zero if ``a+g=0``), and multiply the manager term weight by ``a+g+p``.
  Omitted old budgets defaulted to ``0.15``, ``0.35``, and ``1.0``, respectively. Delete all four old
  parameters, and pass ``cable_cfgs`` and ``robot_cfgs`` by keyword in direct calls. The ignored rate
  limit had no replacement. Preserved the reward formula and scale for current task configurations.
