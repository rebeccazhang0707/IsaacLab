* **Breaking:** Removed the deprecated ``maximum_progress_rate``, ``approach_weight``, ``grasp_weight``,
  and ``task_weight`` parameters from the shoelace dense reward, along with their compatibility logic.
  For old budgets ``a``, ``g``, and ``p``, set ``acquisition_weight=(a+g)/(a+g+p)`` and
  ``approach_fraction=a/(a+g)`` (zero if ``a+g=0``), and multiply the manager term weight by ``a+g+p``.
  Omitted old budgets defaulted to ``0.15``, ``0.35``, and ``1.0``, respectively. Delete all four old
  parameters, and pass ``cable_cfgs`` and ``robot_cfgs`` by keyword in direct calls. The ignored rate
  limit had no replacement. Preserved the reward formula and scale for current task configurations.
