* **Breaking:** Made G1 shoelace finger targets identical across hands and mirrored the collision pads,
  surface landmarks, wrist TCPs, and initial arm poses. Compensated the source USD's asymmetric hand
  joint frames when expressing the right pads in body coordinates. Right-arm resets now mirror the
  left arm's random perturbation. Observation and action dimensions were preserved, but existing G1
  checkpoints require retraining for the changed geometry and reset distribution; use the previous
  task revision to reproduce their original evaluations.
