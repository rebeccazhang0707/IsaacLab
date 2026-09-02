Added
^^^^^

* Added an optional symmetric ADMM two-way coupling mode to ``IsaacContrib-Shoelace-DualFranka``. Select it with
  ``env.coupling_mode=admm`` and tune its interface solve with ``env.admm_iterations``, ``env.admm_rho``,
  ``env.admm_gamma``, ``env.admm_baumgarte``, and ``env.admm_contact_matching``; the existing lagged-impulse proxy
  coupling remained the default. The ADMM defaults used the fixed-grasp validated penalty, position correction,
  and rigid-contact matching settings.
