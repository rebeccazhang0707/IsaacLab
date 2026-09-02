Added
^^^^^

* Added an optional symmetric ADMM two-way coupling mode to ``IsaacContrib-Shoelace-DualFranka``. Select it with
  ``env.coupling_mode=admm`` and tune its interface solve with ``env.admm_iterations`` and ``env.admm_rho``; the
  existing lagged-impulse proxy coupling remained the default.
