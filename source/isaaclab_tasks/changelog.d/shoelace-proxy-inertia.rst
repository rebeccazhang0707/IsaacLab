Added
^^^^^

* Added explicit ``ShoelaceEnvCfg.cable_inertia_regularization`` [kg*m^2] for dynamic shoelace
  segments, with a default isotropic addition of ``1e-6`` matching the previous effective-inertia scale.
  Left masses, pinned anchors and robot inertias unchanged. Documented the matching standalone-demo
  option and the need to revalidate settling and grasping when changing the proxy inertia.
