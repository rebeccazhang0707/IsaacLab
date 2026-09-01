Fixed
^^^^^

* Fixed finger colliders penetrating the shoe by solving robot-shoe contacts directly in MJWarp while retaining
  cable-shoe contacts through the VBD proxy coupling.

Changed
^^^^^^^

* Removed redundant runtime contact-history configuration from the dual-Franka shoelace task. Workflows that need
  persistent VBD contacts must now enable rigid-contact history and collision contact matching explicitly.
