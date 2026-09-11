Changed
~~~~~~~

* Randomized dual-Franka shoelace episode starts with arm joint offsets of up to 0.02 rad and shared shoe/lace
  X/Y translations of up to 0.02 m. Kept grippers open, synchronized arm position targets, and preserved cable
  anchor geometry. To restore deterministic starts, set ``reset_left_arm``, ``reset_right_arm``, and
  ``reset_shoe`` to ``None`` in the event configuration while retaining ``reset_scene``.
* Made ``shoe_asset_cfg()`` return a resettable kinematic ``RigidObjectCfg`` and moved the pinned lace mesh under
  the shoe. Custom prim-path lookups must use ``{ENV_REGEX_NS}/Shoe/ShoelacePinned`` instead of
  ``{ENV_REGEX_NS}/ShoelacePinned``; the scene entity name stayed unchanged.
