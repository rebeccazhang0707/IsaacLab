Removed
^^^^^^^

* **Breaking:** Removed generic import of ``isaaclab:cable:fixedSegments``,
  ``inertiaRegularization``, ``segmentOrientations``, ``jointStiffnesses``,
  ``jointDampings``, and ``isaaclab:physics:fixed``. Move these model overrides to
  manager-based startup events and notify the corresponding Newton model changes.
  Retained the non-Dahl ``NewtonCablePropertiesCfg`` fields only for explicit migration
  errors in the applier. Dahl arrays remained supported before solver construction.
* **Breaking:** Removed the cable contact-material/gap import patch. Generic import now
  follows Newton's native behavior. With Newton 1.6.0, configure generated cable shapes'
  friction, contact stiffness/damping, and gap in startup events and notify
  ``ModelFlags.SHAPE_PROPERTIES``; the shoelace task provided this fallback.
