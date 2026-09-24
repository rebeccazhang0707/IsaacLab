:orphan:

.. Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

Authoring Newton cable assets
=============================

Keep the cable centerline in a linear ``UsdGeom.BasisCurves`` with the usual cable
deformable APIs and material binding. Use native ``physics:masses`` with
``physics:masses:elementType = "segment"`` for positive segment masses, and
``PhysicsElementCollisionFilter`` relationships for collision exclusions. A filter
must be imported together with both of its source prims; package interacting cables
and fixtures beneath one asset root when they share filters.

Native materials and contacts
--------------------------------

Use ``CableCfg`` and ``CableMaterialCfg`` for native curve geometry and uniform
material properties. Compose ``UsdPhysicsRigidBodyMaterialCfg`` and ``NewtonMaterialCfg``
on the bound material for friction and contact response, and use ``NewtonCollisionCfg``
for contact gaps. These configurations author existing USD/Newton attributes.
Native ``physics:masses`` also updates each segment's geometric inertia.

The generic adapter leaves contact import to Newton. The installed Newton 1.6.0
curve importer does not transfer the bound material's ``physics:dynamicFriction``,
``newton:contactStiffness``, ``newton:contactDamping``, or the curve's
``newton:contactGap`` into generated capsule shapes. Until the dependency supports
these fields natively, configure them in a task startup event and notify
``ModelFlags.SHAPE_PROPERTIES``. This synchronizes compact solver views; the collision
pipeline reads ``model.shape_gap`` when computing bounds and generating contacts.
The shoelace event implements this fallback without a generic USD-import patch.

Construction-time Dahl parameters
---------------------------------

``NewtonCablePropertiesCfg`` uses the existing ``SchemaFragment`` interface to author
Dahl arrays. These attributes belong to **Isaac Lab**, not the upstream Newton schema.
Native curve import does not populate the generated joints' Dahl attributes, and VBD
decides whether to enable Dahl when the solver is constructed. Ordinary startup events
therefore cannot first enable Dahl on a solver constructed with all-zero parameters.
Only this construction-time bridge remains in ``isaaclab_newton.sim.usd``.

.. code-block:: python

   from isaaclab_newton.sim.schemas import NewtonCablePropertiesCfg, apply_newton_cable_properties

   # A standalone open curve with three segments has two rod joints.
   properties = NewtonCablePropertiesCfg(dahl_max_strains=[0.2, 0.3], dahl_decay=[0.4, 0.5])
   apply_newton_cable_properties(properties, "/World/Cable/geometry/mesh", stage)

``None`` leaves an existing authored value unchanged. The optional attributes are:

* ``isaaclab:cable:dahlMaxStrains`` (``float[]``): per-joint VBD maximum persistent
  angular strain [rad], mapped to ``vbd:dahl_eps_max``.
* ``isaaclab:cable:dahlDecay`` (``float[]``): per-joint VBD angular memory decay
  length [rad], mapped to ``vbd:dahl_tau``.

The arrays support one standalone open curve with ``N`` segments and ``N - 1`` rod
joints. Entry ``i`` connects segments ``i`` and ``i + 1`` in curve point order.
Do not store finalized or replicated model indices. Both values must be positive
on a joint to enable its Dahl friction; missing values retain the registered defaults.

Author arrays in a stage with ``metersPerUnit = 1`` and bake scale/shear into geometry.
Translation and proper rotation are supported. Incorrect lengths, nonfinite values,
and negative values raise ``ValueError``. Closed, multi-curve, and welded cable arrays
are not supported by this extension.

Task-specific physics belongs in events
------------------------------------------

Use existing manager-based ``startup`` events for one-time fixed segments/fixtures,
inertia regularization, model frames, and per-joint stiffness/damping. Episode
``reset`` events should restore state rather than repeatedly modify this configuration.
After editing Newton model arrays, send the corresponding model-change flags through
the existing ``NewtonManager`` notification path to refresh coupled solvers and caches.

These settings can differ from native USD authoring:

* Zeroing segment mass/inertia keeps topology unchanged, unlike adding attachments.
* Graded per-joint gains need not match uniform material moduli and local-length scaling.
* Model-frame overrides can preserve imported joint rest frames; native curve normals
  define both body and joint rest frames together.
* Newton finalization can correct tiny inertias. Recompute geometric inertia before
  adding regularization if the task needs the original geometric-plus-regularization model.

Migration from former custom USD overrides
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The importer no longer consumes ``isaaclab:cable:fixedSegments``,
``inertiaRegularization``, ``segmentOrientations``, ``jointStiffnesses``, or
``jointDampings``, nor ``isaaclab:physics:fixed`` on rigid bodies. Remove those attributes
from old assets and configure their behavior in startup events. Existing files containing
them import with native Newton behavior; they do not retain those custom overrides.

The corresponding non-Dahl ``NewtonCablePropertiesCfg`` fields are deprecated and
retained only to produce a migration error when passed to the applier, rather than to
author ignored attributes. Keep using its two Dahl fields until native import supports them.

Shoelace example
----------------

The ``IsaacContrib-Shoelace-DualFranka`` example includes a baked
``data/shoelace.usda`` and an offline authoring tool:

.. code-block:: bash

   PXR_WORK_THREAD_LIMIT=1 uv run python -m isaaclab_tasks.contrib.shoelace.generate_asset

The task uses ``UsdFileCfg``-based spawning and ``CableObjectCfg(spawn=None)`` views
on the two cable children. The generator authors native masses, materials, contact
properties, and collision filters. Only Dahl parameters remain in custom cable USD
attributes. Finger/shoe friction uses spawner material configuration. Cable friction,
contact stiffness/damping, and contact gap are applied to generated shapes at startup
from the task's settings, because the installed Newton importer omits those fields.

The ``configure_shoelace_physics`` startup event updates all environments: fixed cable
segments and the shoe receive zero mass/inertia, dynamic segments receive geometric
inertia plus ``cfg.cable_inertia_regularization``, and joints receive the graded tail
stiffness/damping profile. It restores parallel-transport model frames without changing
joint rest frames. Coupler notifications refresh solver data and VBD rest invariants.

The subsequent ``install_settled_default_state`` startup event installs settled poses
and zero velocities. Resets restore and randomize state, not physics parameters.
Keep these startup terms in this order. Set ``ShoelaceEnvCfg.cable_inertia_regularization``;
the deprecated ``ShoelaceUsdCfg.inertia_regularization`` now raises a migration error.
No new callback or ``EventManager`` change is needed; ``prestartup`` is not supported
with ``replicate_physics=True``.

A scene sensor handles finger-tail observations. Buffer capacities stay in the task's
simulation configuration and are resolved after CLI overrides.
