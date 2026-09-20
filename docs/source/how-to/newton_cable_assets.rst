:orphan:

.. Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

Authoring Newton cable assets
=============================

Isaac Lab's Newton backend can import preconfigured cable physics from USD before
replicating the source asset. This allows a standard ``ManagerBasedRLEnv`` to load
an asset without a task-specific ``MODEL_INIT`` callback.

Keep the cable centerline in a linear ``UsdGeom.BasisCurves`` with the usual cable
deformable APIs and material binding. Use native ``physics:masses`` with
``physics:masses:elementType = "segment"`` for positive segment masses, and
``PhysicsElementCollisionFilter`` relationships for collision exclusions. A filter
must be imported together with both of its source prims; package interacting cables
and fixtures beneath one asset root when they share filters.

Additional cable properties
---------------------------

Reuse existing configurations first
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``CableCfg`` and ``CableMaterialCfg`` for native curve geometry and uniform
material properties. Compose ``UsdPhysicsRigidBodyMaterialCfg`` and ``NewtonMaterialCfg``
on the bound material for friction and contact response, and use ``NewtonCollisionCfg``
for contact gaps. These configurations author existing USD/Newton attributes.
Native ``physics:masses`` also updates each segment's geometric inertia; a second
array of the same inertia tensors is unnecessary.

Only the model overrides below need ``NewtonCablePropertiesCfg``, which inherits
the existing ``SchemaFragment`` interface. Its custom applier handles typed arrays,
since the generic scalar fragment writer does not support those arrays. It does not
extend ``MassPropertiesCfg`` or joint-drive configurations: imported cable segments
and joints are generated from the curve and have no individual USD body/joint prims
on which to apply those schemas.

.. code-block:: python

   from isaaclab_newton.sim.schemas import NewtonCablePropertiesCfg, apply_newton_cable_properties

   # Patch only the missing model behavior; keep native masses/materials untouched.
   properties = NewtonCablePropertiesCfg(fixed_segments=[0], inertia_regularization=1.0e-6)
   apply_newton_cable_properties(properties, "/World/Cable/geometry/mesh", stage)

``None`` fields leave existing authored values unchanged. The shoelace asset generator
uses this fragment to preserve its existing solver model. In particular:

* Native attachments add constraints; they do not reproduce a zero-mass fixed segment
  with unchanged topology. Native zero segment masses are not accepted by the importer.
* Native material gains are discretized using each joint's local rest length. The
  shoelace model instead uses mean-length scaling and a graded free-tail profile.
  Per-joint overrides preserve those gains exactly; uniform materials cannot replace them.
* Native normals set both body and joint rest frames. The existing model overrides
  initial segment frames while retaining the imported joint rest frames.
* VBD registers Dahl parameters on generated Newton joints, but native cable USD import
  does not populate them from curve materials or per-joint curve arrays.

Attribute contract
~~~~~~~~~~~~~~~~~~

The following optional attributes belong to **Isaac Lab**, not to the upstream
Newton USD schema. Author them on the curve prim. They support a single standalone,
open curve with ``N`` segments and ``N - 1`` cable joints. Segment indices follow
the curve's point order; joint entry ``i`` connects segments ``i`` and ``i + 1``.
Do not store indices from a finalized or replicated model.

.. list-table:: Attributes in the ``isaaclab:cable:`` namespace
   :header-rows: 1
   :widths: 24 18 58

   * - Attribute
     - USD type
     - Meaning
   * - ``fixedSegments``
     - ``int[]``
     - Local segment indices whose mass and inertia, including inverses, are set to zero.
   * - ``inertiaRegularization``
     - ``double``
     - Nonnegative isotropic addition [kg*m^2] to each dynamic segment's inertia. Default: zero.
   * - ``segmentOrientations``
     - ``double4[]``
     - ``N`` unit xyzw quaternions in the curve prim's local frame, composed with its world rotation.
   * - ``jointStiffnesses``
     - ``double4[]``
     - ``N - 1`` stretch, shear, bend, and twist gains, in that order. Linear gains [N/m]; angular gains [N*m/rad].
   * - ``jointDampings``
     - ``double4[]``
     - ``N - 1`` damping gains in the same order. Linear gains [N*s/m]; angular gains [N*m*s/rad].
   * - ``dahlMaxStrains``
     - ``float[]``
     - ``N - 1`` values for VBD's ``dahl_eps_max`` maximum persistent angular strain [rad].
   * - ``dahlDecay``
     - ``float[]``
     - ``N - 1`` values for VBD's ``dahl_tau`` angular memory decay length [rad].

Mass remains controlled by native USD mass properties; inertia regularization does
not add mass. Fixed segments take precedence over regularization.
Joint gains are the per-joint Newton solver gains, not the structural moduli authored
on the cable material. VBD enables Dahl friction only where both Dahl parameters are
positive. Missing attributes retain the native imported values.

``segmentOrientations`` changes initial body frames without recomputing joint rest
frames. This is useful when reproducing a model with a separately authored initial
configuration. To define material directors and their corresponding rest frames
together, use native per-point curve normals instead.

Author physics arrays in a stage with ``metersPerUnit = 1`` and bake scale/shear into
the geometry before writing them. Translation and proper rotation are supported.
Malformed lengths, nonfinite values, invalid fixed indices, negative gains, and invalid
quaternions raise ``ValueError`` at import.
Closed, multi-curve, and welded cable arrays are not supported by this extension.

The importer also transfers an authored physics material's ``physics:dynamicFriction``,
``newton:contactStiffness``, and ``newton:contactDamping`` to the cable capsules, along
with ``newton:contactGap`` from the curve. These are existing USD/Newton attributes.

Fixed fixtures
--------------

A rigid-body prim can opt into ``isaaclab:physics:fixed`` (``bool``, default false).
It retains its imported body and joint topology, with mass and inertia set to zero.
This supports a fixture whose pose is explicitly reset by the task. It does not
create a fixed joint or change an authored kinematic flag. Native ``physics:mass = 0``
is not a substitute: the Newton USD importer treats that as unspecified mass.

Shoelace example
----------------

The ``IsaacContrib-Shoelace-DualFranka`` example includes a baked
``data/shoelace.usda`` and an offline authoring tool:

.. code-block:: bash

   PXR_WORK_THREAD_LIMIT=1 uv run python -m isaaclab_tasks.contrib.shoelace.generate_asset

The task uses ``UsdFileCfg``-based spawning and ``CableObjectCfg(spawn=None)`` views
on the two cable children. Small USD-level overrides retain the task's configurable
friction and proxy inertia. A scene sensor handles finger-tail observations; startup
and reset event terms handle settled poses and randomization. Buffer capacities stay
in the task's simulation configuration and are resolved after CLI overrides.
