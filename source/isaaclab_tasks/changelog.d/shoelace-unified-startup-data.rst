Changed
^^^^^^^

* Baked shoelace reference orientations, segment lengths, and settled poses into the
  composite USD asset. Startup events read task-local USD attributes instead of
  reopening the source curve and settled-pose archive. Kept native rest geometry
  separate from settled initial state. Regenerate older composite assets with the
  shoelace asset generator before using the updated startup events.
* Moved offline source paths and geometry-authoring constants into ``asset_authoring``
  and removed runtime event imports of that module. Offline tools should import these
  constants from ``asset_authoring`` instead of ``shoelace_constants``.
