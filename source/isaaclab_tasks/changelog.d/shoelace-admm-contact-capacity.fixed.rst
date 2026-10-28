* Scaled Shoelace's internal ADMM contact-reduction hashtable with the per-process environment
  count to avoid contact-reduction hashtable saturation in large training batches.
  Preserved contact matching, its deterministic triangle-pair capacity limit, and
  larger explicit budgets and hashtable factors.
