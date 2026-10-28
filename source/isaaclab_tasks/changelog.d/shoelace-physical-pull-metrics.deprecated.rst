* Deprecated the ambiguous ``Metrics/shoelace/pull_left`` and ``pull_right`` names while retaining
  their values as score aliases. Dashboards should use the new displacement tags for physical progress
  or the explicit score tags for reward diagnostics; historical score curves are not distances and
  cannot be compared numerically with the new displacement curves.
