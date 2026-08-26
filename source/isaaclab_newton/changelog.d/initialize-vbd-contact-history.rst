Fixed
^^^^^

* Initialized the Newton collision pipeline before constructing VBD when rigid-contact history is
  enabled, allowing the history buffers to be allocated before CUDA graph capture.
