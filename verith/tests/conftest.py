# torch must load before the reactor C-extension (libtorch rpath); importing it
# here ensures the right order regardless of a test module's own import order.
import torch  # noqa: F401

# The projects the tests generate and build require ReactLib's packages
# from `$REACTLIB` and build against the Lake package cache
# `$REACTOR_LAKE_CACHE`; `just` sets both (see the justfile). A test that
# needs one and finds it unset is skipped.
