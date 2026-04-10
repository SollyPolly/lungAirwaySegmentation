# Test Layout

The tests folder should mirror the package layout at a smaller scale.

Rules:

- start with deterministic utility tests before large training tests
- test geometry and intensity helpers before models
- keep heavy integration tests separate from fast unit tests
- use tiny synthetic arrays for most preprocessing checks

Early priorities:

1. path resolution and missing-file validation
2. crop boxes and affine updates
3. intensity clipping and normalization
4. metric sanity checks on toy masks
