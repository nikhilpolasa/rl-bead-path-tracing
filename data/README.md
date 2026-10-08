# Data

## `benchmarks/`

Three preserved inputs used in the reported evaluation story:

- `training_square.jpeg` — primary single-contour training/in-distribution benchmark;
- `ood_circle.jpg` — differently scaled/framed out-of-distribution probe;
- `hollow_star.jpg` — sharp convex/concave stress test.

## `closed_shapes_dataset/`

The exact archived procedural dataset used for multi-shape PPO training/validation/testing:

- `train/`: 700 images;
- `val/`: 150 images;
- `test/`: 150 images.

The original generator did not persist its RNG seed, so this checked-in image split is the authoritative historical dataset. `DATASET_METADATA.json` records split counts and aggregate SHA-256 digests.

For new experiments, `scripts/generate_closed_shapes.py` creates a deterministic seeded dataset and manifest. A regenerated dataset should be treated as a new experimental dataset, not as byte-identical historical reconstruction.

A separate historical open/self-intersecting curve dataset existed in the working folder but was not used in the reported experiments, so it is intentionally omitted from this focused repository.
