# Original archive inventory

The source BTP ZIP was enumerated before cleanup. The extracted working directory contained **2,108 files**: primarily 2,000 dataset images, plus Python sources, checkpoints, logs, figures, backups, caches, and an external reference paper.

The archive was classified by artifact type before cleanup. Aggregate counts and public-repository disposition are recorded in [`archive_inventory.csv`](archive_inventory.csv). The original ZIP remains the immutable private source of truth, so the public repository does not expose incidental workstation filenames or redundant working artifacts.

## Cleanup policy

- The exact 1,000-image closed-shape dataset used by reported experiments is preserved verbatim.
- Final / scientifically relevant model checkpoints are preserved verbatim under descriptive names; their hashes match the source archive.
- Core source logic was audited and refactored into `src/bead_rl/` and `scripts/` rather than copying ambiguous `temp`, `draft`, `v7`, or `v8` filenames.
- Raw result artifacts needed to substantiate reported numbers are retained under `results/raw/`.
- Representative figures are retained under `assets/`.
- Duplicate backups, Python caches, TensorBoard event files, redundant intermediate checkpoints, and generated video/GIF media are omitted.
- The separate 1,000-image open/self-intersecting `curve_dataset` is omitted because it was collected for future work and was not used in the reported experiments.
- The third-party reference-paper PDF and text extracts are not redistributed.

## Artifact identity checks

The seven public checkpoint files are byte-for-byte copies of the corresponding archived models (verified by SHA-256). The checked-in 700/150/150 closed-shape dataset also has identical aggregate filename+content SHA-256 digests to the source archive for every split. Detailed hashes for public checkpoints and dataset splits are stored in `models/checkpoint_metadata.json` and `data/closed_shapes_dataset/DATASET_METADATA.json`.

The original ZIP should still be retained privately as the immutable archival source; this repository is the cleaned, research-facing projection of that archive.
