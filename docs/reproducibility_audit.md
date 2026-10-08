# Reproducibility audit

This document records the cleanup and verification performed when converting the original BTP working directory into a public research repository. The goal is to preserve what the experiments actually did, separate verified evidence from inference, and avoid retroactively improving historical methodology without saying so.

## Source archive reviewed

The original working archive contained training scripts, environment variants, checkpoints, logs, datasets, generated media, backups, caches, an external reference paper, and multiple files named with `draft`, `temp`, `v7`, `v8`, `best`, and `final`. The public repository reorganizes those materials by responsibility and preserves only the artifacts needed to understand or reproduce the reported work.

## Refactor-equivalence check

The cleaned SAC environment was compared directly against the archived `environment.py` on the three preserved benchmark images (training square, OOD circle, hollow star). For each benchmark:

- extracted path shape and coordinates matched exactly (`max |Δpath| = 0.0`);
- the first reset observation matched exactly for the same seed (`max |Δobs| = 0.0`);
- 80 fixed continuous actions were stepped through both implementations;
- observations, rewards, termination/truncation state, and sequential coverage matched exactly throughout (`max |Δobs| = 0.0`, `max |Δreward| = 0.0`).

The cleaned historical PPO environment was similarly compared against archived `environment_paper_draft.py` on the training square and OOD circle. Path coordinates, reset observations, 80-step observations/rewards, termination flags, and coverage matched exactly.

The detailed machine-readable summary is in [`refactor_equivalence.json`](refactor_equivalence.json).

## Issues found and resolution

### 1. Missing original from-scratch SAC trainer

**Finding.** The archive contains `FINAL_BTP_MODEL_95pct.zip` and the later DAgger refinement source, but no confirmed original script that created the SAC base checkpoint from scratch.

**Evidence recovered from checkpoint metadata.** The archived base model records 30,000 timesteps, 16-D observations, 2-D actions, learning rate `5e-5`, gamma `0.995`, replay buffer `200000`, batch size `256`, `learning_starts=0`, `tau=.005`, one-step train frequency, one gradient step, fixed entropy coefficient `.0005`, seed `42`, and `[128,128]` actor/critic networks. Historical project notes indicate a 95% sequential success threshold for the base phase.

**Resolution.** `scripts/train_sac_base.py` is explicitly labeled **reconstructed** and uses only these recovered facts. It must not be described as the missing original source.

### 2. Historical PPO smoothing-term discrepancy

**Finding.** The early manuscript described a second-order action-smoothness term. In the archived code, `super().step(action)` updates `prev_action` before the paper reward is evaluated, so the executed expression reduces to a first-order action-change penalty rather than the intended second-order difference.

**Resolution.** `src/bead_rl/ppo_legacy.py` preserves the executed historical behavior so archived PPO checkpoints remain compatible. The discrepancy is documented rather than silently corrected. A future controlled rerun may implement the intended second-order term as a new experiment.

### 3. Multi-shape PPO training-budget mismatch

**Finding.** The early manuscript describes the multi-shape PPO result as a 3M-timestep run. Archived checkpoints show:

- initial final: **3,006,464** timesteps;
- best validation-selected checkpoint: **5,400,004** timesteps;
- extended final: **5,706,464** timesteps.

The OOD comparison uses the best checkpoint, not the ~3M checkpoint.

**Resolution.** All three checkpoints are named unambiguously and their metadata/hashes are recorded. Result descriptions use ~5.4M for the reported best multi-shape PPO comparison.

### 4. Result-file provenance collision

**Finding.** An original summary filename was overwritten during later testing, causing a hollow-star result (about 86.2% sequential, 71.6% geometric @0.10, RMSE ~0.061) to appear where it could be mistaken for a normal training-square standalone evaluation.

**Resolution.** Training-square and hollow-star raw CSVs and normalized summaries now have separate immutable names:

- `results/sac_dagger_training_square.json`
- `results/sac_dagger_hollow_star.json`

The standalone training-square CSV reports 98.27% sequential, 95.59% geometric @0.10, and 0.03246 RMSE over 20 episodes.

### 5. PPO-vs-SAC comparison is not algorithm-only

**Finding.** The archived PPO and SAC pipelines differ in observation dimensionality, reward design, policy architecture, training regime, and algorithm. Cross-pipeline episode rewards are therefore not calibrated to the same objective.

**Resolution.** The repository calls them **training/controller pipelines**, excludes reward from cross-pipeline ranking, and emphasizes geometric coverage plus tracking RMSE. A causal statement that SAC is intrinsically more sample-efficient than PPO is not supported by this experiment alone.

### 6. Procedural six-shape metric labeling

**Finding.** The historical six-shape figure labels a generic `Cov` value of 99.1–100%. Inspection of the corresponding workflow indicates this is waypoint/sequential coverage, not projection-based geometric coverage.

**Resolution.** `results/procedural_batch_sequential.json` records the values explicitly as **sequential coverage**. The README does not claim 99–100% geometric coverage.

### 7. Original procedural dataset seed was not persisted

**Finding.** The exact archived 1,000-image split exists, but the original generator did not record its RNG seed. Therefore regeneration cannot be guaranteed byte-for-byte from the historical script.

**Resolution.** The exact 700/150/150 image split is preserved in the repository, with aggregate SHA-256 digests in `DATASET_METADATA.json`. The cleaned generator accepts an explicit seed for future work and writes a manifest, but newly generated data is not represented as identical to the archived experiment.

### 8. Dependency metadata was incomplete

**Finding.** The original `requirements.txt` omitted packages used by the project and did not capture exact historical versions.

**Resolution.** `pyproject.toml` defines complete runtime/dev dependencies. `requirements-repro.txt` pins versions embedded in the final SAC checkpoint where available and clearly leaves OpenCV/Matplotlib as bounded ranges because their historical versions were not recorded.

### 9. Working-folder clutter / ambiguous names

**Finding.** The archive mixed source, caches, model snapshots, media, draft copies, external papers, and backups in one directory.

**Resolution.** Code is now organized into `src/`, entry points into `scripts/`, immutable artifacts into `models/` and `results/raw/`, presentation figures into `assets/`, and research notes into `docs/`. Ambiguous `v7/v8/temp/final-final` naming is not used in the public source layout.

## Verification performed in the cleanup environment

- Python source/scripts/tests compile successfully with `python -m compileall`.
- Core/unit/artifact tests pass: **13 passed, 1 skipped** in the cleanup environment.
- The single skip is the Gymnasium/Stable-Baselines3 integration test because those libraries are not installed in the isolated cleanup runtime and internet access is unavailable there.
- Exact checkpoint ZIP metadata was read directly from each Stable-Baselines3 archive.
- SHA-256 hashes are recorded for every included checkpoint.
- Dataset split counts and aggregate digests are recorded.
- Raw archived result files were separated from normalized/verified summaries.
- Cleaned SAC and historical PPO environment dynamics were numerically compared against the archived source as described above.

## Remaining verification boundary

A full end-to-end re-execution of all archived neural-network checkpoints was **not possible inside the cleanup sandbox** because Gymnasium and Stable-Baselines3 were unavailable and could not be downloaded. GitHub Actions is configured to install the complete project and run the Gymnasium/SB3 integration test on push/PR. The archived numerical results in this repository are therefore provenance-verified from the original artifacts, not falsely represented as newly reproduced in the cleanup sandbox.
