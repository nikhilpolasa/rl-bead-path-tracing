# Experiment provenance

This file connects the public checkpoints and result summaries to the historical experiment state recovered from the BTP archive.

## Checkpoints

| Public filename | Algorithm / stage | Observation | Archived timesteps | Intended use |
| --- | --- | ---: | ---: | --- |
| `sac_base_95.zip` | SAC base | 16-D | 30,000 | Pre-DAgger base, historical 95% success threshold |
| `sac_dagger_final.zip` | SAC actor after DAgger repair | 16-D | 30,000 SB3 RL steps | Final controller; actor was additionally updated by supervised DAgger |
| `ppo_single_best.zip` | PPO single-shape best snapshot | 25-D | 200,000 | Intermediate archived single-shape checkpoint |
| `ppo_single_final.zip` | PPO single-shape | 25-D | 507,904 | Single-contour baseline used in historical comparison |
| `ppo_multishape_3m_final.zip` | PPO multi-shape initial phase | 25-D | 3,006,464 | End of the initial ~3M run |
| `ppo_multishape_best.zip` | PPO multi-shape validation-selected best | 25-D | 5,400,004 | Checkpoint used for the strongest reported PPO-v2 OOD comparison |
| `ppo_multishape_extended_final.zip` | PPO multi-shape extended final | 25-D | 5,706,464 | End of extended training |

Exact SHA-256 hashes, SB3 versions, and recovered hyperparameters are in [`../models/checkpoint_metadata.json`](../models/checkpoint_metadata.json).

**Important:** the `num_timesteps` stored in `sac_dagger_final.zip` remains 30,000 because DAgger updates the actor by supervised optimization outside `model.learn`; this does not mean DAgger performed zero additional optimization.

## Key result provenance

### SAC+DAgger — training square

Source: archived `complete_model_evaluation.csv`, 20 evaluation episodes.

Normalized summary: `results/sac_dagger_training_square.json`.

Mean values:

- sequential coverage: 0.9826635;
- geometric coverage @0.10: 0.9558563;
- RMSE: 0.0324566;
- reward: 143.4283;
- steps: 1589.9.

The archive also contains a separate training-script final-validation summary of 0.9672 sequential, 0.9186 geometric @0.10, and 0.03814 RMSE. The repository keeps that raw JSON separately rather than averaging or reconciling different evaluation utilities after the fact.

### SAC+DAgger — hollow-star stress test

Source: archived `star_test_results.csv`, 20 episodes.

Normalized summary: `results/sac_dagger_hollow_star.json`.

Mean values:

- sequential coverage: 0.8624642;
- geometric coverage @0.10: 0.7164286;
- RMSE: 0.0607727;
- reward: 108.4641.

### OOD circular probe — three pipelines

Primary normalized summary: `results/ood_circle_comparison_verified.json`.

- SAC+DAgger and multi-shape PPO aggregates come from the original comparison JSON.
- Single-shape PPO's aggregate was recovered from the preserved historical comparison log because the corresponding JSON aggregate was no longer available in the final working directory. It is retained at the precision printed in that log.

Cross-pipeline rewards are retained for provenance but should **not** be interpreted as a fair performance metric because reward formulations differ.

### Six-shape procedural batch

Source: archived `batch_test_results.png`.

The displayed coverage values were transcribed to `results/procedural_batch_sequential.json`. They are labeled sequential waypoint coverage because the historical figure does not establish projection-based geometric coverage for that batch.

## Dataset provenance

The exact archived closed-shape dataset contains:

- 700 training images;
- 150 validation images;
- 150 test images.

Aggregate per-split hashes are in `data/closed_shapes_dataset/DATASET_METADATA.json`. The historical generator did not record a seed, so the exact archived images—not a regenerated approximation—are the authoritative data for the reported multi-shape PPO experiment.
