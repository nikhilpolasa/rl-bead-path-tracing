# Manuscript corrections before submission

The current manuscript is a draft. The following corrections should be applied before it is presented as a submission-ready research paper.

## Required corrections

### Multi-shape PPO budget

Do not describe the reported best PPO-v2 result as coming from only **3M timesteps**. The archived validation-selected checkpoint used in the strongest OOD comparison contains **5,400,004 timesteps**. The archive also contains the initial 3,006,464-step final checkpoint and an extended 5,706,464-step final checkpoint.

Any statement such as “3M vs 500K” or “6x more environment interaction” must be updated for the checkpoint actually being compared, or the 3M checkpoint must be re-evaluated and used consistently.

### Table I / hollow-star result

The result near **reward 108.5, sequential 86.2%, geometric @0.10 ~71.6%, RMSE ~0.061** corresponds to the hollow-star stress test artifacts. It should not be presented as a second generic in-distribution training-square evaluation.

Use the standalone 20-episode training-square evaluation instead if a second in-distribution row is desired: approximately **98.27% sequential, 95.59% geometric @0.10, RMSE 0.03246**.

### SAC-vs-PPO causal language

The experiment does not isolate only the RL algorithm. Historical PPO and SAC differ in observation size, reward, architecture, data regime, and optimization procedure. Therefore replace broad claims such as:

> SAC+DAgger proves a more sample-efficient route than PPO.

with a narrower statement such as:

> Under the evaluated controller/training configurations and compute budgets, the SAC+DAgger pipeline achieved better OOD geometric tracking than the two PPO pipelines.

A true algorithmic sample-efficiency conclusion requires a controlled ablation using the same MDP, observation, reward, network capacity, evaluation protocol, and comparable interaction budgets.

### Cross-model reward

Do not use mean episode reward to rank SAC against PPO when their reward functions differ. Keep reward as an internal per-pipeline diagnostic; use geometric coverage and RMSE for cross-pipeline comparisons.

### Six-shape coverage wording

The archived six-shape plot's 99.1–100% `Cov` values should be called **sequential coverage**. Do not describe those values as geometric coverage unless the trajectories are re-evaluated with the projection-based metric.

### PPO smoothing-term description

The historical PPO code as executed implements a first-order action-change penalty because of update ordering. The current manuscript's second-order smoothness description does not match the archived behavior. Either:

1. describe the historical executed first-order penalty accurately; or
2. implement the intended second-order term as a new model, retrain, and clearly distinguish the new experiment from archived results.

### Reproducibility statement for SAC base

The original from-scratch SAC trainer is missing from the archive. Do not imply complete historical source reproducibility without qualification. The public repository includes a reconstruction from checkpoint metadata, but the recovered script cannot prove exact historical optimizer/runtime details beyond what the checkpoint records.

### Procedural dataset seed

The exact 1,000-image dataset is preserved, but its original RNG seed was not recorded. State that the archived split is provided verbatim and that future dataset generation is seeded; do not claim deterministic regeneration of the historical images from source alone.

## Recommended additions

- Add a table that distinguishes **training environment**, **observation dimension**, **reward definition**, **interaction budget**, and **checkpoint used** for each pipeline.
- Add geometric coverage to the six-shape test rather than relying only on sequential coverage.
- Add a scale-normalized OOD benchmark to test the current hypothesis that relative contour scale/framing drives much of the residual gap.
- Run a same-MDP PPO-vs-SAC ablation if the paper wants to make algorithm-level conclusions.
- State whether reported means use the same episode seeds across models; use paired seeds for future comparisons where possible.
- Consider reporting confidence intervals in addition to standard deviations for the key 20-episode comparisons.

## Authorship / status wording

Until submission/acceptance, describe the document as **“manuscript in preparation”** or **“research manuscript draft.”** Do not list it as a publication.
