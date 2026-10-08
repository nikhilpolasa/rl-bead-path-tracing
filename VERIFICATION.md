# Verification status

**Public-repository verification status: passed, with documented historical/reproduction differences.**

## Clean-machine verification

A fresh Windows environment (Python 3.11.5) was created from the repository and the pinned reproducibility stack was installed. The following checks passed:

- SHA-256 of the distributed repository ZIP matched the expected release candidate.
- `scripts/verify_artifacts.py` verified **7 checkpoint hashes** and **1,000 dataset images**.
- Full test suite: **15 passed, 0 failed**.
- Archived SAC+DAgger checkpoint loaded and ran end-to-end.
- Archived single-shape and multi-shape PPO checkpoints loaded through the repository's legacy compatibility loader and ran end-to-end.

The exact clean-machine reproduction record is stored in [`results/windows_py311_reproduction_2026-10-08.json`](results/windows_py311_reproduction_2026-10-08.json).

## Reproduced OOD-circle benchmark

20 deterministic-policy episodes, seed base 12000:

| Pipeline | Sequential | Geom. @ 0.10 | RMSE |
| --- | ---: | ---: | ---: |
| SAC + DAgger | 65.3295% | 50.9857% | 0.08405 |
| Single-shape PPO | 26.5473% | 16.6571% | 0.18860 |
| Multi-shape PPO | 70.0716% | 44.8143% | 0.11298 |

The SAC+DAgger and single-shape PPO results reproduce their archived OOD aggregates essentially exactly. Multi-shape PPO reproduces similar geometric coverage and RMSE, but its waypoint-based sequential coverage is higher than the archived 53.01% aggregate. This discrepancy is preserved explicitly rather than silently reconciled, because sequential coverage is known to be sensitive to waypoint advancement and is not the project's preferred geometric generalization metric.

## Training-square reproduction

The final SAC+DAgger checkpoint produced 99.1404% sequential coverage, 90.50% geometric coverage @ 0.10, and RMSE 0.05009 in the clean run. This is strong in-distribution tracking, but it does not exactly match the archived standalone aggregate (98.27% / 95.59% / 0.03246). The historical OpenCV version and exact state of the old standalone evaluation utility were not embedded in the checkpoint, so the repository reports both instead of claiming bit-for-bit historical reproduction.

## Refactor equivalence checks

Before packaging, the refactored environments were numerically compared against the archived source on the square, OOD-circle, and hollow-star benchmarks. Core path, observation, reward, and environment-contract checks passed. The exact archived 700/150/150 procedural split and model archives are preserved by hash.

## Release gate

The local runtime gate is satisfied. After the first GitHub push, require GitHub Actions to pass on Python 3.11 before tagging a release. Manuscript claims should continue to use the archived experiment table where appropriate and clearly distinguish historical results from clean-environment reproduction.

## Legacy PPO checkpoint portability

The archived PPO checkpoints embed a cloudpickled custom ``EluMlpExtractor``
from the original ``__main__`` training script. A clean Windows/Python 3.11
reproduction exposed a cross-version deserialization failure (``TypeError:
super() takes no keyword arguments``) before any weights were evaluated. The
repository now loads those historical checkpoints with
``bead_rl.policies.load_archived_ppo()``, which overrides only the serialized
``policy_kwargs`` with the repository-owned architecture of the same shape and
activations. CI/tests explicitly load both the single-shape and multi-shape PPO
checkpoints through this compatibility path.

