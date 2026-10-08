# Archived model checkpoints

These checkpoints are preserved because they are small enough for the repository and are required to audit the historical results.

| File | Meaning |
| --- | --- |
| `sac_base_95.zip` | SAC base checkpoint before DAgger actor refinement |
| `sac_dagger_final.zip` | Final SAC actor after DAgger refinement |
| `ppo_single_best.zip` | Historical intermediate single-shape PPO checkpoint |
| `ppo_single_final.zip` | Final single-shape PPO baseline |
| `ppo_multishape_3m_final.zip` | End of initial ~3M multi-shape PPO phase |
| `ppo_multishape_best.zip` | Validation-selected best multi-shape PPO checkpoint (~5.4M steps) |
| `ppo_multishape_extended_final.zip` | End of extended multi-shape PPO training (~5.7M steps) |

`checkpoint_metadata.json` contains SHA-256 hashes and metadata read directly from the Stable-Baselines3 archives. The checkpoints are historical artifacts; renaming them here does not alter their contents.
