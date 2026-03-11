# Experiment Protocol

## Required metadata per run

- resolved config
- seed bundle
- git commit and dirty flag
- environment mode (`sim`, `trace`, or `live`)
- baseline or policy identifier

## Required outputs per run

- step-level metrics log
- episode summary
- run summary
- checkpoints for training runs

## Baselines

- static UTAA
- heuristic tuning
- PPO meta-controller

## Ablations

- no risk adaptation
- weights only
- risk only
- no action smoothing
