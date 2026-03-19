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
- checkpoints for training runs  # 运行要求：元数据(解析配置/种子包/Git信息/环境模式/标识)；输出(步骤指标/片段/运行汇总/训练检查点)

## Baselines

- static UTAA
- heuristic tuning
- PPO meta-controller

## Ablations

- no risk adaptation
- weights only
- risk only
- no action smoothing
