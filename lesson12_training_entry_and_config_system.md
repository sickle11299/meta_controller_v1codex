# 第 12 课：训练入口与实验配置系统

## 1. 学习目标

- 理解配置继承、参数覆盖、run dir 管理。
- 学会通过配置组织实验，而不是直接改源码。

## 2. 核心概念

- `config inheritance`：子配置通过 `extends` 继承默认配置，再覆盖局部字段。
- `override`：用 CLI `--set key=value` 在运行时覆盖配置。
- `experiment naming`：通过 `experiment_name` 控制实验目录命名，避免结果互相覆盖。
- `manifest`：记录本次实验环境元信息（seed、commit、branch、dirty）。

## 3. 重点文件导读

### 3.1 配置文件

- `configs/default.yaml`
  - 全局默认参数（`env`、`reward`、`controller`、`seed`、`results_root`）。
- `configs/training.yaml`
  - `extends: configs/default.yaml`
  - 覆盖 `experiment_name=ppo_train`，增加 `train` 段。
- `configs/evaluation.yaml`
  - `extends: configs/default.yaml`
  - 覆盖 `experiment_name=ppo_eval`，增加 `evaluate` 段。

注意：这些 `.yaml` 文件当前内容是 JSON 风格，解析时由 `json.loads` 处理。

### 3.2 配置工具

- `src/meta_controller/utils/config.py`
  - `load_config(path)`：递归处理 `extends`，并做深度合并。
  - `_deep_merge(base, override)`：字典层级合并。
  - `apply_overrides(config, overrides)`：应用 `--set` 覆盖。
  - `parse_value(raw)`：自动把字符串转 `bool/int/float`，否则保留字符串。
  - `build_argument_parser()`：统一提供 `--config`、`--set`、`--results-dir`。

- `src/meta_controller/utils/versioning.py`
  - `git_metadata()`：落盘 `commit/branch/dirty`，提升可追溯性。

## 4. 训练入口调用链

入口文件：`src/meta_controller/experiments/train.py`

命令行到产物的关键路径：

1. `main(argv)` 解析参数。
2. `config = apply_overrides(load_config(args.config), args.set)`。
3. `seed_bundle = set_global_seed(...)` 固定随机性。
4. `run_dir = resolve_run_dir(config, args.results_dir)` 创建实验目录。
5. 写入：
   - `config_resolved.json`
   - `manifest.json`
6. 执行 `run_training(config, run_dir)`。
7. 继续写入：
   - `metrics.jsonl`
   - `episode_summary.json`
   - `summary.json`
   - `checkpoints/last.json`

## 5. run dir 规则（必须掌握）

`resolve_run_dir` 的规则：

`<results-root>/<experiment_name>/seed_<seed>/`

其中：

- `<results-root>` 来自 `--results-dir`，否则用配置 `results_root`。
- `experiment_name` 来自配置（也可被 `--set` 覆盖）。
- `seed` 来自配置（也可被 `--set` 覆盖）。

## 6. 动手练习

### 练习 A：单参数覆盖跑实验

```powershell
$env:PYTHONPATH='src'
python -m meta_controller.experiments.train `
  --config configs/training.yaml `
  --results-dir results\lesson12_demo `
  --set seed=11 `
  --set experiment_name=ppo_train_seed11 `
  --set reward.phi=0.2
```

预期目录：

`results\lesson12_demo\ppo_train_seed11\seed_11\`

### 练习 B：对比原始配置与 resolved 配置

比较文件：

- 原始：`configs/training.yaml`
- 解析后：`results\lesson12_demo\ppo_train_seed11\seed_11\config_resolved.json`

重点检查字段：

- `seed` 是否从 `7 -> 11`
- `experiment_name` 是否从 `ppo_train -> ppo_train_seed11`
- `reward.phi` 是否从 `0.1 -> 0.2`
- 继承字段（如 `env.*`、`controller.*`）是否仍存在

### 练习 C：查看 manifest 可追溯信息

查看：

`results\lesson12_demo\ppo_train_seed11\seed_11\manifest.json`

检查：

- `seed_bundle`
- `commit`
- `branch`
- `dirty`

## 7. 本次实操执行记录（已跑通）

### 7.1 执行命令

```powershell
$env:PYTHONPATH='src'
python -m meta_controller.experiments.train --config configs/training.yaml --results-dir results\lesson12_demo --set seed=11 --set experiment_name=ppo_train_seed11 --set reward.phi=0.2
```

### 7.2 关键输出

- `run_dir`: `results\lesson12_demo\ppo_train_seed11\seed_11`
- `mean_reward`: `0.5121149203794395`
- `mean_success_rate`: `0.5541666666666667`
- `mean_hazard`: `0.08387787679175185`
- `mean_latency_ms`: `2.225975181585371`

### 7.3 落盘文件

- `config_resolved.json`
- `manifest.json`
- `metrics.jsonl`
- `episode_summary.json`
- `summary.json`
- `checkpoints/last.json`

## 8. 配置驱动实验操作模板（可直接复用）

```powershell
# 1) 设定 Python 路径
$env:PYTHONPATH='src'

# 2) 选择基线配置（train / evaluation / ablation）
$config = 'configs/training.yaml'

# 3) 设置实验标识（建议包含日期/seed/改动点）
$exp = 'ppo_train_seed11_phi02'

# 4) 只改一类参数（单变量原则）
python -m meta_controller.experiments.train `
  --config $config `
  --results-dir results\lesson12_template `
  --set experiment_name=$exp `
  --set seed=11 `
  --set reward.phi=0.2

# 5) 核对解析配置与产物
Get-Content results\lesson12_template\$exp\seed_11\config_resolved.json
Get-Content results\lesson12_template\$exp\seed_11\summary.json
Get-Content results\lesson12_template\$exp\seed_11\manifest.json
```

## 9. 常见错误与修正

- 错误：一次用 `--set` 改太多参数，无法判断因果。
  - 修正：单变量实验，其他参数固定。
- 错误：只看控制台，不检查 `config_resolved.json`。
  - 修正：每次跑完都先核对 resolved 配置。
- 错误：`experiment_name` 不变导致结果覆盖或混淆。
  - 修正：把改动信息放进 `experiment_name`。
- 错误：忽略 `manifest.json`，后续找不到对应代码版本。
  - 修正：把 `commit + dirty` 当成实验记录必查项。

## 10. 本课验收清单

- 能解释 `extends` + 深合并 + `--set` 覆盖的顺序。
- 能准确说出 run dir 的命名规则。
- 能独立跑一次覆盖参数实验并验证 `config_resolved.json`。
- 能使用“配置驱动实验模板”复现实验流程。
