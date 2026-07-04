# Experiment Configs

These files are shell-style environment configs for the scripts in `scripts/`.

Example:

```bash
CONFIG=configs/main_600_pipeline.env bash scripts/run_pipeline.sh
```

CPU examples:

```bash
CONFIG=configs/main_600_compare.env bash scripts/run_compare_retrieval.sh
CONFIG=configs/main_600_pipeline.env bash scripts/run_pipeline.sh
CONFIG=configs/main_600_tables.env bash scripts/make_tables.sh
```

GPU examples:

```bash
CONFIG=configs/main_600_compare_gpu.env bash scripts/run_compare_retrieval.sh
CONFIG=configs/main_600_pipeline_gpu.env bash scripts/run_pipeline.sh
CONFIG=configs/main_600_tables_gpu.env bash scripts/make_tables.sh
CONFIG=configs/controlled_length_100_gpu.env bash scripts/run_controlled_length_experiment.sh
```

When `CONFIG=... bash scripts/...` is used, values in the config file are loaded
first and then script defaults fill any missing variables. To override one value
for a single run, either edit the config or source it manually and then assign
the override before launching the script.
