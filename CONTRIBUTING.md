# Contributing

Thank you for your interest in JiR. The project is small by design, so contributions are easiest to review when they stay close to the existing config-driven style.

## Development Style

- Keep experiment behavior in YAML configs rather than hard-coding paths or hyperparameters.
- Follow the existing mmseg-like registry pattern for new networks and losses.
- Keep dataset loaders deterministic and explicit about expected file names.
- Return the standard batch keys: `input`, `target`, `cls_code`, `class`, and `id`.
- Avoid committing local data paths, checkpoints, W&B runs, generated NIfTI files, or cluster scratch outputs.

## Before Opening a Pull Request

Please run a lightweight sanity check:

```bash
python -m compileall main.py trainer_fabric.py configs data evaluation models
```

If you changed training logic, include the command you used for a smoke run, for example:

```bash
python main.py --config_path=mri2pet/diffusion --run_name=debug --no_train
```

## Adding a Dataset

1. Add a loader in `data/dataset/`.
2. Register the dispatch branch in `main.py::build_datasets()`.
3. Document the required directory layout in `docs/DATA.md`.
4. Add at least one config under `configs/<task>/`.

## Adding a Backbone

1. Implement the model in `models/nets/`.
2. Register it with `@MODELS.register(type="YourModel")`.
3. Import it in `models/nets/__init__.py`.
4. Add a config using `model_cfg.net.type: "YourModel"`.

## Reporting Issues

Please include:

- the config path,
- the command line,
- GPU and PyTorch versions,
- dataset layout summary,
- the first relevant traceback lines.
