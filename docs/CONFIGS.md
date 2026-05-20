# Configs and Code Style

JiR follows a lightweight mmseg-style organization:

- experiment behavior is declared in YAML configs,
- object construction is driven by registries,
- datasets, networks, models, losses, and trainer logic are separated,
- `main.py` wires the pieces together without hard-coding experiment details.

## Config Loading

The entrypoint calls:

```python
configs.load_and_merge_config(config_path)
```

where `config_path` is a path relative to `configs/` without the `.yaml` suffix. For example:

```bash
python main.py --config_path=mri2pet/diffusion
```

loads:

```text
configs/mri2pet/diffusion.yaml
```

and merges it with:

```text
configs/default_config.yaml
```

The merged config is saved to:

```text
work_dir/<dataset>/<run_name>/config.yaml
```

## Main Config Blocks

### `data_cfg`

Controls the dataset and image transforms:

```yaml
data_cfg:
  dataset: "mri2pet"
  data_root: null
  crop_size: [128, 128, 128]
  resize_size: null
  patch_based: False
```

`dataset` is dispatched in `main.py::build_datasets()`. Current values are `mri2pet` and `brats`.

### `model_cfg`

Controls the high-level model wrapper and backbone:

```yaml
model_cfg:
  model_type: "diffusion"
  net:
    type: "CondUNet"
    input_channels: 1
    n_stages: 6
    features_per_stage: [32, 64, 128, 256, 320, 320]
    num_classes: 1
    use_img_cond: True
    use_t_emb: True
    use_y_emb: False
```

`model_type` selects one of:

- `diffusion`: `models.diffusion_model.DiffusionModel`
- `encoder_decoder` / `normal`: `models.encoder_decoder.EncoderDecoder`

`net.type` is resolved by `models/register.py`. Backbones register themselves with:

```python
@MODELS.register(type="CondUNet")
class CondUNet(nn.Module):
    ...
```

### `model_cfg.diffusion`

Controls JiR-specific flow and sampling behavior:

```yaml
diffusion:
  use_img_cond: True
  flow_mode: "BiF"
  drift_scale: 1
  prediction_space: "x"
  loss_space: "x"
  t_weighted: True
  use_t_consistency: False
  lambda_t: 1
  t_sample: "beta"
  beta1: 1
  beta2: 1
  sampling_method: "heun"
  num_sampling_steps: 1
```

Important fields:

- `flow_mode`: `NiF`, `PiF`, or `BiF`; configs use lowercase-insensitive values.
- `prediction_space`: `x` for direct image prediction or `v` for velocity prediction.
- `loss_space`: target space for the supervision objective.
- `t_weighted`: applies time-dependent loss weighting.
- `use_t_consistency`: enables endpoint-consistency regularization.
- `num_sampling_steps`: `1` gives one-step deterministic generation.

### `train_cfg`

Controls hardware, dataloader, epochs, and checkpoint frequency:

```yaml
train_cfg:
  devices: 1
  strategy: "auto"
  precision: "32-true"
  batch_size: 4
  num_workers: 4
  num_epochs: 500
  save_freq: 50
  loss: ["mse"]
```

`devices: "auto"` uses all visible GPUs. If `automatic_lr_rescale: True`, `main.py` scales the base learning rate by `sqrt(num_devices)`.

### `val_cfg`

Controls validation cadence and sliding-window settings:

```yaml
val_cfg:
  start_val: 0
  val_freq: 50
  limit_val_batches: .inf
  infer_mode: "constant"
```

Validation metrics are grouped by class and also averaged across all samples.

### `optim_cfg`

Controls timm optimizer and scheduler creation:

```yaml
optim_cfg:
  optimizer: "adam"
  base_lr: 2e-4
  warmup_epochs: 5
  scheduler: "warmup"
  betas: [0.9, 0.95]
  weight_decay: 0.0
```

Supported scheduler names include `cosine`, `plateau`, `poly`, and `warmup`.

## Current Recipes

| Config | Task | Role |
| --- | --- | --- |
| `mri2pet/diffusion` | ADNI T1w MRI -> FDG-PET | JiR / BiF / image prediction |
| `mri2pet/baseline_resunet` | ADNI T1w MRI -> FDG-PET | regression-style baseline |
| `brats/diffusion` | BraTS T1 -> T1ce | JiR / BiF / time consistency |
| `brats/baseline-resunet` | BraTS T1 -> T1ce | regression-style baseline |

## Adding a New Recipe

Create:

```text
configs/<task>/<name>.yaml
```

Then run:

```bash
python main.py --config_path=<task>/<name> --run_name=<experiment_name>
```

Keep local machine paths such as `data_root` out of committed configs when possible. Prefer:

```bash
export DATA_ROOT=/path/to/data
```

so the same YAML can run across machines and clusters.
