# Training, Evaluation, and W&B

This document collects practical commands for running JiR locally or on a cluster.

## Single-GPU Training

ADNI MR-to-PET:

```bash
export DATA_ROOT=/path/to/mri2pet/adni_split_data/fold1
wandb login

python main.py \
  --config_path=mri2pet/diffusion \
  --run_name=jir_bif_xpred
```

BraTS T1-to-T1ce:

```bash
export DATA_ROOT=/path/to/brats/fold1

python main.py \
  --config_path=brats/diffusion \
  --run_name=jir_bif_t1_to_t1ce
```

Baseline:

```bash
export DATA_ROOT=/path/to/mri2pet/adni_split_data/fold1

python main.py \
  --config_path=mri2pet/baseline_resunet \
  --run_name=resunet_regression
```

## Resume Training

When `--resume` is passed, the trainer looks for the latest checkpoint under the run's `checkpoints/` folder:

```bash
python main.py \
  --config_path=mri2pet/diffusion \
  --run_name=jir_bif_xpred \
  --resume
```

## Multi-GPU Training

Use Lightning Fabric launch:

```bash
fabric run \
  --accelerator=gpu \
  --strategy=ddp_find_unused_parameters_true \
  --devices=4 \
  --num-nodes=1 \
  --precision=32-true \
  main.py --config_path=mri2pet/diffusion --run_name=jir_4gpu
```

For SLURM + Apptainer, adapt:

```bash
bash dist_train_h100_sif.sh --config_path=mri2pet/diffusion --run_name=jir_h100
```

Set the following variables before submission:

```bash
export DATA_ARCHIVE=/path/to/mri2pet/fold1.tar
export SIF_PATH=/path/to/env.sif
export PROJECT_ROOT=/path/to/JiR
```

The script stages data to `$TMPDIR`, exports `DATA_ROOT`, and launches Fabric with 4 GPUs.

## Evaluation

Evaluate a checkpoint inside the run directory:

```bash
export DATA_ROOT=/path/to/mri2pet/adni_split_data/fold1

python main.py \
  --config_path=mri2pet/diffusion \
  --run_name=jir_bif_xpred \
  --no_train \
  --if_eval \
  --short_ckpt_path=best_mse_epoch450.ckpt
```

Evaluate an arbitrary checkpoint:

```bash
python main.py \
  --config_path=mri2pet/diffusion \
  --run_name=eval_external_ckpt \
  --no_train \
  --if_eval \
  --full_ckpt_path=/path/to/checkpoint.ckpt
```

Generated volumes are written to:

```text
work_dir/<dataset>/<run_name>/outputs/
```

## W&B

The code uses `wandb.integration.lightning.fabric.WandbLogger`. By default, the project is named `MRI2PET` and the run name is `<dataset>/<run_name>`.

Online logging:

```bash
wandb login
export WANDB_MODE=online
python main.py --config_path=mri2pet/diffusion --run_name=jir_wandb
```

Offline logging:

```bash
export WANDB_MODE=offline
python main.py --config_path=mri2pet/diffusion --run_name=jir_offline
wandb sync work_dir/mri2pet/jir_offline/logs/wandb/offline-run-*
```

Logged values include:

- `params_total_M` and `params_trainable_M`
- `step/total_loss`, `step/mse`, and `step/lr`
- `epoch/lr` and epoch-averaged losses
- `validation/mae`, `validation/mse`, `validation/psnr`, `validation/ssim`, `validation/ms_ssim`
- train/validation middle-slice images
- colorized target/prediction maps
- error heatmaps

## Common Troubleshooting

### `data_root does not exist`

Set:

```bash
export DATA_ROOT=/absolute/path/to/fold1
```

or update `data_cfg.data_root` in a local, uncommitted config.

### `No CUDA GPU detected`

`main.py` currently creates the trainer with `accelerator="gpu"`. Run on a CUDA machine or modify the entrypoint for CPU debugging.

### Unknown model type

Make sure the backbone class is imported in `models/nets/__init__.py`; otherwise the registry will not see it.

### W&B blocked on a cluster

Use:

```bash
export WANDB_MODE=offline
```

and sync the run after the job finishes.
