# JiR: Just image Regularizer for Medical Image Translation

[![Paper](https://img.shields.io/badge/MICCAI-2026-blue)](https://github.com/siyuan-mei/JiR)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C)](https://pytorch.org/)
[![MONAI](https://img.shields.io/badge/MONAI-1.x-00A6D6)](https://monai.io/)
[![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)

This repository contains the official PyTorch implementation of:

> **Time Matters: Rethinking Diffusion and Flow Models in One-Step Medical Image Translation**  
> Siyuan Mei, Yanteng Zhang, Yan Xia, Qizhen Lan, Yipeng Sun, Siming Bayer, Zirong Li, Chengze Ye, Daiqi Liu, Xiaoqian Jiang, Fuxin Fan, Yixing Huang, and Andreas Maier.

JiR revisits diffusion and flow models for fidelity-driven medical image translation. Instead of relying on diversity sampling or long iterative denoising, JiR keeps the useful part for paired medical translation: the time-conditioned training signal. In this view, diffusion becomes a **Just image Regularizer** with `t`, enabling a deterministic one-step generator for tasks such as MR-to-PET and multi-contrast MR translation.

The current codebase follows a compact mmseg-style design: datasets, model definitions, losses, trainer logic, and experiment recipes are separated by config files. You can switch tasks and model variants mainly by changing a YAML config.

## Highlights

- **One-step deterministic translation** for paired 3D medical images.
- **Time-conditioned regularization** through diffusion/flow-style interpolation.
- **Task-driven forward processes**, including `NiF`, `PiF`, and the default bridge-image flow `BiF`.
- **Image-space prediction** for direct target reconstruction rather than velocity-only supervision.
- **Optional time-consistency loss** to align predictions at arbitrary time points with endpoint predictions.
- **3D CondUNet backbone** with timestep/class conditioning through FiLM-style modulation.
- **Lightning Fabric trainer** with checkpointing, validation metrics, distributed training, and W&B visualization.

## Repository Layout

```text
JiR/
|-- configs/
|   |-- default_config.yaml
|   |-- mri2pet/
|   |   |-- diffusion.yaml
|   |   `-- baseline_resunet.yaml
|   `-- brats/
|       |-- diffusion.yaml
|       `-- baseline-resunet.yaml
|-- data/
|   |-- dataset/
|   |   |-- mri2pet.py
|   |   `-- brats.py
|   |-- transforms.py
|   `-- utils.py
|-- evaluation/
|   `-- image_metrics.py
|-- models/
|   |-- diffusion_model.py
|   |-- encoder_decoder.py
|   |-- losses.py
|   |-- my_callbacks.py
|   |-- register.py
|   `-- nets/
|       |-- cond_unet.py
|       `-- swinunetr.py
|-- trainer_fabric.py
|-- main.py
|-- dist_train_h100_sif.sh
|-- requirements.txt
`-- requirements_cluster.txt
```

More detailed notes are available in [docs/DATA.md](docs/DATA.md), [docs/CONFIGS.md](docs/CONFIGS.md), and [docs/TRAINING.md](docs/TRAINING.md).

## Installation

Clone the repository:

```bash
git clone https://github.com/siyuan-mei/JiR.git
cd JiR
```

Create an environment and install dependencies:

```bash
conda create -n jir python=3.10 -y
conda activate jir

# Install a PyTorch build that matches your CUDA driver.
# Example for CUDA 12.6:
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu126

pip install -r requirements.txt
```

For the cluster environment used in our experiments, see [requirements_cluster.txt](requirements_cluster.txt).

## Data Preparation

The loaders expect paired NIfTI volumes (`.nii.gz`) with deterministic file names. The data root can be set either in the config under `data_cfg.data_root` or at runtime with:

```bash
export DATA_ROOT=/path/to/dataset/fold1
```

### ADNI MR-to-PET

`data/dataset/mri2pet.py` expects each split to contain diagnostic class folders, and each subject folder to contain one T1w MRI and one FDG-PET volume:

```text
DATA_ROOT/
|-- train/
|   |-- AD/
|   |   `-- 002S0413/
|   |       |-- 002S0413_M00_T1w.nii.gz
|   |       `-- 002S0413_M00_fdg_pet.nii.gz
|   |-- CN/
|   |-- PMCI/
|   `-- SMCI/
`-- test/
    |-- AD/
    |-- CN/
    |-- PMCI/
    `-- SMCI/
```

The class labels are mapped as `AD=0`, `CN=1`, `PMCI=2`, and `SMCI=3`. The loader currently validates MR/PET volumes against shape `(1, 112, 128, 112)` before MONAI transforms crop/pad/resize them.

### BraTS Multi-Contrast MR

`data/dataset/brats.py` expects each patient folder to contain a source T1w volume and target T1ce volume:

```text
DATA_ROOT/
|-- train/
|   `-- BraTS2023_00000/
|       |-- BraTS2023_00000_t1.nii.gz
|       `-- BraTS2023_00000_t1ce.nii.gz
`-- test/
    `-- BraTS2023_00001/
        |-- BraTS2023_00001_t1.nii.gz
        `-- BraTS2023_00001_t1ce.nii.gz
```

The default BraTS config center-crops volumes to `128 x 192 x 192` and resamples to `96 x 160 x 160` for memory efficiency.

## Config System

JiR uses OmegaConf YAML configs in a style similar to mmseg/mmengine projects:

- `configs/default_config.yaml` defines shared defaults.
- Task configs such as `configs/mri2pet/diffusion.yaml` override dataset, model, optimizer, and validation settings.
- `main.py --config_path=mri2pet/diffusion` loads `configs/mri2pet/diffusion.yaml` and merges it with the defaults.
- Networks are built by registry through `model_cfg.net.type`, for example `CondUNet`.

A minimal config has four main blocks:

```yaml
data_cfg:
  dataset: "mri2pet"
  data_root: null
  crop_size: [128, 128, 128]

model_cfg:
  model_type: "diffusion"
  net:
    type: "CondUNet"
  diffusion:
    flow_mode: "BiF"
    prediction_space: "x"
    loss_space: "x"

train_cfg:
  devices: 1
  batch_size: 4
  num_epochs: 500

optim_cfg:
  optimizer: "adam"
  base_lr: 2e-4
```

## Training

Train JiR on ADNI MR-to-PET:

```bash
export DATA_ROOT=/path/to/mri2pet/adni_split_data/fold1
wandb login

python main.py \
  --config_path=mri2pet/diffusion \
  --run_name=jir_bif_xpred
```

Train the regression-style baseline:

```bash
export DATA_ROOT=/path/to/mri2pet/adni_split_data/fold1

python main.py \
  --config_path=mri2pet/baseline_resunet \
  --run_name=resunet_regression
```

Train JiR on BraTS T1-to-T1ce:

```bash
export DATA_ROOT=/path/to/brats/fold1

python main.py \
  --config_path=brats/diffusion \
  --run_name=jir_bif_t1_to_t1ce
```

Run with Lightning Fabric distributed launch:

```bash
fabric run \
  --accelerator=gpu \
  --strategy=ddp_find_unused_parameters_true \
  --devices=4 \
  --num-nodes=1 \
  main.py --config_path=mri2pet/diffusion --run_name=jir_4gpu
```

An example SLURM + Apptainer script for H100 nodes is provided in [dist_train_h100_sif.sh](dist_train_h100_sif.sh).

## Evaluation

Validate a trained checkpoint and save generated NIfTI volumes:

```bash
export DATA_ROOT=/path/to/mri2pet/adni_split_data/fold1

python main.py \
  --config_path=mri2pet/diffusion \
  --run_name=jir_bif_xpred \
  --no_train \
  --if_eval \
  --short_ckpt_path=best_mse_epoch450.ckpt
```

You can also pass an absolute checkpoint path:

```bash
python main.py \
  --config_path=mri2pet/diffusion \
  --run_name=eval_external_ckpt \
  --no_train \
  --if_eval \
  --full_ckpt_path=/path/to/checkpoint.ckpt
```

Evaluation reports MAE, MSE, PSNR, SSIM, MS-SSIM, and a custom 3D SSIM implementation. Predictions are saved under:

```text
work_dir/<dataset>/<run_name>/outputs/
|-- input_<case_id>.nii.gz
|-- syn_<case_id>.nii.gz
`-- target_<case_id>.nii.gz
```

## W&B Visualization

The entrypoint creates a W&B run named `<dataset>/<run_name>` in the `MRI2PET` project by default. The logger records:

- total and trainable parameter counts,
- step-level training losses and learning rate,
- epoch-level validation metrics,
- middle-slice input/target/prediction panels,
- colorized prediction and target maps,
- absolute-error heatmaps during validation.

Useful environment variables:

```bash
export WANDB_PROJECT=MRI2PET
export WANDB_MODE=online
export LOG_ROOT=/path/to/logs
export OUTPUT_ROOT=/path/to/outputs
```

If you run on an offline cluster:

```bash
export WANDB_MODE=offline
python main.py --config_path=mri2pet/diffusion --run_name=offline_debug
wandb sync work_dir/mri2pet/offline_debug/logs/wandb/offline-run-*
```

## Extending JiR

To add a dataset:

1. Implement a `torch.utils.data.Dataset` under `data/dataset/`.
2. Return a dict containing at least `input`, `target`, `cls_code`, `class`, and `id`.
3. Register it in `build_datasets()` inside [main.py](main.py).
4. Add a YAML recipe under `configs/<task>/`.

To add a model backbone:

1. Implement the module under `models/nets/`.
2. Decorate the class with `@MODELS.register(type="YourModel")`.
3. Import it in `models/nets/__init__.py`.
4. Set `model_cfg.net.type: "YourModel"` in a config.

## Citation

If this repository is useful for your research, please cite:

```bibtex
@inproceedings{mei2026jir,
  title={Time Matters: Rethinking Diffusion and Flow Models in One-Step Medical Image Translation},
  author={Mei, Siyuan and Zhang, Yanteng and Xia, Yan and Lan, Qizhen and Sun, Yipeng and Bayer, Siming and Li, Zirong and Ye, Chengze and Liu, Daiqi and Jiang, Xiaoqian and Fan, Fuxin and Huang, Yixing and Maier, Andreas},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year={2026}
}
```

## Acknowledgements

This implementation builds on PyTorch, MONAI, Lightning Fabric, timm, SimpleITK, and Weights & Biases. The repository style is inspired by clean research codebases such as [LTH14/JiT](https://github.com/LTH14/JiT).

## License

This project is released under the [Apache License 2.0](LICENSE). The license applies to code in this repository; datasets are governed by their original providers.
