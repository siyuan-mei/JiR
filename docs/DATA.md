# Data Preparation

This project uses paired 3D medical images saved as NIfTI files (`.nii.gz`). Dataset classes live in `data/dataset/` and return a dict consumed by MONAI transforms and the Fabric trainer.

Each sample contains:

```python
{
    "input": <source volume tensor>,
    "target": <target volume tensor>,
    "cls_code": <integer condition label>,
    "class": <string label>,
    "id": <case id>,
}
```

The loader reads images with SimpleITK as `[D, H, W]`, adds a channel dimension, applies task-specific checks, and then sends both source and target through the same spatial transforms.

## Runtime Data Root

You can configure the dataset root in two equivalent ways:

```bash
export DATA_ROOT=/path/to/fold1
```

or inside a local config:

```yaml
data_cfg:
  data_root: /path/to/fold1
```

`DATA_ROOT` takes priority over `data_cfg.data_root`. This is convenient on clusters where data is staged to `$TMPDIR`.

## ADNI MR-to-PET Layout

Loader: `data/dataset/mri2pet.py`

Expected split names:

- `train`
- `test`

Expected diagnostic classes:

- `AD`
- `CN`
- `PMCI`
- `SMCI`

Expected file names inside each subject folder:

- `<subject_id>_M00_T1w.nii.gz`
- `<subject_id>_M00_fdg_pet.nii.gz`

Example:

```text
DATA_ROOT/
|-- train/
|   |-- AD/
|   |   |-- 002S0413/
|   |   |   |-- 002S0413_M00_T1w.nii.gz
|   |   |   `-- 002S0413_M00_fdg_pet.nii.gz
|   |   `-- ...
|   |-- CN/
|   |-- PMCI/
|   `-- SMCI/
`-- test/
    |-- AD/
    |-- CN/
    |-- PMCI/
    `-- SMCI/
```

Class mapping:

```python
{"AD": 0, "CN": 1, "PMCI": 2, "SMCI": 3}
```

The current loader checks that both input and target have shape `(1, 112, 128, 112)` before transforms. If your preprocessing produces a different shape, update the check in `Mri2PetDataset.__getitem__()` or standardize the volumes before training.

## BraTS T1-to-T1ce Layout

Loader: `data/dataset/brats.py`

Expected split names:

- `train`
- `test`

Expected file names inside each patient folder:

- `<patient_id>_t1.nii.gz`
- `<patient_id>_t1ce.nii.gz`

Example:

```text
DATA_ROOT/
|-- train/
|   |-- BraTS2023_00000/
|   |   |-- BraTS2023_00000_t1.nii.gz
|   |   `-- BraTS2023_00000_t1ce.nii.gz
|   `-- ...
`-- test/
    |-- BraTS2023_00001/
    |   |-- BraTS2023_00001_t1.nii.gz
    |   `-- BraTS2023_00001_t1ce.nii.gz
    `-- ...
```

Class mapping:

```python
{"t1": 0, "t1ce": 1}
```

## Transform Pipeline

Training and validation transforms are defined in `data/transforms.py`.

Training:

- `CenterSpatialCropd`
- optional `DivisiblePadd` and `Resized`
- `ScaleIntensityd` to `[0, 1]`
- random flip along spatial axis 2
- `EnsureTyped`

Validation:

- same deterministic crop/pad/resize path
- `ScaleIntensityd` to `[0, 1]`
- no random augmentation

The diffusion model internally rescales inputs from `[0, 1]` to `[-1, 1]` during training and converts predictions back to `[0, 1]` during validation.

## Output Layout

Training artifacts are written under:

```text
work_dir/<dataset>/<run_name>/
|-- config.yaml
|-- checkpoints/
|-- logs/
`-- outputs/
```

Validation and evaluation outputs use the following naming convention:

```text
outputs/
|-- input_<case_id>.nii.gz
|-- syn_<case_id>.nii.gz
`-- target_<case_id>.nii.gz
```

`work_dir/`, checkpoints, W&B logs, and NIfTI outputs are ignored by git.
