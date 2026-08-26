# Data contract

JiR operates on paired 3D NIfTI volumes. The loaders keep source and target transforms synchronized and return tensors in [0, 1] before model-specific scaling.

## MR-to-PET

~~~text
<root>/
├── train/<class>/<case-id>/
│   ├── <case-id>_M00_T1w.nii.gz
│   └── <case-id>_M00_fdg_pet.nii.gz
└── test/<class>/<case-id>/...
~~~

The default class folders are AD, CN, PMCI, and SMCI.

## Multi-contrast MR

~~~text
<root>/
├── train/<case-id>/
│   ├── <case-id>_t1.nii.gz
│   └── <case-id>_t1ce.nii.gz
└── test/<case-id>/...
~~~

Use DATA_ROOT for the machine-local root. Keep resampling, cropping, and anonymization steps outside this repository; the supplied transforms perform deterministic validation preprocessing and the training augmentation used by the recipes.
