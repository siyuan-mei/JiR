# Reproducible workflow

1. Install PyTorch and the packages in requirements.txt.
2. Prepare paired volumes in the layout described in DATA.md.
3. Set DATA_ROOT and, when desired, LOG_ROOT and OUTPUT_ROOT.
4. Select a recipe under configs/mri2pet/ or configs/brats/.
5. Launch main.py from the repository root.

Example:

~~~bash
export DATA_ROOT=/absolute/path/to/data
export WANDB_PROJECT=JiR
python main.py --config_path mri2pet/diffusion --run_name jir_mri2pet
~~~

The configuration controls the interpolant (flow_mode), prediction space, time sampling, optional time consistency, and the number of integration steps. Setting num_sampling_steps: 1 selects the one-step deterministic path used in the paper.
