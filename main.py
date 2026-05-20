import argparse
import math
import os

import torch
import torch.backends.cudnn as cudnn
from monai.data import list_data_collate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from wandb.integration.lightning.fabric import WandbLogger

import configs
from data.dataset.brats import BratsDataset
from data.dataset.mri2pet import Mri2PetDataset
from data.transforms import build_train_pipeline, build_val_pipeline
from models.diffusion_model import DiffusionModel
from models.encoder_decoder import EncoderDecoder
from models.my_callbacks import CheckpointCallback, VisualizationCallback
from trainer_fabric import Trainer


torch.set_float32_matmul_precision("high")
cudnn.benchmark = True
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")


def scale_lr_with_devices(config, model_type=None):
    devices = config["train_cfg"]["devices"]
    if devices == "auto":
        num_devices = max(torch.cuda.device_count(), 1)
    elif isinstance(devices, (list, tuple)):
        num_devices = max(len(devices), 1)
    else:
        num_devices = max(int(devices), 1)

    scale = round(math.sqrt(num_devices), 2)
    if model_type == "GAN":
        config["optim_cfg_g"]["base_lr"] *= scale
        config["optim_cfg_d"]["base_lr"] *= scale
        print(
            f"[Info] Scaled base_lr by sqrt({num_devices}) ~= {scale}: "
            f"G_lr={config['optim_cfg_g']['base_lr']:.2e}, "
            f"D_lr={config['optim_cfg_d']['base_lr']:.2e}"
        )
    else:
        config["optim_cfg"]["base_lr"] *= scale
        print(
            f"[Info] Scaled base_lr by sqrt({num_devices}) ~= {scale}: "
            f"LR={config['optim_cfg']['base_lr']:.2e}"
        )
    return config


def resolve_data_root(config):
    data_root = os.environ.get("DATA_ROOT") or config.data_cfg.get("data_root")
    if not data_root:
        raise ValueError(
            "No data root configured. Set DATA_ROOT or data_cfg.data_root in a local config."
        )

    data_root = os.path.abspath(os.path.expanduser(str(data_root)))
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"data_root does not exist: {data_root}")
    return data_root


def build_model(config):
    model_type = str(config.model_cfg.get("model_type", "encoder_decoder")).lower()
    if model_type == "diffusion":
        return DiffusionModel(config)
    if model_type in {"normal", "encoder_decoder", "encoderdecoder"}:
        return EncoderDecoder(config)

    raise ValueError(
        f"Unsupported model_type '{config.model_cfg.get('model_type')}'. "
        "This repository currently includes 'diffusion' and 'encoder_decoder'."
    )


def build_loader_kwargs(num_workers):
    kwargs = {
        "num_workers": num_workers,
        "collate_fn": list_data_collate,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    return kwargs


def validate_gpu_config(config):
    num_available_gpus = torch.cuda.device_count()
    configured_devices = config.train_cfg.get("devices", 1)
    print(f"Detected {num_available_gpus} GPU(s).")

    if num_available_gpus == 0:
        raise RuntimeError("No CUDA GPU detected, but this entrypoint uses accelerator='gpu'.")

    if configured_devices == "auto":
        return

    if isinstance(configured_devices, int) and configured_devices > num_available_gpus:
        raise ValueError(
            f"Config train_cfg.devices={configured_devices} exceeds detected GPUs={num_available_gpus}."
        )


def build_datasets(config, data_root, train_pipeline, val_pipeline):
    dataset = config.data_cfg.dataset
    if dataset == "mri2pet":
        train_dataset = Mri2PetDataset(
            data_root=data_root,
            split_path="train",
            transform=train_pipeline,
        )
        val_dataset = Mri2PetDataset(
            data_root=data_root,
            split_path="test",
            transform=val_pipeline,
        )
    elif dataset == "brats":
        train_dataset = BratsDataset(
            data_root=data_root,
            split_path="train",
            transform=train_pipeline,
        )
        val_dataset = BratsDataset(
            data_root=data_root,
            split_path="test",
            transform=val_pipeline,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return train_dataset, val_dataset, val_dataset


def run(
    config,
    workdir,
    run_name="run",
    if_train=True,
    if_eval=False,
    resume=False,
    short_ckpt_path=None,
    full_ckpt_path=None,
):
    """Run training and/or evaluation for one config."""
    data_root = resolve_data_root(config)
    log_root = os.environ.get("LOG_ROOT", os.path.join(workdir, "logs"))
    output_dir = os.environ.get("OUTPUT_ROOT", os.path.join(workdir, "outputs"))
    checkpoint_dir = os.path.join(workdir, "checkpoints")

    os.makedirs(log_root, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Resolved data_root path: {data_root}")
    OmegaConf.save(config, os.path.join(workdir, "config.yaml"))
    logger = WandbLogger(
        name=run_name,
        project=os.environ.get("WANDB_PROJECT", "MRI2PET"),
        save_dir=log_root,
        log_model=False,
        config={"yaml": os.path.join(workdir, "config.yaml")},
    )

    model_type = config.model_cfg.get("model_type", "encoder_decoder")
    model = build_model(config)

    if config.get("automatic_lr_rescale", True):
        config = scale_lr_with_devices(config, model_type)

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log_metrics({
        "params_total_M": n_total / 1e6,
        "params_trainable_M": n_trainable / 1e6,
    })

    train_pipeline = build_train_pipeline(
        crop_size=config.data_cfg.crop_size,
        resize_size=config.data_cfg.resize_size,
    )
    val_pipeline = build_val_pipeline(
        resize_size=config.data_cfg.resize_size,
        crop_size=config.data_cfg.crop_size,
    )
    train_dataset, val_dataset, test_dataset = build_datasets(
        config,
        data_root,
        train_pipeline,
        val_pipeline,
    )

    batch_size = config.train_cfg.batch_size
    num_workers = config.train_cfg.num_workers
    loader_kwargs = build_loader_kwargs(num_workers)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = val_loader

    print("train set: ", len(train_dataset))
    print("val set: ", len(val_dataset))
    print("test_set: ", len(test_dataset))

    checkpoint_callback = CheckpointCallback(
        interval=config.train_cfg.save_freq,
        save_last=False,
        max_keep_ckpts=1,
        save_best=["mse", "psnr", "ssim"],
        mode=["min", "max", "max"],
        dirpath=checkpoint_dir,
        specific_save_epoch=config.val_cfg.full_val_epoch,
    )

    vis_callback = VisualizationCallback(
        output_dir=output_dir,
        epoch_interval=50,
        batch_interval=20,
        save_input=True,
        save_target=True,
        logger=logger,
        dimension=config.data_cfg.get("dimension", 3),
    )

    validate_gpu_config(config)
    trainer = Trainer(
        config,
        workdir,
        model,
        strategy=config.train_cfg.strategy,
        devices=config.train_cfg.devices,
        precision=config.train_cfg.precision,
        callbacks=[checkpoint_callback, vis_callback],
        accelerator="gpu",
        loggers=logger,
        checkpoint_dir=checkpoint_dir,
    )

    if if_train:
        trainer.fit(train_loader, val_loader, resume=resume)
    if if_eval:
        trainer.evaluate(
            test_loader, short_ckpt_path=short_ckpt_path, full_ckpt_path=full_ckpt_path
        )
    logger.experiment.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="",
        help="Path to the config YAML file (optional)",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="default_cfg",
        help="Wandb run name (e.g., mri2pet/exp1)",
    )
    parser.add_argument(
        "--if_train",
        dest="if_train",
        action="store_false",
        help="Disable training (kept for backward compatibility).",
    )
    parser.add_argument("--no_train", dest="if_train", action="store_false")
    parser.add_argument("--if_eval", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--short_ckpt_path", type=str, default=None)
    parser.add_argument("--full_ckpt_path", type=str, default=None)
    args = parser.parse_args()

    if args.config_path:
        config = configs.load_and_merge_config(args.config_path)
    else:
        config = configs.default_config

    work_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"work_dir/{config.data_cfg.dataset}/{args.run_name}",
    )

    run(
        config,
        workdir=work_dir,
        run_name=f"{config.data_cfg.dataset}/{args.run_name}",
        if_train=args.if_train,
        if_eval=args.if_eval,
        resume=args.resume,
        short_ckpt_path=args.short_ckpt_path,
        full_ckpt_path=args.full_ckpt_path,
    )
