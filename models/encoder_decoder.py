import torch
from torch import Tensor
from data.utils import revert_rescale
from models.losses import get_loss_fn
from evaluation.image_metrics import evaluate
import lightning as L
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.plateau_lr import PlateauLRScheduler
from timm.scheduler.poly_lr import PolyLRScheduler
from timm.optim import create_optimizer
from types import SimpleNamespace
from .register import build_net
from monai.inferers import SlidingWindowInferer



class EncoderDecoder(L.LightningModule):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.config = config
        self.init_net(config)
        self.patch_based = config.data_cfg.patch_based
        self._current_epoch = 0
        if self.patch_based:
            self.inferer = SlidingWindowInferer(
                roi_size=config.patch_size,
                sw_batch_size=1,
                overlap=0.5
            )

    def init_net(self, config):
        net_cfg = config.model_cfg.get("net", None)
        self.net = build_net(net_cfg)
        self.loss_fn = get_loss_fn(config.train_cfg.loss)
        self.loss_weights = config.train_cfg.loss_weights

    def forward(self, inputs: Tensor):
        x = self.net(inputs)
        return x

    def compute_loss(self, pred, target):
        loss_items = {}
        for name, fn in self.loss_fn.items():
            if "perceptual" in name:
                perceptual_fn = getattr(
                    fn, "loss", fn
                )  # support if wrapped in MaskedLoss or similar
                if not hasattr(perceptual_fn, "_moved"):
                    perceptual_fn.to(pred.device)
                    perceptual_fn._moved = True
            loss_val = fn(pred, target)
            loss_items[name] = loss_val
        total_loss = sum(
            self.loss_weights.get(name, 1.0) * loss_items[name]
            for name in self.loss_fn
        )
        loss_items["total_loss"] = total_loss
        return loss_items

    def training_step(self, batch):
        x = batch["input"]
        y = batch["target"]
        y_hat = self(x)
        batch.update({"prediction": y_hat})
        loss_items = self.compute_loss(y_hat, y)
        return batch, loss_items

    def validation_step(self, batch):
        x = batch["input"]
        y = batch["target"]

        if self.patch_based:
            y_hat = self.inferer(x, self.forward)
        else:
            y_hat = self(x)
        # if self.config.data_cfg.rescale:
        #     y_hat = revert_rescale(y_hat)
        y_hat = torch.clamp(y_hat, min=0., max=1.)
        batch.update({"prediction": y_hat})
        val_metrics = evaluate(y_hat, y)
        return batch, val_metrics

    def configure_optimizers_schedulers(self, num_steps):
        args = SimpleNamespace()
        args.betas = getattr(self.config.optim_cfg, "betas", (0.9, 0.999))
        args.weight_decay = getattr(self.config.optim_cfg, "weight_decay", 0.05)
        args.lr = getattr(self.config.optim_cfg, "base_lr", 1e-3)
        args.opt = getattr(self.config.optim_cfg, "optimizer", "adamw")
        args.eps = getattr(self.config.optim_cfg, "eps", 1e-8)
        args.layer_decay = getattr(self.config.optim_cfg, "layer_decay", None)
        args.momentum = 0.9
        args.param_group_fn = None
        optimizer = create_optimizer(args=args, model=self)
        scheduler_cfg = param_scheduler(
            self.config.optim_cfg,
            optimizer,
            num_steps,
            self.config.train_cfg.num_epochs,
        )
        return optimizer, scheduler_cfg

    @property
    def current_epoch(self) -> int:
        return self._current_epoch

    @current_epoch.setter
    def current_epoch(self, epoch: int) -> None:
        self._current_epoch = int(epoch)



def param_scheduler(optim_cfg, optimizer, num_steps, num_epochs):
    start_factor = getattr(optim_cfg, "start_factor", 0.001)
    warmup_lr_init = optim_cfg.base_lr * start_factor
    lr_min = optim_cfg.lr_min
    scheduler_type = optim_cfg.scheduler
    warmup_epochs = getattr(optim_cfg, "warmup_epochs", 0)
    warmup_t = warmup_epochs * num_steps
    t_initial = num_steps * num_epochs
    if scheduler_type is not None:
        if scheduler_type == "plateau":
            scheduler = PlateauLRScheduler(
                optimizer,
                mode="min",
                decay_rate=0.1,
                patience_t=10,
                lr_min=optim_cfg.lr_min,
                warmup_t=warmup_t,
                warmup_lr_init=warmup_lr_init,
            )
            monitor = "val/mae"
            interval = "epoch"
            frequency = 1
        elif scheduler_type == "cosine":
            scheduler = CosineLRScheduler(
                optimizer,
                t_initial=t_initial,
                lr_min=lr_min,
                warmup_t=warmup_t,
                warmup_lr_init=warmup_lr_init,
                warmup_prefix=True,
                k_decay=1.0,
            )
            monitor = None
            interval = "step"
            frequency = 1
        elif scheduler_type == "poly":
            scheduler = PolyLRScheduler(
                optimizer,
                power=getattr(optim_cfg, "poly_power", 1.0),
                t_initial=t_initial,
                lr_min=lr_min,
                warmup_t=warmup_t,
                warmup_lr_init=warmup_lr_init,
                warmup_prefix=True,
            )
            monitor = None
            interval = "step"
            frequency = 1
        elif scheduler_type == "warmup":
            scheduler = PolyLRScheduler(
                optimizer,
                power=0,
                t_initial=t_initial,
                lr_min=lr_min,
                warmup_t=warmup_t,
                warmup_lr_init=warmup_lr_init,
                warmup_prefix=True,
            )
            monitor = None
            interval = "step"
            frequency = 1
        scheduler_cfg = {
            "type": scheduler_type,
            "scheduler": scheduler,
            "monitor": monitor,
            "interval": interval,
            "frequency": frequency,
        }

    else:
        scheduler_cfg = None
    return scheduler_cfg
