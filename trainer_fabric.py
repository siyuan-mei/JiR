import os
from collections import defaultdict
from collections.abc import Mapping
from functools import partial
from typing import Any, Iterable, List, Literal, Optional, Tuple, Union, cast
from lightning.pytorch.utilities.model_summary import ModelSummary
try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
import numpy as np
import lightning as L
import torch
from lightning.fabric.accelerators import Accelerator
from lightning.fabric.loggers import Logger
from lightning.fabric.strategies import Strategy
from lightning_utilities import apply_to_collection
from tqdm import tqdm
import json


class Trainer:
    def __init__(
        self,
        config,
        workdir,
        model,
        accelerator: Union[str, Accelerator] = "auto",
        strategy: Union[str, Strategy] = "auto",
        devices: Union[List[int], str, int] = "auto",
        precision: Union[str, int] = "32-true",
        loggers: Optional[Union[Logger, List[Logger]]] = None,
        callbacks: Optional[Union[List[Any], Any]] = None,
        grad_accum_steps: int = 1,
        use_distributed_sampler: bool = True,
        checkpoint_dir: str = "./checkpoints",
    ) -> None:
        """Trainer with Fabric.

        Args:
            accelerator: The hardware to run on. Possible choices are:
                ``"cpu"``, ``"cuda"``, ``"mps"``, ``"gpu"``, ``"tpu"``, ``"auto"``.
            strategy: Strategy for how to run across multiple devices. Possible choices are:
                ``"dp"``, ``"ddp"``, ``"ddp_spawn"``, ``"deepspeed"``, ``"fsdp"``.
            devices: Number of devices to train on (``int``),
                which GPUs to train on (``list`` or ``str``), or ``"auto"``.
                The value applies per node.
            precision: Double precision (``"64"``), full precision (``"32"``), half precision AMP (``"16-mixed"``),
                or bfloat16 precision AMP (``"bf16-mixed"``).
            loggers: A single logger or a list of loggers. See :meth:`~lightning.fabric.fabric.Fabric.log` for more
                information.
            callbacks: A single callback or a list of callbacks. The following hooks are supported:
                - on_train_epoch_start
                - on train_epoch_end
                - on_train_batch_start
                - on_train_batch_end
                - on_before_backward
                - on_after_backward
                - on_before_zero_grad
                - on_before_optimizer_step
                - on_validation_model_eval
                - on_validation_model_train
                - on_validation_epoch_start
                - on_validation_epoch_end
                - on_validation_batch_start
                - on_validation_batch_end
            grad_accum_steps: How many batches to process before each optimizer step
            limit_train_batches: Limits the number of train batches per epoch
                If greater than number of batches in the dataloader, this has no effect.
            checkpoint_dir: Directory to store checkpoints to.

        """
        self.config = config
        self.fabric = L.Fabric(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            precision=precision,
            callbacks=callbacks,
            loggers=loggers,
        )
        # if self.config.train_cfg.devices in [1, "auto"]:
        #     print("self.config.train_cfg.devices", self.config.train_cfg.devices)
        #     self.fabric.launch()

        self.wandb_logger = loggers
        self.grad_accum_steps: int = grad_accum_steps
        self._epoch_loss_accumulator = defaultdict(float)

        self.max_epochs = config.train_cfg.num_epochs
        self.max_steps = None
        self.should_stop = False

        self.workdir = workdir
        self.model = model
        self.global_step = 0
        self.current_epoch = 0
        # ensures limit_X_batches is either int or inf
        limit_train_batches = config.train_cfg.limit_train_batches
        limit_val_batches = config.val_cfg.limit_val_batches
        # self.full_val_epoch = self.config.val_cfg.get(
        #     "full_val_epoch", [self.max_epochs]
        # )
        if not isinstance(limit_train_batches, int):
            assert limit_train_batches == float("inf")

        if not isinstance(limit_val_batches, int):
            assert limit_val_batches == float("inf")

        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches

        self.start_val = config.val_cfg.start_val
        self.validation_frequency = config.val_cfg.val_freq
        self.use_distributed_sampler = use_distributed_sampler
        self._current_train_return: Union[torch.Tensor, Mapping[str, Any]] = {}
        self._current_val_return: Optional[Union[torch.Tensor, Mapping[str, Any]]] = {}
        self.checkpoint_dir = checkpoint_dir
        self.grad_clip = config.optim_cfg.grad_clip


    def _init_wandb_metrics(self):
        """Define W&B metrics safely (after Fabric logger initialized)."""
        if not self.fabric.is_global_zero:
            return
        try:
            wandb_run = self.fabric.logger.experiment

            import time
            for _ in range(30):
                if getattr(wandb_run, "step", None) is not None:
                    break
                time.sleep(0.2)

            wandb_run.define_metric("epoch/epochs")
            for prefix in ["train/*", "validation/*", "epoch/*"]:
                wandb_run.define_metric(prefix, step_metric="epoch/epochs")

            wandb_run.log({"epoch/epochs": 0})
            print("[W&B] Metrics initialized with epoch as x-axis.")
        except Exception as e:
            print(f"[W&B] Metric initialization skipped: {e}")

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        seed=42,
        resume=False,
    ):
        train_dir = os.path.join(self.workdir, "train")
        os.makedirs(train_dir, exist_ok=True)
        self.fabric.seed_everything(seed + self.fabric.global_rank, workers=True)

        self._init_wandb_metrics()

        # setup dataloaders
        train_loader = self.fabric.setup_dataloaders(
            train_loader, use_distributed_sampler=self.use_distributed_sampler
        )
        if val_loader is not None:
            val_loader = self.fabric.setup_dataloaders(
                val_loader, use_distributed_sampler=self.use_distributed_sampler
            )

        # setup models and optimizer
        num_steps = len(train_loader)
        optimizer, scheduler_cfg = self._parse_optimizers_schedulers(
            self.model.configure_optimizers_schedulers(num_steps)
        )
        model, optimizer = self.fabric.setup(self.model, optimizer)

        # model.register_forward_method(model.training_step)

        self.wandb_logger.watch(model)
        assert optimizer is not None

        # assemble state (current epoch and global step will be added in save)
        state = {
            "model": model,
            "optim": optimizer,
            "scheduler": scheduler_cfg,
            "epoch": self.current_epoch,
            "step": self.global_step,
        }

        # load last checkpoint if available
        if resume and os.path.isdir(self.checkpoint_dir):
            latest_checkpoint_path = self.get_latest_checkpoint(self.checkpoint_dir)
            if latest_checkpoint_path is not None:
                self.load(state, latest_checkpoint_path)
                # check if we even need to train here
                if (
                    self.max_epochs is not None
                    and self.current_epoch >= self.max_epochs
                ):
                    self.should_stop = True

        while not self.should_stop:
            model.current_epoch = self.current_epoch
            train_epoch_loss = self.train_loop(
                model, optimizer, train_loader, self.limit_train_batches, scheduler_cfg
            )
            state.update(step=self.global_step, epoch=self.current_epoch)
            if self.should_validate:
                # if self.current_epoch in self.full_val_epoch:
                #     limit_batches = float("inf")
                # else:
                #     limit_batches = self.limit_val_batches

                self.val_loop(model, val_loader, limit_batches=self.limit_val_batches)
                self.fabric.call(
                    "checkpoint_on_validation_epoch_end",
                    fabric=self.fabric,
                    state=state,
                    metrics=self._current_val_return,
                )
            self.fabric.call(
                "checkpoint_on_train_epoch_end", state=state, fabric=self.fabric
            )
            self.step_scheduler(
                scheduler_cfg, level="epoch", current_value=self.current_epoch
            )

            self.fabric.log_dict(
                {
                    "epoch/epochs": self.current_epoch,
                    "epoch/lr": optimizer.param_groups[0]["lr"],
                    **train_epoch_loss,
                },
                step=self.current_epoch,
            )

            # stopping condition on epoch level
            if self.max_epochs is not None and self.current_epoch >= self.max_epochs:
                self.should_stop = True

            self.current_epoch += 1

        # reset for next fit call
        self.should_stop = False

    def evaluate(
        self, test_loader, short_ckpt_path=None, full_ckpt_path=None, exist_gt=False
    ):
        eval_dir = os.path.join(self.workdir, "eval")
        os.makedirs(eval_dir, exist_ok=True)
        # setup dataloaders
        test_loader = self.fabric.setup_dataloaders(
            test_loader, use_distributed_sampler=self.use_distributed_sampler
        )

        # model.register_forward_method(model.training_step)
        model = self.fabric.setup(
            self.model,
        )
        self.wandb_logger.watch(model)

        # assemble state (current epoch and global step will be added in save)
        state = {"model": model}

        assert short_ckpt_path is not None or full_ckpt_path is not None

        if full_ckpt_path:
            ckpt_path = full_ckpt_path
        else:
            ckpt_path = os.path.join(self.checkpoint_dir, short_ckpt_path)

        print("ckpt_path: ", ckpt_path)

        self.fabric.load(ckpt_path, state, strict=False)

        if not exist_gt:
            self.test_loop(model, test_loader)
        else:
            self.val_loop(model, test_loader)
            with open(os.path.join(eval_dir, "results.txt"), "w") as f:
                json.dump(self._current_val_return, f, indent=4)

    def train_loop(
        self,
        model: L.LightningModule,
        optimizer: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        limit_batches: Union[int, float] = float("inf"),
        scheduler_cfg: Optional[
            Mapping[str, Union[LRScheduler, bool, str, int]]
        ] = None,
    ):
        iterable = self.progbar_wrapper(
            train_loader,
            total=min(len(train_loader), limit_batches),
            desc=f"Epoch {self.current_epoch}",
        )

        epoch_loss_accumulator = defaultdict(float)

        for batch_idx, batch in enumerate(iterable):
            # end epoch if stopping training completely or max batches for this epoch reached
            if self.should_stop or batch_idx >= limit_batches:
                break

            # check if optimizer should step in gradient accumulation
            should_optim_step = self.global_step % self.grad_accum_steps == 0
            if should_optim_step:
                # currently only supports a single optimizer
                # optimizer step runs train step internally through closure
                # optimizer.step(partial(self.training_step, model=model, batch=batch, batch_idx=batch_idx))
                self.training_step(model, batch, batch_idx)
                # self.fabric.backward(loss)
                optimizer.step()

                if getattr(model, "use_ema", False):
                    model.update_ema()

                optimizer.zero_grad()
            else:
                # gradient accumulation -> no optimizer step
                self.training_step(model=model, batch=batch, batch_idx=batch_idx)

            # this guard ensures, we only step the scheduler once per global step
            if should_optim_step:
                self.step_scheduler(
                    scheduler_cfg, level="step", current_value=self.global_step
                )

            for k, v in self._current_train_return.items():
                epoch_loss_accumulator[k] += float(v)
            # add output values to progress bar
            self._format_iterable(iterable, self._current_train_return, "train")

            self.fabric.log_dict(
                {
                    "step/steps": self.global_step,
                    "step/lr": optimizer.param_groups[0]["lr"],
                },
                step=self.global_step,
            )

            self.fabric.log_dict(
                {f"step/{k}": v for k, v in self._current_train_return.items()},
                step=self.global_step,
            )

            # only increase global step if optimizer stepped
            self.global_step += int(should_optim_step)

            # stopping criterion on step level
            if self.max_steps is not None and self.global_step >= self.max_steps:
                self.should_stop = True
                break

        avg_epoch_losses = {
            f"epoch/{k}": v / len(train_loader)
            for k, v in epoch_loss_accumulator.items()
        }
        return avg_epoch_losses

    def training_step(
        self, model: L.LightningModule, batch: Any, batch_idx: int
    ) -> dict:
        """A single training step, running forward and backward. The optimizer step is called separately, as this is
        given as a closure to the optimizer step.

        Args:
            model: the lightning module to train
            batch: the batch to run the forward on

        """
        outputs = model.training_step(batch)

        out_batch = outputs[0]
        loss = (
            outputs[1]
            if isinstance(outputs, torch.Tensor)
            else outputs[1]["total_loss"]
        )

        self.fabric.backward(loss)
        # if self.grad_clip:
        #     self.fabric.clip_gradients(model, optimizer, max_norm=self.grad_clip)

        # avoid gradients in stored/accumulated values -> prevents potential OOM
        self._current_train_return = apply_to_collection(
            outputs[1], dtype=torch.Tensor, function=lambda x: x.detach().cpu()
        )
        self.fabric.call(
            "vis_on_train_batch_end", self.current_epoch, out_batch, batch_idx, self.fabric.is_global_zero
        )
        return loss

    def val_loop(
        self,
        model: L.LightningModule,
        val_loader: Optional[torch.utils.data.DataLoader],
        limit_batches: Union[int, float] = float("inf"),
    ):
        """The validation loop running a single validation epoch.

        Args:
            model: the LightningModule to evaluate
            val_loader: The dataloader yielding the validation batches.
            limit_batches: Limits the batches during this validation epoch.
                If greater than the number of batches in the ``val_loader``, this has no effect.

        """
        # no validation if val_loader wasn't passed
        if val_loader is None:
            return

        model.eval()
        torch.set_grad_enabled(False)

        iterable = self.progbar_wrapper(
            val_loader, total=min(len(val_loader), limit_batches), desc="Validation"
        )

        # cum_loss = 0
        cum_acc_dict = defaultdict(lambda: defaultdict(list))
        for batch_idx, batch in enumerate(iterable):
            # end epoch if stopping training completely or max batches for this epoch reached
            if self.should_stop or batch_idx >= limit_batches:
                break

            out_batch, acc_dict = model.validation_step(batch)
            # avoid gradients in stored/accumulated values -> prevents potential OOM
            acc_dict = apply_to_collection(acc_dict, torch.Tensor, lambda x: x.detach())

            self._format_iterable(iterable, acc_dict, "val")

            self.fabric.call(
                "vis_on_validation_batch_end", self.current_epoch, out_batch, batch_idx, self.fabric.is_global_zero
            )

            cls = batch.get("class", ["unknown"])[0]
            for key, value in acc_dict.items():
                cum_acc_dict[cls][key].append(value)

        self._current_val_return = {}

        for cls, metrics_dict in cum_acc_dict.items():
            for key, values in metrics_dict.items():
                values_np = np.array(values)
                mean, std = float(np.mean(values_np)), float(np.std(values_np))
                self._current_val_return[f"c_{cls}/{key}_mean"] = mean
                self._current_val_return[f"c_{cls}/{key}_std/"] = std

        # --- also log overall mean across all anatomies ---
        all_metrics = defaultdict(list)
        for metrics_dict in cum_acc_dict.values():
            for key, vals in metrics_dict.items():
                all_metrics[key].extend(vals)

        for key, vals in all_metrics.items():
            mean = float(np.mean(vals))
            self._current_val_return[f"a/{key}_mean"] = mean
            self._current_val_return[f"a/{key}_std"] = float(np.std(vals))
            self._current_val_return[key] = mean

        # --- log to wandb ---
        self.fabric.log_dict(
            {f"validation/{k}": v for k, v in self._current_val_return.items()},
            step=self.current_epoch,
        )

        model.train()
        torch.set_grad_enabled(True)

        #     for key, value in acc_dict.items():
        #         if key not in cum_acc_dict:
        #             cum_acc_dict[key] = []
        #         cum_acc_dict[key].append(value)  # append as float
        #
        # self._current_val_return = {}
        #
        # for key, values in cum_acc_dict.items():
        #     values_np = np.array(values)
        #     self._current_val_return[f"{key}_mean"] = float(np.mean(values_np))
        #     self._current_val_return[f"{key}_std"] = float(np.std(values_np))
        #
        # self.fabric.log_dict(
        #     {f"validation/{k}": v for k, v in self._current_val_return.items()},
        #     step=self.current_epoch,
        # )
        #
        # model.train()
        # torch.set_grad_enabled(True)

    def test_loop(
        self,
        model: L.LightningModule,
        test_loader: Optional[torch.utils.data.DataLoader],
    ):
        if test_loader is None:
            return
        model.eval()
        torch.set_grad_enabled(False)
        iterable = self.progbar_wrapper(
            test_loader, total=len(test_loader), desc="Test"
        )
        for batch_idx, batch in enumerate(iterable):
            out_batch = model.test_step(batch)
            self.fabric.call(
                "vis_on_validation_batch_end",
                self.current_epoch,
                out_batch,
                batch_idx,
                self.fabric.is_global_zero,
                True,
                False,
            )

    def step_scheduler(
        self,
        scheduler_cfg: Optional[
            Mapping[str, Union[LRScheduler, bool, str, int]]
        ],
        level: Literal["step", "epoch"],
        current_value: int,
    ) -> None:
        """Steps the learning rate scheduler if necessary.

        Args:
            models: The LightningModule to train
            scheduler_cfg: The learning rate scheduler configuration.
                Have a look at :meth:`lightning.pytorch.LightningModule.configure_optimizers` for supported values.
            level: whether we are trying to step on epoch- or step-level
            current_value: Holds the current_epoch if ``level==epoch``, else holds the ``global_step``

        """

        # no scheduler
        if scheduler_cfg is None:
            return
        # wrong interval (step vs. epoch)
        if scheduler_cfg["interval"] != level:
            return

        # right interval, but wrong step wrt frequency
        if current_value % cast(int, scheduler_cfg["frequency"]) != 0:
            return

        # rely on models hook for actual step
        scheduler_type = scheduler_cfg["type"]
        if scheduler_type in ["cosine", 'warmup', 'poly']:
            scheduler_cfg["scheduler"].step(self.global_step)
        elif scheduler_type == "plateau":
            scheduler_cfg["scheduler"].step(
                self.global_step, metric=self._current_val_return
            )
        else:
            raise ValueError(f"Scheduler {scheduler_type} not defined")

    @property
    def should_validate(self) -> bool:
        """Whether to currently run validation."""
        return (
            self.current_epoch % self.validation_frequency == 0
            and self.current_epoch >= self.start_val
        )

    def progbar_wrapper(self, iterable: Iterable, total: int, **kwargs: Any):
        """Wraps the iterable with tqdm for global rank zero.

        Args:
            iterable: the iterable to wrap with tqdm
            total: the total length of the iterable, necessary in case the number of batches was limited.

        """
        if self.fabric.is_global_zero:
            return tqdm(iterable, total=total, **kwargs)
        return iterable

    def load(self, state: Optional[Mapping], path: str) -> None:
        """Loads a checkpoint from a given file into state.

        Args:
            state: a mapping contaning models, optimizer and lr scheduler
            path: the path to load the checkpoint from

        """
        if state is None:
            state = {}
        remainder = self.fabric.load(path, state)
        self.global_step = state["step"]
        self.current_epoch = state["epoch"] + 1

        if remainder:
            raise RuntimeError(f"Unused Checkpoint Values: {remainder}")

    @staticmethod
    def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
        """Returns the latest checkpoint from the ``checkpoint_dir``

        Args:
            checkpoint_dir: the directory to search for checkpoints

        """
        if not os.path.isdir(checkpoint_dir):
            return None

        files = os.listdir(checkpoint_dir)
        items = [
            os.path.join(checkpoint_dir, f)
            for f in files
            if f.startswith("epoch") or f.startswith("last")
        ]

        if not items:
            return None

        # Sort by modification time
        latest = max(items, key=os.path.getmtime)
        print("resuming checkpoint from: ", latest)
        return latest

    def _parse_optimizers_schedulers(
        self, configure_optim_output
    ) -> Tuple[
        Optional[L.fabric.utilities.types.Optimizable],
        Optional[
            Mapping[str, Union[LRScheduler, bool, str, int]]
        ],
    ]:
        """Recursively parses the output of :meth:`lightning.pytorch.LightningModule.configure_optimizers`.

        Args:
            configure_optim_output: The output of ``configure_optimizers``.
                For supported values, please refer to :meth:`lightning.pytorch.LightningModule.configure_optimizers`.

        """
        _lr_sched_defaults = {
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss",
        }

        # single optimizer
        if isinstance(configure_optim_output, L.fabric.utilities.types.Optimizable):
            return configure_optim_output, None

        # single lr scheduler
        if isinstance(configure_optim_output, LRScheduler):
            return None, _lr_sched_defaults.update(scheduler=configure_optim_output)

        # single lr scheduler config
        if isinstance(configure_optim_output, Mapping):
            _lr_sched_defaults.update(configure_optim_output)
            return None, _lr_sched_defaults

        # list or tuple
        if isinstance(configure_optim_output, (list, tuple)):
            if all(
                isinstance(_opt_cand, L.fabric.utilities.types.Optimizable)
                for _opt_cand in configure_optim_output
            ):
                # single optimizer in list
                if len(configure_optim_output) == 1:
                    return configure_optim_output[0][0], None

                raise NotImplementedError("BYOT only supports a single optimizer")

            if all(
                isinstance(_lr_cand, (LRScheduler, Mapping))
                for _lr_cand in configure_optim_output
            ):
                # single scheduler in list
                if len(configure_optim_output) == 1:
                    return None, self._parse_optimizers_schedulers(
                        configure_optim_output[0]
                    )[1]

            # optimizer and lr scheduler
            elif len(configure_optim_output) == 2:
                opt_cands, lr_cands = (
                    self._parse_optimizers_schedulers(configure_optim_output[0])[0],
                    self._parse_optimizers_schedulers(configure_optim_output[1])[1],
                )
                return opt_cands, lr_cands

        return None, None

    @staticmethod
    def _format_iterable(
        prog_bar,
        candidates: Optional[
            Union[torch.Tensor, Mapping[str, Union[torch.Tensor, float, int]]]
        ],
        prefix: str,
    ):
        """Adds values as postfix string to progressbar.

        Args:
            prog_bar: a progressbar (on global rank zero) or an iterable (every other rank).
            candidates: the values to add as postfix strings to the progressbar.
            prefix: the prefix to add to each of these values.

        """
        if isinstance(prog_bar, tqdm) and candidates is not None:
            postfix_str = ""
            float_candidates = apply_to_collection(
                candidates, torch.Tensor, lambda x: x.item()
            )
            if isinstance(candidates, torch.Tensor):
                postfix_str += f" {prefix}_loss: {float_candidates:.3f}"
            elif isinstance(candidates, Mapping):
                for k, v in float_candidates.items():
                    postfix_str += f" {prefix}_{k}: {v:.3f}"

            if postfix_str:
                prog_bar.set_postfix_str(postfix_str)
