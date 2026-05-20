import os
import torch
import SimpleITK as sitk
from collections import deque
from typing import Optional, Union, List
import wandb
from lightning_fabric.wrappers import _unwrap_objects
from lightning_utilities import apply_to_collection
import numpy as np
from matplotlib import cm
from sklearn.decomposition import PCA


class TorchPCA(object):
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        self.mean_ = X.mean(dim=0)
        unbiased = X - self.mean_.unsqueeze(0)
        U, S, V = torch.pca_lowrank(unbiased, q=self.n_components, center=False, niter=4)
        self.components_ = V.T
        self.singular_values_ = S
        return self

    def transform(self, X):
        t0 = X - self.mean_.unsqueeze(0)
        projected = t0 @ self.components_.T
        return projected

def pca(feature, n_components=3, use_torch_pca=False, target_size=80):
    device = feature.device
    def flatten(tensor, target_size=target_size):
        if target_size is not None:
            tensor = torch.nn.functional.interpolate(tensor, (target_size, target_size), mode="bilinear")
        B, C, H, W = tensor.shape
        return tensor.permute(1, 0, 2, 3).reshape(C, B * H * W).permute(1, 0).detach().cpu()
    feature = feature.float()
    if use_torch_pca:
        fit_pca = TorchPCA(n_components=n_components).fit(flatten(feature))
        feature_pca = fit_pca.transform(flatten(feature))
    else:
        fit_pca = PCA(n_components=n_components)
        feature_pca = fit_pca.fit_transform(flatten(feature).cpu().numpy())
    if isinstance(feature_pca, np.ndarray):
        feature_pca = torch.from_numpy(feature_pca)
    feature_pca -= feature_pca.min(dim=0, keepdim=True).values
    feature_pca /= feature_pca.max(dim=0, keepdim=True).values
    B, C, H, W = feature.shape
    feature_pca = feature_pca.reshape(B, target_size, target_size, n_components).permute(0, 3, 1, 2).to(device)
    return feature_pca

class CheckpointCallback:
    def __init__(
        self,
        interval: int = 1,
        max_keep_ckpts: int = -1,
        save_last: bool = True,
        save_best: Optional[Union[str, List[str]]] = None,
        mode: Union[str, List[str]] = "max",
        dirpath: str = "work_dir/checkpoints",
        filename_tmpl: Optional[str] = None,
        specific_save_epoch=None,
    ):
        self.interval = interval
        self.max_keep_ckpts = max_keep_ckpts
        self.save_last = save_last
        self.save_best = (
            save_best
            if isinstance(save_best, list)
            else [save_best]
            if save_best
            else []
        )
        self.mode = mode if isinstance(mode, list) else [mode] * len(self.save_best)
        self.dirpath = dirpath
        os.makedirs(self.dirpath, exist_ok=True)
        self.filename_tmpl = filename_tmpl or "epoch_{}.ckpt"
        self.best_ckpt_path = {k: None for k in self.save_best}
        self.best_score = {
            k: -float("inf") if m == "max" else float("inf")
            for k, m in zip(self.save_best, self.mode)
        }
        self.keep_ckpt_ids = deque(
            maxlen=max_keep_ckpts if max_keep_ckpts > 0 else None
        )
        self.specific_save_epoch = specific_save_epoch

    def checkpoint_on_train_epoch_end(self, *, fabric, state):
        epoch = state["epoch"]
        if epoch % self.interval == 0:
            filename = self.filename_tmpl.format(epoch)
            self._save(fabric, filename, state, step=epoch)

        if self.specific_save_epoch is not None:
            specific_epochs = (
                self.specific_save_epoch
                if isinstance(self.specific_save_epoch, (list, tuple, set))
                else [self.specific_save_epoch]
            )
            if epoch in specific_epochs:
                self._save(
                    fabric,
                    f"specific_epoch{epoch}.ckpt",
                    {"model": state["model"]},
                    step=epoch,
                )

        if self.save_last:
            self._save(fabric, "last.ckpt", state, step=epoch)

    def checkpoint_on_validation_epoch_end(self, *, fabric, state, metrics):
        best_state = {"model": state["model"]}
        epoch = state["epoch"]

        for k in self.save_best:
            if k not in metrics:
                continue

            score = metrics[k]
            rule = self.mode[self.save_best.index(k)]
            is_better = (
                score > self.best_score[k]
                if rule == "max"
                else score < self.best_score[k]
            )

            if is_better:
                self.best_score[k] = score
                safe_key = str(k).replace("/", "_")
                filename = f"best_{safe_key}_epoch{epoch}.ckpt"
                ckpt_path = os.path.join(self.dirpath, filename)

                old_path = self.best_ckpt_path.get(k, None)

                if fabric.is_global_zero:
                    if old_path and os.path.exists(old_path) and old_path != ckpt_path:
                        try:
                            os.remove(old_path)
                            print(
                                f"[CheckpointCallback] Removed old best checkpoint for {k}: {old_path}"
                            )
                        except Exception as e:
                            print(
                                f"[CheckpointCallback] Failed to remove old checkpoint for {k}: {e}"
                            )
                self.best_ckpt_path[k] = ckpt_path
                self._save(fabric, filename, best_state, step=epoch)

    def _save(self, fabric, filename, state, step):
        ckpt_path = os.path.join(self.dirpath, filename)
        state = state or {}

        # fabric.save(ckpt_path, state)
        fabric._strategy.save_checkpoint(path=ckpt_path, state=_unwrap_objects(state))

        if not fabric.is_global_zero:
            return

        print(f"[CheckpointCallback] Saved checkpoint: {ckpt_path}")

        if self.max_keep_ckpts > 0 and not filename.startswith(
            ("last", "best", "specific")
        ):
            try:
                if len(self.keep_ckpt_ids) == self.keep_ckpt_ids.maxlen:
                    old_step = self.keep_ckpt_ids[0]
                    old_path = os.path.join(
                        self.dirpath, self.filename_tmpl.format(old_step)
                    )
                    if os.path.exists(old_path):
                        os.remove(old_path)
                        print(
                            f"[CheckpointCallback] Removed old checkpoint: {old_path}"
                        )
            except Exception as e:
                print(f"[CheckpointCallback] Failed to remove old checkpoint: {e}")
            finally:
                self.keep_ckpt_ids.append(step)


class VisualizationCallback:
    def __init__(
        self,
        output_dir="work_dir/outputs",
        epoch_interval=10,
        batch_interval=10,
        save_input=False,
        dimension=3,
        save_target=False,
        log_to_wandb=True,
        logger=None,
        compress=True,
    ):
        self.save_dir = output_dir
        self.batch_interval = batch_interval
        self.epoch_interval = epoch_interval
        self.save_input = save_input
        self.save_target = save_target
        self.log_to_wandb = log_to_wandb
        self.logger = logger
        self.compress = compress
        self.dimension = dimension

    def norm_(self, img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float32)
        img = (img - np.min(img)) / (
            np.max(img) - np.min(img) + 1e-8
        )  # normalize to [0,1]
        img = (img * 255).clip(0, 255).astype(np.uint8)  # convert to uint8
        return img

    def _wandb_log_images(self, key, images, captions, epoch):
        """Log image batches with epoch as the x-axis when possible."""
        if (
            hasattr(self.logger, "experiment")
            and isinstance(self.logger.experiment, wandb.wandb_sdk.wandb_run.Run)
        ):
            wandb.log({
                "epoch/epochs": epoch,
                key: [wandb.Image(img, caption=cap) for img, cap in zip(images, captions)],
            })
        else:
            # Fallback for loggers that expose a Lightning-style image API.
            self.logger.log_image(
                key=key, images=images, caption=captions, step=epoch
            )

    def _extract_slice(self, tensor, n_samples):
        """Extract middle slice from 3D or center frame from 2D tensor"""
        if self.dimension == 3:
            mid = tensor.shape[2] // 2
            slices = tensor[:n_samples, :, mid].detach().cpu().squeeze().numpy()
        else:
            slices = tensor[:n_samples].detach().cpu().squeeze().numpy()

        if n_samples == 1:
            return [slices]
        return [slices[i] for i in range(len(slices))]

    def _to_rgb(self, tensor, n_samples):
        """Convert 3-channel spatial state to RGB"""
        if self.dimension == 3:
            mid = tensor.shape[2] // 2
            slices = tensor[:n_samples, :, mid]
            if slices.shape[1] > 3:
                rgb = pca(slices, 3).detach().cpu().numpy()
            else:
                rgb = slices.detach().cpu().numpy()
        else:
            rgb = tensor[:n_samples, :3].detach().cpu().numpy()

        # Normalize each channel to [0, 255]
        rgb_norm = []
        for i in range(len(rgb)):
            img = rgb[i].transpose(1, 2, 0)  # C,H,W -> H,W,C
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img = (img * 255).clip(0, 255).astype(np.uint8)
            rgb_norm.append(img)
        return rgb_norm

    def vis_on_train_batch_end(self, epoch, batch, batch_idx, is_global_zero=False, n=4):
        if not is_global_zero or batch_idx != 0:
            return
        if epoch % self.epoch_interval != 0 or not self.log_to_wandb:
            return

        n = min(n, batch["input"].shape[0])
        captions = [f"epoch{epoch}_{i}" for i in batch["id"][:n]]

        input_images = [self.norm_(img) for img in self._extract_slice(batch["input"], n)]
        target_images = [self.norm_(img) for img in self._extract_slice(batch["target"], n)]
        self._wandb_log_images("media_train/input", input_images, captions, epoch)
        self._wandb_log_images("media_train/target", target_images, captions, epoch)

        if batch.get("prediction") is not None:
            output_images = [self.norm_(img) for img in self._extract_slice(batch["prediction"], n)]
            self._wandb_log_images("media_train/prediction", output_images, captions, epoch)

    def _to_colormap(self, img_2d, cmap_name="turbo"):
        img_norm = self.norm_(img_2d)
        rgb = (cm.get_cmap(cmap_name)(img_norm)[..., :3] * 255).astype(np.uint8)
        return rgb

    def _compute_diff_map(self, pred, target, method="percentile", enhance_factor=2.0):
        diff = np.abs(pred - target)
        if method == "percentile":
            # Use 95th percentile for robust normalization
            diff_p95 = np.percentile(diff, 95)
            diff_enh = np.clip(diff / (diff_p95 + 1e-8), 0.0, 1.0)
        elif method == "log":
            # Log enhancement preserves small error details
            diff_enh = np.log1p(diff * 10) / np.log1p(10)
        else:  # adaptive
            # Simple min-max normalization with enhancement
            diff_enh = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
            diff_enh = np.clip(diff_enh * enhance_factor, 0.0, 1.0)
        return diff_enh

    def vis_on_validation_batch_end(
        self, epoch, batch, batch_idx, is_global_zero=False, save=True, exist_gt=True
    ):
        batch_np = apply_to_collection(
            batch,
            dtype=torch.Tensor,
            function=lambda t: t.detach().cpu().squeeze().numpy(),
        )
        id_ = batch["id"][0]
        save_dir = self.save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Save predictions if available
        if save:
            def write_itk(array, name, dtype):
                img = sitk.Cast(sitk.GetImageFromArray(array), dtype)
                out_path = os.path.join(save_dir, f"{name}_{id_}.nii.gz")
                sitk.WriteImage(img, out_path)

            if "prediction" in batch_np:
                write_itk(batch_np["prediction"], "syn", sitk.sitkFloat32)

            if self.save_input:
                write_itk(batch_np["input"], "input", sitk.sitkFloat32)

            if self.save_target and exist_gt:
                write_itk(batch_np["target"], "target", sitk.sitkFloat32)

            print(f"[VisualizationCallback] Saved: {save_dir}/syn_{id_}.nii.gz")

        # WandB visualization
        if not self.log_to_wandb or not is_global_zero:
            return

        mid = batch_np["input"].shape[0] // 2
        captions = [f"{id_}_epoch{epoch}"]

        # input
        input_img = [self.norm_(batch_np["input"][mid])]
        self._wandb_log_images(f"media_val/{id_}input", input_img, captions, epoch)

        # target
        if exist_gt:
            target_img = [self.norm_(batch_np["target"][mid])]
            target_color = [self._to_colormap(batch_np["target"][mid])]  # jet by default
            self._wandb_log_images(f"media_val/{id_}_target", target_img, captions, epoch)
            self._wandb_log_images(f"media_val/{id_}_target_color", target_color, captions, epoch)

        # Standard prediction
        if "prediction" in batch_np:
            pred_img = [self.norm_(batch_np["prediction"][mid])]
            pred_color = [self._to_colormap(batch_np["prediction"][mid])]  # jet by default
            self._wandb_log_images(f"media_val/{id_}_prediction", pred_img, captions, epoch)
            self._wandb_log_images(f"media_val/{id_}_prediction_color", pred_color, captions, epoch)

            # Diff map with percentile normalization
            if exist_gt:
                diff_enh = self._compute_diff_map(
                    batch_np["prediction"][mid],
                    batch_np["target"][mid],
                    method="percentile"
                )
                diff_rgb = [self._to_colormap(diff_enh, "hot")]  # hot for error map
                self._wandb_log_images(f"media_val/{id_}_diffmap", diff_rgb, captions, epoch)
