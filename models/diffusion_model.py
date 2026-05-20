import torch
from data.utils import revert_rescale, rescale
from evaluation.image_metrics import evaluate
from .losses import get_loss_fn
from .register import build_net
from .encoder_decoder import EncoderDecoder
import copy


class DiffusionModel(EncoderDecoder):
    def __init__(self, config):
        super().__init__(config)

    def init_net(self, config):
        self.net = build_net(config.model_cfg.net)
        self.loss_fn = get_loss_fn(config.train_cfg.loss)
        self.loss_weights = config.train_cfg.loss_weights

        args = config.model_cfg.diffusion

        self.use_img_cond = args.use_img_cond
        self.flow_mode = args.flow_mode

        self.use_t_consistency = args.use_t_consistency
        self.lambda_t = args.lambda_t
        self.prediction_space = args.prediction_space
        self.loss_space = args.loss_space

        self.t_weighted = args.t_weighted
        self.t_scale = args.t_scale

        self.label_drop_prob = args.label_drop_prob
        self.condition_drop_prob = args.condition_drop_prob
        self.noise_scale = args.noise_scale
        self.drift_scale = args.drift_scale
        self.num_classes = args.class_num

        # t sampling
        self.if_regression = args.if_regression
        self.t_sample = args.t_sample
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.zero_t_prob = args.zero_t_prob
        self.one_t_prob = args.one_t_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps

        # ema
        self.use_ema = args.use_ema
        self.ema_decay = args.ema_decay
        if self.use_ema:
            self.net_ema = copy.deepcopy(self.net)
            self.net_ema.eval()
            for p in self.net_ema.parameters():
                p.requires_grad_(False)

        # generation hyper params
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.use_cfg = args.use_cfg
        self.cfg_scale = args.cfg_scale
        self.cfg_interval = (args.interval_min, args.interval_max)

    def drop_labels(self, labels):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        return out

    def drop_conditions(self, img_cond):
        drop = torch.rand(img_cond.size(0), device=img_cond.device) < self.condition_drop_prob
        out = img_cond.clone()
        out[drop] = 0
        return out

    def sample_t(self, n: int, device=None):
        if self.if_regression:
            return torch.zeros(n, device=device)
        else:
            if self.t_sample == "uniform":
                t = torch.rand(n, device=device)
            elif self.t_sample == "beta":
                t = torch.distributions.Beta(self.beta1, self.beta2).sample((n,)).to(device)
            else:
                t = torch.randn(n, device=device) * self.P_std + self.P_mean
                t = torch.sigmoid(t)

            if self.zero_t_prob > 0:
                mask = torch.rand(n, device=device) < self.zero_t_prob
                t = torch.where(mask, torch.zeros_like(t), t)

            if self.one_t_prob > 0:
                mask = torch.rand(n, device=device) < self.one_t_prob
                t = torch.where(mask, torch.ones_like(t), t)

            return t


    def forward(self, x, img_cond, t, y, if_train=False):
        if if_train:
            if y is not None and self.label_drop_prob > 0:
                y = self.drop_labels(y)

            if img_cond is not None and self.condition_drop_prob > 0:
                img_cond = self.drop_conditions(img_cond)

        x_pred = self.net(x, img_cond, t.flatten(), y)
        return x_pred

    def compute_loss(self, pred, x1, x0, x_t, n, t):
        if self.prediction_space == "x":
            if self.loss_space == "v":
                pred = pred / (1 - t).clamp_min(self.t_eps)
                target = x1 / (1 - t).clamp_min(self.t_eps)
            elif self.loss_space == "x":
                target = x1
            else:
                raise NotImplementedError(f"Unsupported loss_space: {self.loss_space}")

        elif self.prediction_space == "v":
            if self.flow_mode.lower() == "nif":
                target = x1 - n
            elif self.flow_mode.lower() == "pif":
                target = x1 - x0
            elif self.flow_mode.lower() == "bif":
                sigma_t = torch.sqrt(self.drift_scale * t / (1 - t).clamp_min(self.t_eps))
                target = (x1 - x0) - sigma_t * n
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError(f"Unsupported prediction_space: {self.prediction_space}")

        loss_items = {}
        # for name, fn in self.loss_fn.items():
        #     if "perceptual" in name:
        #         perceptual_fn = getattr(
        #             fn, "loss", fn
        #         )  # support if wrapped in MaskedLoss or similar
        #         if not hasattr(perceptual_fn, "_moved"):
        #             perceptual_fn.to(pred.device)
        #             perceptual_fn._moved = True
        #     loss_val = fn(pred, target)
        #
        #     if self.t_weighted:
        #         loss_val = loss_val * (1 - t).clamp_min(self.t_eps)
        #
        #     loss_items[name] = loss_val
        # total_loss = sum(
        #     self.loss_weights.get(name, 1.0) * loss_items[name]
        #     for name in self.loss_fn
        #
        loss = torch.pow(pred - target, 2).mean(dim=(1, 2, 3, 4))
        if self.t_weighted:
            w = 1 / (4 * t * (1 - t)).clamp_min(self.t_eps)
            w = w.view(w.size(0))
            loss = w * loss
        loss_items["mse"] = loss.mean()
        loss_items["total_loss"] = loss_items["mse"]
        return loss_items

    def training_step(self, batch):
        x0 = rescale(batch["input"])
        x1 = rescale(batch["target"])
        y = batch["cls_code"]

        e = torch.randn_like(x1) * self.noise_scale
        t = self.sample_t(x1.size(0), device=x1.device).view(-1, *([1] * (x1.ndim - 1)))

        if self.flow_mode.lower() == "nif":
            x_t = t * x1 + (1 - t) * e
        elif self.flow_mode.lower() == "pif":
            x_t = t * x1 + (1 - t) * x0
        elif self.flow_mode.lower() == "bif":
            x_t = t * x1 + (1 - t) * x0 + torch.sqrt(self.drift_scale * t * (1 - t)) * e
        else:
            raise NotImplementedError

        # if not self.mixed_batch:
        pred = self.forward(x=x_t,
                            img_cond=x0 if self.use_img_cond else None,
                            t=t.flatten(),
                            y=y,
                            if_train=True)
        batch.update({"prediction": pred})
        loss_items = self.compute_loss(pred,
                                 x1=x1,
                                 x0=x0,
                                 x_t=x_t,
                                 n=e,
                                 t=t)

        if self.use_t_consistency:
            t1 = torch.ones_like(t)

            if self.use_ema:
                anchor = self.forward_ema(
                    x=x1,
                    img_cond=x0 if self.use_img_cond else None,
                    t=t1,
                    y=y
                )
            else:
                anchor = self.forward(
                    x=x1,
                    img_cond=x0 if self.use_img_cond else None,
                    t=t1.flatten(),
                    y=y,
                    if_train=False
                )

            t_reg_loss = torch.pow(pred - anchor.detach(), 2).mean(dim=(1, 2, 3, 4))
            # t_reg_loss = (1 / t.clamp_min(self.t_eps)) * t_reg_loss
            # if self.t_weighted:
            #     w = (1 - self.t_scale * t).clamp_min(self.t_eps)
            #     w = w.view(w.size(0))  # [B]
            #     t_reg_loss = w * t_reg_loss
            loss_items["t_reg"] = t_reg_loss.mean()
            loss_items["total_loss"] = loss_items["total_loss"] + self.lambda_t * loss_items["t_reg"]

        return batch, loss_items

    def validation_step(self, batch):
        x0 = rescale(batch["input"])
        x1 = batch["target"]
        y = batch["cls_code"]

        device = x0.device
        bsz = x0.size(0)

        if self.flow_mode.lower() == "nif":
            z = self.noise_scale * torch.randn_like(x0)
        else:
            z = x0.clone()

        timesteps = torch.linspace(0.0, 1.0, self.steps + 1, device=device)
        timesteps = timesteps.view(-1, 1, *([1] * (z.ndim - 1))).expand(-1, bsz, *([1] * (z.ndim - 1)))

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        # ode
        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next, y, img_cond=x0 if self.use_img_cond else None)
        # last step euler
        sample = self._euler_step(z, timesteps[-2], timesteps[-1], y, img_cond=x0 if self.use_img_cond else None)
        sample = revert_rescale(sample)

        sample = torch.clamp(sample, min=0., max=1.)

        batch.update({"prediction": sample})
        val_metrics = evaluate(sample, x1)
        return batch, val_metrics

    @torch.no_grad()
    def _forward_sample(self, z, t, labels, img_cond):
        def _get_pred_v(z, t, labels, img_cond):
            if self.patch_based:

                if img_cond is not None:
                    model_input = torch.cat([z, img_cond], dim=1)
                else:
                    model_input = z

                def model_fn(x_patch):
                    if img_cond is not None:
                        z_patch = x_patch[:, :z.shape[1]]
                        cond_patch = x_patch[:, z.shape[1]:]
                    else:
                        z_patch = x_patch
                        cond_patch = None

                    return self.forward(
                        z_patch,
                        cond_patch,
                        t.flatten(),
                        labels
                    )

                pred = self.inferer(model_input, model_fn)

            else:
                pred = self.forward(z, img_cond, t.flatten(), labels)
            # pred = self.forward(z, img_cond, t.flatten(), labels)
            if self.prediction_space == "x":
                v = (pred - z) / (1.0 - t).clamp_min(self.t_eps)
            elif self.prediction_space == "v":
                v = pred
            return v

        if not self.use_cfg:
            v_cond = _get_pred_v(z, t, labels, img_cond)
            return v_cond
        else:
            # conditional
            v_cond = _get_pred_v(z, t, labels, img_cond)
            # unconditional
            v_uncond = _get_pred_v(z, t, torch.full_like(labels, self.num_classes), torch.zeros_like(img_cond))
            # cfg interval
            low, high = self.cfg_interval
            interval_mask = (t < high) & ((low == 0) | (t > low))
            cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)

            return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels, img_cond):
        v_pred = self._forward_sample(z, t, labels, img_cond)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, labels, img_cond):
        v_pred_t = self._forward_sample(z, t, labels, img_cond)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, labels, img_cond)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next

    # @torch.no_grad()
    # def update_ema(self):
    #     source_params = list(self.parameters())
    #     for targ, src in zip(self.ema_params1, source_params):
    #         targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)

    @torch.no_grad()
    def forward_ema(self, x, img_cond, t, y):
        return self.net_ema(x, img_cond, t.flatten(), y)

    @torch.no_grad()
    def update_ema(self):
        """Call this after optimizer.step() in your trainer."""
        if not self.use_ema:
            return
        decay = self.ema_decay

        self.net_ema.to(next(self.net.parameters()).device)

        for p, p_ema in zip(self.net.parameters(), self.net_ema.parameters()):
            p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)

        # sync buffers (safe even if no BN)
        for b, b_ema in zip(self.net.buffers(), self.net_ema.buffers()):
            b_ema.copy_(b)



