import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.model.backbone import CORABackbone
from src.utility.loss_mask import combine_time_channel_masks


def first_tensor(output):
    return output[0] if isinstance(output, (tuple, list)) else output


def _make_logger(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(f"cora.{log_dir}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(os.path.join(log_dir, "train.log"))
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


class ReconModule(LightningModule):
    def __init__(self, cfg, data, logger):
        super().__init__()
        self.cfg = cfg
        self.run_logger = logger
        self.data_ref = data

        self.runtime_c_in = int(data["meta"].get("C", len(data["input_coords"])))
        self.target_canonical = int(len(data["target_coords"]))

        self.time_mask_ratio = float(getattr(cfg, "time_mask_ratio", 0.0))
        self.channel_mask_ratio = float(getattr(cfg, "channel_mask_ratio", 0.0))
        self.channel_mask_mode = str(getattr(cfg, "channel_mask_mode", "none"))
        self.min_keep_channels = int(getattr(cfg, "min_keep_channels", 1))
        self.ssp_blocks = int(getattr(cfg, "ssp_blocks", 8))
        self.ssp_even = bool(getattr(cfg, "ssp_even", False))
        self.ssp_min_block = int(getattr(cfg, "ssp_min_block", 4))
        self.topo_group_frac = float(getattr(cfg, "topo_group_frac", 0.25))

        self.base_channels = int(getattr(cfg, "base_channels", 32))
        self.use_diff_latent = bool(getattr(cfg, "use_diff_latent", True))
        self.diff_num_heads = int(getattr(cfg, "diff_num_heads", 4))
        self.diff_mlp_ratio = float(getattr(cfg, "diff_mlp_ratio", 2.0))
        self.diff_dropout = float(getattr(cfg, "diff_dropout", 0.0))
        self.diff_smooth_kernel = int(getattr(cfg, "diff_smooth_kernel", 5))
        self.diff_lambda_max = float(getattr(cfg, "diff_lambda_max", 1.0))

        model_name = str(getattr(cfg, "model_name", "CORABackbone"))
        default_use_spatial = (model_name == "CORABackbone")
        self.use_spatial_adapter = bool(getattr(cfg, "use_spatial_adapter", default_use_spatial))

        self.cora_entropy_weight = float(getattr(cfg, "cora_entropy_weight", 0.0))
        self.cora_locality_weight = float(getattr(cfg, "cora_locality_weight", 0.0))
        self.cora_head_div_weight = float(getattr(cfg, "cora_head_div_weight", 0.0))
        self.cora_target_entropy = float(getattr(cfg, "cora_target_entropy", 2.5))

        self.guide_kind = str(getattr(cfg, "guide_kind", "none"))
        self.guide_mode = str(getattr(cfg, "guide_mode", "kl"))
        self.guide_geo_sigma = float(getattr(cfg, "guide_geo_sigma", 0.3))
        self.guide_geo_frac = float(getattr(cfg, "guide_geo_frac", 0.5))
        self.sloreta_guide_path = str(getattr(cfg, "sloreta_guide_path", ""))

        self.model = self._build_model(cfg, data)
        self.criterion = self._build_loss(cfg)

        self._maybe_load_guide()

        self.save_hyperparameters(ignore=["data", "logger"])

    def _build_model(self, cfg, data):
        if self.use_spatial_adapter:
            backbone_in_channels = self.target_canonical
            final_out_channels = self.runtime_c_in
        else:
            backbone_in_channels = self.runtime_c_in
            final_out_channels = self.runtime_c_in

        return CORABackbone(
            in_channels=backbone_in_channels,
            out_channels=final_out_channels,
            repr_mode=str(getattr(cfg, "repr_mode", "bottleneck")),
            use_spatial_adapter=self.use_spatial_adapter,
            input_coords=data["input_coords"],
            target_coords=data["target_coords"],
            coord_dim=int(getattr(cfg, "coord_dim", 3)),
            adapter_dim=int(getattr(cfg, "adapter_dim", 128)),
            adapter_heads=int(getattr(cfg, "adapter_heads", 4)),
            adapter_coord_hidden_dim=int(getattr(cfg, "adapter_coord_hidden_dim", 64)),
            adapter_dropout=float(getattr(cfg, "adapter_dropout", 0.0)),
            base_channels=int(getattr(cfg, "base_channels", 32)),
            use_diff_latent=bool(getattr(cfg, "use_diff_latent", True)),
            diff_num_heads=int(getattr(cfg, "diff_num_heads", 4)),
            diff_mlp_ratio=float(getattr(cfg, "diff_mlp_ratio", 2.0)),
            diff_dropout=float(getattr(cfg, "diff_dropout", 0.0)),
            diff_smooth_kernel=int(getattr(cfg, "diff_smooth_kernel", 5)),
            diff_lambda_max=float(getattr(cfg, "diff_lambda_max", 1.0)),
        )

    @staticmethod
    def _build_loss(cfg):
        loss_name = str(getattr(cfg, "loss", "L1Loss"))
        if loss_name in {"L1Loss", "L1"}:
            return nn.L1Loss()
        if loss_name in {"MSELoss", "MSE"}:
            return nn.MSELoss()
        raise ValueError(f"Unsupported loss={loss_name}")

    def _maybe_load_guide(self):
        if self.guide_kind not in ("sloreta", "both"):
            return

        if not self.sloreta_guide_path:
            raise ValueError(
                f"guide_kind={self.guide_kind} requires sloreta_guide_path"
            )

        if not hasattr(self.model, "load_sloreta_guide"):
            raise AttributeError(
                "Model does not implement load_sloreta_guide(path)."
            )

        self.model.load_sloreta_guide(self.sloreta_guide_path)
        if self.run_logger is not None:
            self.run_logger.info(
                f"[GUIDE] Loaded sLORETA guide from: {self.sloreta_guide_path}"
            )

    def maybe_mask(self, x):
        if self.time_mask_ratio <= 0.0 and self.channel_mask_ratio <= 0.0:
            return x

        coords_for_topology = None
        if self.channel_mask_mode == "topological":
            coords_for_topology = getattr(self.model, "in_pos_buf", None)

        x_masked, _ = combine_time_channel_masks(
            x,
            {
                "time_mask_ratio": self.time_mask_ratio,
                "num_preserved_blocks": self.ssp_blocks,
                "evenly_spaced": self.ssp_even,
                "min_block_len": self.ssp_min_block,
                "min_keep_channels": self.min_keep_channels,
                "topo_group_frac": self.topo_group_frac,
            },
            self.channel_mask_ratio,
            self.channel_mask_mode,
            coords_for_topology=coords_for_topology,
        )
        return x_masked

    def forward(self, x, return_repr=False, return_weights=False):
        if self.use_spatial_adapter:
            return self.model(
                x,
                return_repr=return_repr,
                input_coords=self.data_ref["input_coords"],
                target_coords=self.data_ref["target_coords"],
                return_weights=return_weights,
            )
        return self.model(
            x,
            return_repr=return_repr,
            return_weights=return_weights,
        )

    def _compute_cora_reg_loss(self):
        has_any_reg = (
            self.cora_entropy_weight > 0.0
            or self.cora_locality_weight > 0.0
            or self.cora_head_div_weight > 0.0
            or self.guide_kind != "none"
        )
        if not has_any_reg:
            return torch.tensor(0.0, device=self.device)

        if not hasattr(self.model, "compute_cora_losses"):
            raise AttributeError(
                "Model does not implement compute_cora_losses(...)."
            )

        if hasattr(self.cfg, "get_guide_weight"):
            guide_weight = float(self.cfg.get_guide_weight(self.current_epoch))
        else:
            guide_weight = float(getattr(self.cfg, "guide_weight_base", 0.0))

        reg_loss = self.model.compute_cora_losses(
            entropy_weight=self.cora_entropy_weight,
            locality_weight=self.cora_locality_weight,
            head_diversity_weight=self.cora_head_div_weight,
            target_entropy=self.cora_target_entropy,
            guide_kind=self.guide_kind,
            guide_weight=guide_weight,
            guide_geo_sigma=self.guide_geo_sigma,
            guide_mode=self.guide_mode,
            guide_geo_frac=self.guide_geo_frac,
        )

        if not torch.is_tensor(reg_loss):
            reg_loss = torch.tensor(float(reg_loss), device=self.device)

        return reg_loss

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.float()

        x_target = x
        x_in = self.maybe_mask(x)

        output = first_tensor(self(x_in, return_repr=False, return_weights=False))
        recon_loss = self.criterion(output, x_target)

        reg_loss = self._compute_cora_reg_loss()
        total_loss = recon_loss + reg_loss

        if hasattr(self.cfg, "get_guide_weight"):
            guide_weight = float(self.cfg.get_guide_weight(self.current_epoch))
        else:
            guide_weight = float(getattr(self.cfg, "guide_weight_base", 0.0))

        self.log("train_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/recon_loss", recon_loss, prog_bar=False, on_step=True, on_epoch=True)
        self.log("train/reg_loss", reg_loss, prog_bar=False, on_step=True, on_epoch=True)
        self.log("train/guide_w", guide_weight, prog_bar=False, on_step=True, on_epoch=True)

        if getattr(self.model, "use_diff_latent", False):
            parts = self.model.get_last_diff_parts(detach=False)
            if parts is not None and "lambda" in parts:
                lam = parts["lambda"]
                if torch.is_tensor(lam):
                    self.log("diff_lambda", lam, prog_bar=False, on_step=True, on_epoch=True)

        return total_loss

    def configure_optimizers(self):
        lr = float(getattr(self.cfg, "lr", 1e-4))
        weight_decay = float(getattr(self.cfg, "weight_decay", 1e-2))
        optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        epochs = int(getattr(self.cfg, "epochs", 50))
        max_lr = float(getattr(self.cfg, "onecycle_max_lr", lr * 2.0))
        estimated_steps = getattr(self.trainer, "estimated_stepping_batches", None)

        if estimated_steps is None:
            steps_per_epoch = 1
        else:
            steps_per_epoch = max(1, estimated_steps // epochs)

        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.15,
            anneal_strategy="cos",
            div_factor=2.0,
            final_div_factor=100.0,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


def run_reconstruction(config, data, extra_callbacks=None):
    os.makedirs(config.out_dir, exist_ok=True)

    probe_file = getattr(config, "probe_file", "")
    run_name = Path(probe_file).stem if probe_file else "probe_run"

    run_dir = os.path.join(config.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    tb_logger = TensorBoardLogger(save_dir=run_dir, name="tb")
    logger = _make_logger(tb_logger.log_dir)

    dataloader = data["loader"]
    meta = data["meta"]
    sample_batch, _ = next(iter(dataloader))
    batch_size, channels, time_steps = sample_batch.shape

    n_samples = meta.get("N")
    if n_samples is None:
        try:
            n_samples = len(dataloader.dataset)
        except Exception:
            n_samples = -1

    logger.info(
        f"[DATA] {meta.get('name', '')} | X_{getattr(config, 'split', 'train')}: "
        f"N={n_samples} | C={channels} | T:{meta.get('T_before', '?')}->{time_steps} | batch={batch_size}"
    )

    ckpt_dir = os.path.join(tb_logger.log_dir, "models")
    os.makedirs(ckpt_dir, exist_ok=True)

    lightning_model = ReconModule(config, data, logger)

    logger.info(
        f"[MODEL] model_name={getattr(config, 'model_name', 'CORABackbone')} | "
        f"use_spatial_adapter={lightning_model.use_spatial_adapter} | "
        f"base_channels={getattr(config, 'base_channels', 32)} | "
        f"use_diff_latent={getattr(config, 'use_diff_latent', True)} | "
        f"runtime_in={lightning_model.runtime_c_in} | "
        f"canonical_in={lightning_model.target_canonical if lightning_model.use_spatial_adapter else lightning_model.runtime_c_in}"
    )

    logger.info(
        f"[CORA] entropy={getattr(config, 'cora_entropy_weight', 0.0)} | "
        f"locality={getattr(config, 'cora_locality_weight', 0.0)} | "
        f"head_div={getattr(config, 'cora_head_div_weight', 0.0)} | "
        f"target_entropy={getattr(config, 'cora_target_entropy', 2.5)}"
    )
    logger.info(
        f"[GUIDE] kind={getattr(config, 'guide_kind', 'none')} | "
        f"mode={getattr(config, 'guide_mode', 'kl')} | "
        f"geo_sigma={getattr(config, 'guide_geo_sigma', 0.3)} | "
        f"geo_frac={getattr(config, 'guide_geo_frac', 0.5)} | "
        f"sloreta={getattr(config, 'sloreta_guide_path', '') or '(none)'}"
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best-ep{epoch:03d}",
            auto_insert_metric_name=False,
            monitor="train_loss_epoch",
            mode="min",
            save_top_k=1,
            save_last=True,
        ),
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="ep{epoch:03d}",
            auto_insert_metric_name=False,
            save_top_k=-1,
            every_n_epochs=1,
            save_on_train_epoch_end=True,
        ),
    ]
    if extra_callbacks:
        callbacks.extend(list(extra_callbacks))
        logger.info(f"[CB] +{len(extra_callbacks)} extra callback(s)")

    trainer = Trainer(
        default_root_dir=run_dir,
        logger=tb_logger,
        callbacks=callbacks,
        max_epochs=int(getattr(config, "epochs", 50)),
        accelerator="cpu" if getattr(config, "cpu", False) else ("gpu" if torch.cuda.is_available() else "cpu"),
        devices=1,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        enable_progress_bar=bool(getattr(config, "enable_progress_bar", True)),
    )

    trainer.fit(lightning_model, train_dataloaders=dataloader)
    logger.info(f"[DONE] checkpoints at: {ckpt_dir}")

    return lightning_model, lightning_model.model, ckpt_dir
