import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import LightningModule

from src.model.backbone import CORABackbone


class SimpleHead(nn.Module):
    def __init__(self, input_dim=256, num_classes=2, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class LightweightClassifier(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=64, num_classes=2, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight=None, reduction: str = "mean"):
        super().__init__()
        self.gamma = float(gamma)
        if weight is not None:
            self.register_buffer("weight", torch.as_tensor(weight, dtype=torch.float32))
        else:
            self.weight = None
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class ClsModule(LightningModule):
    def __init__(self, cfg, data, logger):
        super().__init__()
        self.cfg = cfg
        self.run_logger = logger
        self.data_ref = data

        self.runtime_c_in = int(data["meta"].get("C", len(data["input_coords"])))
        self.target_canonical = int(len(data["target_coords"]))
        self.base_channels = int(getattr(cfg, "base_channels", 32))
        self.use_diff_latent = bool(getattr(cfg, "use_diff_latent", True))

        bottleneck_dim = self.base_channels * 8

        self.backbone = CORABackbone(
            in_channels=self.target_canonical,
            out_channels=self.runtime_c_in,
            repr_mode="bottleneck",
            use_spatial_adapter=True,
            input_coords=data["input_coords"],
            target_coords=data["target_coords"],
            coord_dim=int(getattr(cfg, "coord_dim", 3)),
            adapter_dim=int(getattr(cfg, "adapter_dim", 128)),
            adapter_heads=int(getattr(cfg, "adapter_heads", 4)),
            adapter_coord_hidden_dim=int(getattr(cfg, "adapter_coord_hidden_dim", 64)),
            adapter_dropout=float(getattr(cfg, "adapter_dropout", 0.0)),
            base_channels=self.base_channels,
            use_diff_latent=self.use_diff_latent,
        )

        self._load_pretrained(cfg.pretrained_ckpt)

        self.freeze_backbone = bool(getattr(cfg, "freeze_backbone", True))
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("[CLS] backbone frozen")
        else:
            print("[CLS] backbone trainable")

        num_classes = int(getattr(cfg, "num_classes", 2))
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        self.num_classes = num_classes

        cls_arch = str(getattr(cfg, "cls_arch", "mlp"))
        if cls_arch == "simple":
            self.classifier = SimpleHead(
                input_dim=bottleneck_dim,
                num_classes=self.num_classes,
                dropout=float(getattr(cfg, "cls_dropout", 0.5)),
            )
        elif cls_arch == "mlp":
            self.classifier = LightweightClassifier(
                input_dim=bottleneck_dim,
                hidden_dim=int(getattr(cfg, "cls_hidden_dim", 64)),
                num_classes=self.num_classes,
                dropout=float(getattr(cfg, "cls_dropout", 0.3)),
            )
        else:
            raise ValueError(f"Unknown cls_arch: {cls_arch}")

        weight_vec = getattr(cfg, "class_weight_vec", None)
        if weight_vec is not None:
            class_weight_t = torch.tensor(weight_vec, dtype=torch.float32)
        else:
            class_weight_t = None

        if bool(getattr(cfg, "use_focal_loss", False)):
            self.criterion = FocalLoss(
                gamma=float(getattr(cfg, "focal_gamma", 2.0)),
                weight=class_weight_t,
            )
        else:
            self.criterion = nn.CrossEntropyLoss(weight=class_weight_t)

        self._val_logits_buf = []
        self._val_y_buf = []

        self.save_hyperparameters(ignore=["data", "logger"])

    def _load_pretrained(self, ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location="cpu")

        state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        stripped = {k[len("model."):]: v for k, v in state.items()
                    if k.startswith("model.")}
        if not stripped:
            stripped = dict(state)

        model_state = self.backbone.state_dict()
        kept = {k: v for k, v in stripped.items()
                if k in model_state and model_state[k].shape == v.shape}
        self.backbone.load_state_dict(kept, strict=False)
        print(f"[CLS] pretrained loaded: {len(kept)}/{len(model_state)} keys")

    def forward(self, x):
        _, repr_vec, W = self.backbone(
            x,
            input_coords=self.data_ref["input_coords"],
            target_coords=self.data_ref["target_coords"],
            return_repr=True,
            return_weights=True,
        )
        logits = self.classifier(repr_vec)
        return logits, W

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, _ = self(x.float())
        loss = self.criterion(logits, y.long())
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits, _ = self(x.float())
        loss = self.criterion(logits, y.long())
        self._val_logits_buf.append(logits.detach().cpu())
        self._val_y_buf.append(y.detach().cpu())
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        if not self._val_logits_buf:
            return
        logits = torch.cat(self._val_logits_buf, dim=0)
        y = torch.cat(self._val_y_buf, dim=0)
        self._val_logits_buf.clear()
        self._val_y_buf.clear()

        pred = logits.argmax(dim=1)
        acc = (pred == y).float().mean().item()

        classes_present = torch.unique(y)
        recalls = []
        for c in classes_present:
            mask = (y == c)
            if mask.sum() > 0:
                recalls.append((pred[mask] == c).float().mean().item())
        bal_acc = float(sum(recalls) / max(len(recalls), 1)) if recalls else 0.0

        try:
            from sklearn.metrics import roc_auc_score
            probs = torch.softmax(logits, dim=1).numpy()
            y_np = y.numpy()
            if logits.shape[1] == 2:
                auroc = float(roc_auc_score(y_np, probs[:, 1]))
            else:
                auroc = float(roc_auc_score(y_np, probs, multi_class="ovr", average="macro"))
        except Exception:
            auroc = float("nan")

        self.log("val_acc", acc, prog_bar=True)
        self.log("val_bal_acc", bal_acc, prog_bar=True)
        self.log("val_auroc", auroc, prog_bar=True)

    def configure_optimizers(self):
        lr = float(getattr(self.cfg, "lr", 1e-3))
        lr_backbone = float(getattr(self.cfg, "lr_backbone", 1e-5))
        wd = float(getattr(self.cfg, "weight_decay", 1e-2))

        if self.freeze_backbone:
            optimizer = optim.AdamW(self.classifier.parameters(), lr=lr, weight_decay=wd)
        else:
            optimizer = optim.AdamW([
                {"params": self.backbone.parameters(), "lr": lr_backbone},
                {"params": self.classifier.parameters(), "lr": lr},
            ], weight_decay=wd)
        return optimizer
