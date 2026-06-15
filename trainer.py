"""Trainer for CFBMNet flow curve prediction."""
import os
import torch
import torch.nn as nn
from tqdm import tqdm
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Optional

from utils.losses import (
    regression_curve_loss,
    physical_constraint_loss,
    smoothness_loss,
)

logger = logging.getLogger(__name__)


class UncertaintyLossWeighter(nn.Module):
    """
    Uncertainty-based loss weighting (Kendall et al.).

    For each loss term L_i with learnable log_sigma_i:
        weighted_i = exp(-2*log_sigma_i) * L_i + log_sigma_i
    Total = sum_i weighted_i

    Notes:
    - This implicitly learns dynamic weights during training.
    - We keep the +log_sigma regularizer to prevent trivial solutions.
    """

    def __init__(self, loss_keys):
        super().__init__()
        self.log_sigma = nn.ParameterDict({k: nn.Parameter(torch.zeros(())) for k in loss_keys})

    def forward(self, losses: dict):
        total = 0.0
        weights = {}
        for k, L in losses.items():
            if k not in self.log_sigma:
                continue
            s = self.log_sigma[k]
            w = torch.exp(-2.0 * s)
            weights[k] = w
            total = total + (w * L + s)
        return total, weights


class RegressionTrainer:
    """
    Trainer for direct curve regression with optional physics-informed losses.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler=None,
        lambda_reg=1.0,
        lambda_phys=1.0,
        loss_type='l1',
        lambda_smooth=0.0,
        sob_alpha=0.2,
        sob_beta=0.05,
        sob_delta=1.0,
        fev1_weight=1.0,
        fvc_weight=1.0,
        pef_weight=2.0,
        loss_weighter: Optional[UncertaintyLossWeighter] = None,
        uw_warmup_epochs: int = 0,
        grad_clip_norm: float = 1.0,
        device='cuda',
        checkpoint_dir='checkpoints',
        early_stopping_patience=20,
        fold_info=None,
        checkpoint_tag=None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lambda_reg = lambda_reg
        self.lambda_phys = lambda_phys
        self.loss_type = loss_type
        self.lambda_smooth = lambda_smooth
        self.sob_alpha = sob_alpha
        self.sob_beta = sob_beta
        self.sob_delta = sob_delta
        self.fev1_weight = fev1_weight
        self.fvc_weight = fvc_weight
        self.pef_weight = pef_weight
        self.loss_weighter = loss_weighter.to(device) if loss_weighter is not None else None
        self.uw_warmup_epochs = int(uw_warmup_epochs) if uw_warmup_epochs is not None else 0
        self.grad_clip_norm = float(grad_clip_norm)
        self._use_uw_now = False
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.early_stopping_patience = early_stopping_patience
        self.fold_info = fold_info
        self.checkpoint_tag = checkpoint_tag

        os.makedirs(checkpoint_dir, exist_ok=True)

        self.best_val_loss = float('inf')
        self.current_epoch = 0
        self.patience_counter = 0
        self.early_stopped = False

        self.train_loss_history = []
        self.val_loss_history = []

    def _forward_batch(self, batch):
        mel = batch['mel'].to(self.device)
        flow_gt = batch['flow'].to(self.device)
        labels = batch['labels'].to(self.device)
        fev1_gt = labels[:, 0]
        fvc_gt = labels[:, 1]
        pef_gt = labels[:, 2]

        demographic = batch.get('demographic', None)
        if demographic is not None:
            demographic = demographic.to(self.device)

        if demographic is not None:
            flow_pred = self.model(mel, demographic)
        else:
            flow_pred = self.model(mel)
        flow_pred = torch.clamp(flow_pred, min=0.0)

        loss_reg = regression_curve_loss(
            flow_pred,
            flow_gt,
            loss_type=self.loss_type,
            sob_alpha=self.sob_alpha,
            sob_beta=self.sob_beta,
            sob_delta=self.sob_delta,
        )
        loss_phys = None
        if self.lambda_phys and self.lambda_phys > 0:
            loss_phys = physical_constraint_loss(
                flow_pred,
                fev1_gt,
                fvc_gt,
                pef_gt,
                fev1_weight=self.fev1_weight,
                fvc_weight=self.fvc_weight,
                pef_weight=self.pef_weight,
            )

        loss_smooth = None
        if self.lambda_smooth and self.lambda_smooth > 0:
            loss_smooth = smoothness_loss(flow_pred, order=2)

        if (self.loss_weighter is not None) and bool(self._use_uw_now):
            losses_for_weight = {'reg': loss_reg}
            if loss_phys is not None:
                losses_for_weight['phys'] = loss_phys
            if loss_smooth is not None:
                losses_for_weight['smooth'] = loss_smooth
            total_loss, learned_w = self.loss_weighter(losses_for_weight)
        else:
            total_loss = self.lambda_reg * loss_reg
            if loss_phys is not None:
                total_loss = total_loss + self.lambda_phys * loss_phys
            if loss_smooth is not None:
                total_loss = total_loss + self.lambda_smooth * loss_smooth
            learned_w = None

        loss_dict = {
            'loss_reg': float(loss_reg.item()),
            'loss_phys': float(loss_phys.item()) if loss_phys is not None else 0.0,
            'total_loss': float(total_loss.item()),
        }
        if loss_smooth is not None:
            loss_dict['loss_smooth'] = float(loss_smooth.item())
        if learned_w is not None:
            # Report learned weights as floats for logging only
            for k, w in learned_w.items():
                loss_dict[f'uw_{k}'] = float(w.detach().item())
        return total_loss, loss_dict

    def _update_uw_state(self):
        """Update uncertainty weighting state based on current epoch."""
        if self.loss_weighter is not None and self.uw_warmup_epochs > 0:
            self._use_uw_now = bool(self.current_epoch > self.uw_warmup_epochs)
        else:
            self._use_uw_now = bool(self.loss_weighter is not None)
        if self.loss_weighter is not None:
            for p in self.loss_weighter.parameters():
                p.requires_grad = bool(self._use_uw_now)

    def train_epoch(self):
        self.model.train()

        if (self.loss_weighter is not None) and (not self._use_uw_now) and self.uw_warmup_epochs > 0:
            logger.info(
                f"{(self.fold_info + ' - ') if self.fold_info else ''}"
                f"Uncertainty weighting warmup: epoch {self.current_epoch}/{self.uw_warmup_epochs} (fixed weights)"
            )
        if (self.loss_weighter is not None) and self._use_uw_now and self.uw_warmup_epochs > 0 and self.current_epoch == self.uw_warmup_epochs + 1:
            logger.info(
                f"{(self.fold_info + ' - ') if self.fold_info else ''}"
                f"Uncertainty weighting enabled after warmup (epoch {self.current_epoch})"
            )

        epoch_losses = {'loss_reg': 0.0, 'loss_phys': 0.0, 'total_loss': 0.0}
        if self.lambda_smooth > 0:
            epoch_losses['loss_smooth'] = 0.0
        num_batches = 0

        desc = f'Epoch {self.current_epoch} [Train]'
        if self.fold_info:
            desc = f'{self.fold_info} - {desc}'
        pbar = tqdm(self.train_loader, desc=desc)

        for batch in pbar:
            total_loss, loss_dict = self._forward_batch(batch)

            self.optimizer.zero_grad()
            total_loss.backward()
            params_to_clip = list(self.model.parameters())
            if self.loss_weighter is not None:
                params_to_clip += list(self.loss_weighter.parameters())
            torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=self.grad_clip_norm)
            self.optimizer.step()

            # Accumulate known losses
            for key in epoch_losses:
                if key in loss_dict:
                    epoch_losses[key] += float(loss_dict[key])
            # Accumulate uncertainty weights (appear only when enabled)
            for k, v in loss_dict.items():
                if str(k).startswith('uw_'):
                    if k not in epoch_losses:
                        epoch_losses[k] = 0.0
                    epoch_losses[k] += float(v)
            num_batches += 1

            avg_losses = {k: epoch_losses[k] / max(num_batches, 1) for k in epoch_losses}
            pbar.set_postfix({k: f"{v:.4f}" for k, v in avg_losses.items()})

        for key in list(epoch_losses.keys()):
            epoch_losses[key] /= max(num_batches, 1)
        return epoch_losses

    @torch.no_grad()
    def validate_epoch(self):
        self.model.eval()
        epoch_losses = {'loss_reg': 0, 'loss_phys': 0, 'total_loss': 0}
        if self.lambda_smooth > 0:
            epoch_losses['loss_smooth'] = 0
        num_batches = 0

        desc = f'Epoch {self.current_epoch} [Val]'
        if self.fold_info:
            desc = f'{self.fold_info} - {desc}'
        pbar = tqdm(self.val_loader, desc=desc)

        for batch in pbar:
            total_loss, loss_dict = self._forward_batch(batch)
            for key in epoch_losses:
                if key in loss_dict:
                    epoch_losses[key] += float(loss_dict[key])
            num_batches += 1
            pbar.set_postfix({'loss': f"{loss_dict['total_loss']:.4f}"})

        for key in epoch_losses:
            epoch_losses[key] /= max(num_batches, 1)
        return epoch_losses

    def train(self, num_epochs):
        fold_prefix = f"{self.fold_info} - " if self.fold_info else ""
        logger.info(f"{fold_prefix}Starting training for {num_epochs} epochs")
        logger.info(f"{fold_prefix}Early stopping patience: {self.early_stopping_patience} epochs")

        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            self._update_uw_state()
            train_losses = self.train_epoch()
            val_losses = self.validate_epoch()

            self.train_loss_history.append(train_losses['total_loss'])
            self.val_loss_history.append(val_losses['total_loss'])

            val_sel_for_sched = val_losses['loss_reg'] + val_losses['loss_phys']
            if self.scheduler is not None:
                old_lr = self.optimizer.param_groups[0]['lr']
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_sel_for_sched)
                else:
                    self.scheduler.step()
                new_lr = self.optimizer.param_groups[0]['lr']
                if old_lr != new_lr:
                    logger.info(f"{fold_prefix}Epoch {epoch}: Learning rate reduced from {old_lr:.2e} to {new_lr:.2e}")

            logger.info(
                f"{fold_prefix}Epoch {epoch}/{num_epochs} - "
                f"Train Loss: {train_losses['total_loss']:.4f}, "
                f"Val Loss: {val_losses['total_loss']:.4f}, "
                f"Val loss_reg: {val_losses['loss_reg']:.4f}"
            )
            uw_keys = [k for k in train_losses.keys() if str(k).startswith('uw_')]
            if uw_keys:
                uw_msg = ", ".join([f"{k}={train_losses[k]:.4f}" for k in sorted(uw_keys)])
                logger.info(f"{fold_prefix}Epoch {epoch}/{num_epochs} - {uw_msg}")

            # Model selection: loss_reg + loss_phys (when physics loss is active).
            # Both terms are computed identically regardless of whether uncertainty
            # weighting is active, so they are comparable across warmup and UW phases.
            val_sel = val_losses['loss_reg'] + val_losses['loss_phys']
            if val_sel < self.best_val_loss:
                self.best_val_loss = val_sel
                self.patience_counter = 0
                self.save_best_model(epoch)
                logger.info(f"{fold_prefix}[BEST] New best model saved with val selection loss: {self.best_val_loss:.4f}")
            else:
                self.patience_counter += 1
                logger.info(f"{fold_prefix}No improvement for {self.patience_counter} epoch(s)")

            if self.patience_counter >= self.early_stopping_patience:
                logger.info(f"{fold_prefix}Early stopping triggered after {epoch} epochs")
                logger.info(f"{fold_prefix}Best val selection loss: {self.best_val_loss:.4f}")
                self.early_stopped = True
                break

        if not self.early_stopped:
            logger.info(f"{fold_prefix}Training completed!")
        logger.info(f"{fold_prefix}Best val selection loss: {self.best_val_loss:.4f}")

    def save_best_model(self, epoch):
        suffix = f"_{self.checkpoint_tag}" if self.checkpoint_tag else ""
        model_path = os.path.join(self.checkpoint_dir, f'best_model{suffix}.pth')
        torch.save(self.model.state_dict(), model_path)

    def plot_loss_curves(self, save_path=None):
        if save_path is None:
            suffix = f"_{self.checkpoint_tag}" if self.checkpoint_tag else ""
            save_path = os.path.join(self.checkpoint_dir, f'loss_curves{suffix}.png')

        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.train_loss_history) + 1)
        plt.plot(epochs, self.train_loss_history, 'b-', label='Train Loss', linewidth=2)
        plt.plot(epochs, self.val_loss_history, 'r-', label='Val Loss', linewidth=2)
        if len(self.val_loss_history) > 0:
            best_epoch = self.val_loss_history.index(min(self.val_loss_history)) + 1
            best_val_loss = min(self.val_loss_history)
            plt.plot(best_epoch, best_val_loss, 'r*', markersize=15, label=f'Best (Epoch {best_epoch})')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        logger.info(f"Loss curves saved to {save_path}")



