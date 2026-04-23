"""
Defocus Estimation Training Script — Scalar Head Architecture

Single-stage training: DME-subnet predicts scalar blur in [0, 1].
DD-subnet removed — scalar blur estimation only.

Usage:
    python train.py --config training_config.yaml --data-dir data/synthetic
"""

import argparse
import logging
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import create_dme_dataloaders, DMEDataset
from losses import DMELoss
from model import DefocusNet, model_summary

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
from physics import validate_training_config, ConfigError

# Windows + tkinter GUI causes multiprocessing issues
_NUM_WORKERS = 0 if platform.system() == "Windows" else 4

logger = logging.getLogger(__name__)

# Ensure per-epoch training stats reach the console when nobody upstream has
# configured logging (e.g. GUI-launched runs). Skip if the root logger already
# has handlers so we don't clobber an app-level config.
if not logging.getLogger().handlers and not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


class Trainer:
    """
    Trainer for DefocusNet scalar head (DME-only).
    """

    def __init__(
        self,
        config: Dict[str, Any],
        data_dir: Path,
        output_dir: Path,
        device: str = 'auto',
        stop_flag: Optional[callable] = None
    ):
        """
        Args:
            config: Configuration dictionary
            data_dir: Path to training data
            output_dir: Path for checkpoints and logs
            device: 'auto', 'cuda', or 'cpu'
            stop_flag: Optional callable that returns True when training should stop
        """
        self.config = config
        self.data_dir = Path(data_dir)
        # output_dir is the model folder. checkpoints/ holds .pth files;
        # logs/ holds tensorboard events plus the per-run metadata and
        # history artifacts (run_metadata.json, training_history.yaml,
        # training_curves.png). training_config.yaml sits at the root
        # since the trainer resolves it from inputs on startup.
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.output_dir / 'checkpoints'
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.logs_dir = self.output_dir / 'logs'
        self.logs_dir.mkdir(exist_ok=True)

        training_mode = config.get('training', {}).get('training_mode', 'optical')
        config_warnings = validate_training_config(config, training_mode)
        for w in config_warnings:
            logger.warning(f"Config: {w}")

        # Device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        logger.info(f"Using device: {self.device}")

        # cuDNN settings for reproducibility
        if self.device.type == 'cuda':
            torch.backends.cudnn.deterministic = True  # Reproducible results
            # benchmark=True (default) allows cuDNN to find fastest convolution algorithms
            logger.info("cuDNN: deterministic=True (reproducible mode)")

        # Training config
        train_cfg = config.get('training', {})
        self.batch_size = train_cfg.get('batch_size', 128)
        self.epochs_dme = train_cfg.get('epochs_dme', 400)
        self.lr = train_cfg.get('lr', train_cfg.get('learning_rate', 0.0002))
        self.override_checkpoint_lr = train_cfg.get('override_checkpoint_lr', False)
        self.lr_decay_start = train_cfg.get('lr_decay_start_epoch', 200)
        self.lr_decay_rate = train_cfg.get('lr_decay_rate', 0.005)
        self.save_every = train_cfg.get('save_every_epochs', 1)
        self.save_only_best = train_cfg.get('save_only_best', False)
        self.log_eps = train_cfg.get('log_eps', 0.01)

        # Optimizer config
        self.optimizer_type = train_cfg.get('optimizer', 'adam').lower()
        self.adam_beta1 = float(train_cfg.get('adam_beta1', 0.9))
        self.adam_beta2 = float(train_cfg.get('adam_beta2', 0.999))
        self.weight_decay = float(train_cfg.get('weight_decay', 0.0))
        self.grad_clip_norm = float(train_cfg.get('grad_clip_norm', 0.0))
        self.lr_schedule = train_cfg.get('lr_schedule', 'step').lower()
        self.lr_min = float(train_cfg.get('lr_min', 1e-6))
        # num_workers hardcoded to 0 - multiprocessing doesn't work when launched from GUI on Windows

        # Data config
        data_cfg = config.get('data', {})
        # Compute max_blur and min_blur from metadata (at model scale)
        from dataset import compute_max_blur_from_metadata, compute_min_blur_from_metadata
        self.max_blur = compute_max_blur_from_metadata(self.data_dir, margin_percent=5.0)
        self.min_blur = compute_min_blur_from_metadata(self.data_dir)
        self.stratified = train_cfg.get('stratified', False)
        self.training_mode = train_cfg.get('training_mode', 'optical')
        self.blur_term = "σ" if self.training_mode == "direct" else "CoC"

        # Model
        self.model = DefocusNet.from_config(config).to(self.device)
        model_summary(self.model)

        # Loss — single source of truth for max_blur
        self.dme_loss_fn = DMELoss(max_blur=self.max_blur, eps=self.log_eps)

        # Tensorboard
        self.writer = SummaryWriter(self.logs_dir)

        # Training history tracking
        self.history_file = self.logs_dir / 'training_history.yaml'
        self.training_history = self._load_training_history()

        # Session timestamp for unique checkpoint names
        self.session_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Current session info (will be updated during training)
        self.current_session = {
            'session_id': self.session_timestamp,
            'stage': None,
            'start_epoch': None,
            'end_epoch': None,
            'epochs_trained': 0,
            'lr_start': self.lr,
            'lr_end': self.lr,
            'lr_changes': [],
            'best_found': False,
            'best_epoch': None,
            'best_val_loss': None,
            'notes': []
        }

        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_dme_mae_px = float('inf')  # best scalar blur MAE (px) for DME stage
        # best MAE for current training session (resets on resume)
        self.best_current_run_mae = float('inf')
        self.train_split = train_cfg.get('train_split', 0.8)
        self.seed = train_cfg.get('seed', 42)

        # Stop flag for graceful shutdown
        self.stop_flag = stop_flag or (lambda: False)

        # Training curve history (for PNG export)
        self.curve_history = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'train_weighted_mae': [],
            'val_weighted_mae': [],
            'lr': [],
            'train_bin_maes': [],  # list of 4-element lists
            'val_bin_maes': [],    # list of 4-element lists
        }

        # Calculate bin weights from beta distribution
        self.bin_weights = self._calculate_bin_weights_from_beta()

    def _load_training_history(self) -> dict:
        """Load existing training history or create new one."""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                history = yaml.safe_load(f)
                if history is None:
                    history = self._create_empty_history()
                return history
        else:
            return self._create_empty_history()

    def _create_empty_history(self) -> dict:
        """Create empty training history structure."""
        return {
            'training_summary': {
                'best_model_epoch': None,
                'max_epoch_reached': 0,
                'total_epochs_trained': 0,
                'effective_epochs': None,
                'wasted_epochs': 0,
                'best_metrics': {
                    'val_loss': None,
                    'val_mae_px': None
                }
            },
            'training_sessions': [],
            'lr_history': [],
            'config': self.config
        }

    def _save_training_history(self):
        with open(self.history_file, 'w') as f:
            yaml.dump(self.training_history, f, default_flow_style=False, sort_keys=False)

    def _update_session_info(self, stage: str, epoch: int, val_loss: float = None):
        if self.current_session['start_epoch'] is None:
            self.current_session['start_epoch'] = epoch
            self.current_session['stage'] = stage

        self.current_session['end_epoch'] = epoch
        self.current_session['epochs_trained'] = epoch - self.current_session['start_epoch'] + 1

        # Track if this epoch improved the best model
        if val_loss is not None and (
                self.training_history['training_summary']['best_metrics']['val_loss'] is
                None or val_loss < self.training_history['training_summary']['best_metrics']
                ['val_loss']):
            self.current_session['best_found'] = True
            self.current_session['best_epoch'] = epoch
            self.current_session['best_val_loss'] = val_loss

            # Update global summary
            self.training_history['training_summary']['best_model_epoch'] = epoch
            self.training_history['training_summary']['best_metrics']['val_loss'] = val_loss
            self.training_history['training_summary']['effective_epochs'] = epoch

        # Update max epoch reached (highest epoch number ever)
        if epoch > self.training_history['training_summary']['max_epoch_reached']:
            self.training_history['training_summary']['max_epoch_reached'] = epoch

        # Calculate wasted epochs (based on max epoch reached)
        if self.training_history['training_summary']['effective_epochs'] is not None:
            self.training_history['training_summary']['wasted_epochs'] = (
                self.training_history['training_summary']['max_epoch_reached'] -
                self.training_history['training_summary']['effective_epochs']
            )

    def _finalize_session(self, notes: str = None):
        if notes:
            self.current_session['notes'].append(notes)

        # Add session to history
        self.training_history['training_sessions'].append(self.current_session.copy())

        # Accumulate total epochs trained (cumulative across all sessions)
        self.training_history['training_summary']['total_epochs_trained'] += self.current_session['epochs_trained']

        self._save_training_history()

    def _track_lr_change(self, epoch: int, old_lr: float, new_lr: float, reason: str):
        lr_change = {
            'epoch': epoch,
            'old_lr': old_lr,
            'new_lr': new_lr,
            'reason': reason
        }
        self.training_history['lr_history'].append(lr_change)
        self.current_session['lr_changes'].append(lr_change)
        self.current_session['lr_end'] = new_lr

    def _get_lr(self, epoch: int, base_lr: float) -> float:
        if self.lr_schedule == 'none':
            return base_lr
        if epoch < self.lr_decay_start:
            return base_lr
        decay_epochs = epoch - self.lr_decay_start
        if self.lr_schedule == 'cosine':
            import math
            total = max(self.epochs_dme - self.lr_decay_start, 1)
            progress = min(decay_epochs / total, 1.0)
            lr = self.lr_min + 0.5 * (base_lr - self.lr_min) * (1 + math.cos(math.pi * progress))
        elif self.lr_schedule == 'exponential':
            lr = base_lr * (1.0 - self.lr_decay_rate) ** decay_epochs
        else:  # 'step'
            lr = base_lr * (1.0 - self.lr_decay_rate) ** decay_epochs
        return max(lr, self.lr_min)

    def _should_stop(self) -> bool:
        return self.stop_flag()

    def _calculate_bin_weights_from_beta(self) -> list:
        from utils import calculate_bin_weights_from_beta
        return calculate_bin_weights_from_beta(self.config)

    def _update_lr(self, optimizer: optim.Optimizer, epoch: int, base_lr: float):
        lr = self._get_lr(epoch, base_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def _get_bins(self):
        """Compute 4 equal bins from min_blur to max_blur."""
        blur_range = self.max_blur - self.min_blur
        bin_size = blur_range / 4.0
        return [(self.min_blur + i * bin_size, self.min_blur + (i + 1) * bin_size) for i in range(4)]

    def _print_data_distribution(self, train_loader, val_loader, stage_name="Training"):
        """Print distribution of samples across blur bins for train and val sets."""
        import torch

        logger.info(f"\n{'='*60}")
        logger.info(f"{stage_name} Data Split Distribution")
        logger.info(f"{'='*60}")
        logger.info(f"Train: {len(train_loader.dataset):,} samples")
        logger.info(f"Val:   {len(val_loader.dataset):,} samples")
        logger.info(f"Seed:  {self.seed}")
        logger.info(
            f"Method: {'Stratified (balanced by ' + self.blur_term + ' bins)' if self.stratified else 'Random (seed-based)'}")

        bins = self._get_bins()
        bin_labels = [f"{low:.1f}-{high:.1f}" for low, high in bins]

        # Analyze train distribution
        logger.info(f"\nTrain Distribution by {self.blur_term} Bin:")
        train_bin_counts = [0] * len(bins)
        with torch.no_grad():
            for batch in train_loader:
                # blur_px is the 3rd element (index 2)
                blur_px = batch[2]
                blur_values = blur_px.numpy() if isinstance(blur_px, torch.Tensor) else blur_px

                for i, (low, high) in enumerate(bins):
                    count = np.sum((blur_values >= low) & (blur_values < high))
                    train_bin_counts[i] += count

        train_total = sum(train_bin_counts)
        for label, count in zip(bin_labels, train_bin_counts):
            pct = (count / train_total * 100) if train_total > 0 else 0
            logger.info(f"  {label:6s} px: {count:6,d} samples ({pct:5.1f}%)")

        # Analyze val distribution
        logger.info(f"\nVal Distribution by {self.blur_term} Bin:")
        val_bin_counts = [0] * len(bins)
        with torch.no_grad():
            for batch in val_loader:
                # blur_px is the 3rd element (index 2)
                blur_px = batch[2]
                blur_values = blur_px.numpy() if isinstance(blur_px, torch.Tensor) else blur_px

                for i, (low, high) in enumerate(bins):
                    count = np.sum((blur_values >= low) & (blur_values < high))
                    val_bin_counts[i] += count

        val_total = sum(val_bin_counts)
        for label, count in zip(bin_labels, val_bin_counts):
            pct = (count / val_total * 100) if val_total > 0 else 0
            logger.info(f"  {label:6s} px: {count:6,d} samples ({pct:5.1f}%)")

        logger.info(f"{'='*60}\n")

    def train_dme(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        resume_from: Optional[str] = None
    ) -> None:
        """
        Train DME-subnet (scalar head).

        Args:
            train_loader: Training data (blur_img, blur_norm, blur_px)
            val_loader: Validation data
            epochs: Number of epochs
            resume_from: Optional checkpoint path to resume from
        """
        logger.info("\n" + "=" * 60)
        logger.info("Training DME-subnet (scalar head)")
        logger.info("=" * 60)
        logger.info("\nCheckpoint Strategy:")
        logger.info("  • dme_best.pth - Global best (never overwritten unless beaten)")
        logger.info("  • dme_best_current_session.pth - Best in this training run")
        logger.info("  • dme_epoch_X.pth - Recovery checkpoints (every epoch)")

        params = self.model.dme_subnet.parameters()
        if self.optimizer_type == 'adamw':
            optimizer = optim.AdamW(params, lr=self.lr,
                                    betas=(self.adam_beta1, self.adam_beta2),
                                    weight_decay=self.weight_decay)
        elif self.optimizer_type == 'sgd':
            optimizer = optim.SGD(params, lr=self.lr, momentum=self.adam_beta1,
                                  weight_decay=self.weight_decay)
        else:
            optimizer = optim.Adam(params, lr=self.lr,
                                   betas=(self.adam_beta1, self.adam_beta2),
                                   weight_decay=self.weight_decay)
        logger.info(f"Optimizer: {self.optimizer_type} | weight_decay={self.weight_decay}")

        # Resume from checkpoint if provided
        start_epoch = 1
        if resume_from and Path(resume_from).exists():
            logger.info(f"\n⟳ Resuming from checkpoint: {resume_from}")
            checkpoint = self.load_checkpoint(resume_from, optimizer=optimizer)
            start_epoch = checkpoint.get('epoch', 0) + 1
            self.global_step = checkpoint.get('global_step', 0)
            if 'val_loss' in checkpoint:
                self.best_val_loss = checkpoint['val_loss']
            if 'val_mae_px' in checkpoint:
                self.best_dme_mae_px = checkpoint['val_mae_px']

            # Verify max_blur consistency
            _bt = self.blur_term.lower()
            ckpt_max = checkpoint.get('max_blur', checkpoint.get('max_coc'))
            if ckpt_max is not None:
                if abs(ckpt_max - self.max_blur) > 0.01:
                    logger.warning(f"WARNING:max {_bt} mismatch!")
                    logger.warning(f"  Checkpoint max {_bt}: {ckpt_max:.4f} px")
                    logger.warning(f"  Current max {_bt}:    {self.max_blur:.4f} px")
                    logger.warning(f"  Using checkpoint value for consistency")
                    self.max_blur = ckpt_max
                    # Update loss function with checkpoint max_blur
                    self.dme_loss_fn = DMELoss(max_blur=self.max_blur, eps=self.log_eps)
                else:
                    logger.info(f"OK:max {_bt} verified: {self.max_blur:.4f} px")

            # Handle LR: use checkpoint LR by default, or override if requested
            checkpoint_lr = optimizer.param_groups[0]['lr']
            if self.override_checkpoint_lr:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.lr
                logger.info(
                    f"Resuming from epoch {start_epoch}, training for {epochs} more epochs with LR={self.lr:.2e} (overriding checkpoint LR={checkpoint_lr:.2e})")
            else:
                self.lr = checkpoint_lr  # Use checkpoint LR
                logger.info(
                    f"Resuming from epoch {start_epoch}, training for {epochs} more epochs with LR={checkpoint_lr:.2e} (from checkpoint)")
        elif resume_from:
            logger.warning(f"WARNING:Checkpoint not found: {resume_from}, starting from scratch")

        target_epoch = start_epoch + epochs - 1
        bins = self._get_bins()
        bin_labels = [f"{low:.1f}-{high:.1f}" for low, high in bins]

        for epoch in range(start_epoch, target_epoch + 1):
            # Update learning rate
            lr = self._update_lr(optimizer, epoch, self.lr)

            self.model.dme_subnet.train()
            train_loss = 0.0

            # Binned MAE tracking for training
            train_bin_errors = [[] for _ in bins]

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{target_epoch}")
            batch_idx = 0
            for blur_img, blur_norm, blur_px in pbar:
                # Check for stop request
                if self._should_stop():
                    logger.warning("\nWARNING:  Training stopped by user")
                    return

                try:
                    blur_img = blur_img.to(self.device)
                    blur_norm = blur_norm.to(self.device)
                    blur_px = blur_px.to(self.device)

                    optimizer.zero_grad()

                    # Forward — model returns (B, 1) scalar in [0, 1]
                    pred_norm = self.model(blur_img)
                    loss = self.dme_loss_fn(pred_norm, blur_norm)

                    # Backward
                    loss.backward()
                    if self.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.dme_subnet.parameters(), self.grad_clip_norm)
                    optimizer.step()

                    train_loss += loss.item()

                    # Track binned MAE during training
                    with torch.no_grad():
                        pred_px = pred_norm * self.max_blur
                        gt_px = blur_px.view_as(pred_px)

                        errors = torch.abs(pred_px - gt_px).cpu().numpy().flatten()
                        gt_values = gt_px.cpu().numpy().flatten()

                        for i, (low, high) in enumerate(bins):
                            mask = (gt_values >= low) & (gt_values < high)
                            if mask.any():
                                train_bin_errors[i].extend(errors[mask].tolist())

                    pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{lr:.6f}'})

                    self.global_step += 1
                    batch_idx += 1

                    # Clear references
                    del blur_img, blur_norm, blur_px, pred_norm, loss

                except RuntimeError as e:
                    if 'out of memory' in str(e).lower() or 'CUDA' in str(e):
                        sample_range = f"{batch_idx * self.batch_size} - {(batch_idx + 1) * self.batch_size}"
                        logger.error(
                            f"\nWARNING:CUDA Error at batch {batch_idx} (samples {sample_range})")
                        logger.error(f"Error: {e}")
                        logger.error(
                            "Try: 1) Reduce batch size, 2) Use CPU, 3) Check images around this index")
                        raise
                    else:
                        raise

            train_loss /= len(train_loader)

            # Calculate training binned MAEs
            train_bin_maes = []
            train_weighted_mae = 0.0
            for i, (low, high) in enumerate(bins):
                if train_bin_errors[i]:
                    train_bin_mae = np.mean(train_bin_errors[i])
                    train_bin_maes.append(train_bin_mae)
                    train_weighted_mae += self.bin_weights[i] * train_bin_mae
                else:
                    train_bin_maes.append(0.0)

            val_loss, val_mae_px, val_bin_maes, val_weighted_mae = self._validate_dme(val_loader)

            self.writer.add_scalar('DME/train_loss', train_loss, epoch)
            self.writer.add_scalar('DME/train_weighted_mae', train_weighted_mae, epoch)
            self.writer.add_scalar('DME/val_loss', val_loss, epoch)
            self.writer.add_scalar('DME/val_mae_px', val_mae_px, epoch)
            self.writer.add_scalar('DME/val_weighted_mae', val_weighted_mae, epoch)
            self.writer.add_scalar('DME/learning_rate', lr, epoch)

            # Log binned MAEs for both train and val
            for i, ((low, high), train_mae, val_mae) in enumerate(zip(bins, train_bin_maes, val_bin_maes)):
                self.writer.add_scalar(
                    f'DME/train_mae_bin_{low:.1f}-{high:.1f}px', train_mae, epoch)
                self.writer.add_scalar(f'DME/val_mae_bin_{low:.1f}-{high:.1f}px', val_mae, epoch)

            self.curve_history['epochs'].append(epoch)
            self.curve_history['train_loss'].append(train_loss)
            self.curve_history['val_loss'].append(val_loss)
            self.curve_history['train_weighted_mae'].append(train_weighted_mae)
            self.curve_history['val_weighted_mae'].append(val_weighted_mae)
            self.curve_history['lr'].append(lr)
            self.curve_history['train_bin_maes'].append(list(train_bin_maes))
            self.curve_history['val_bin_maes'].append(list(val_bin_maes))

            logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            logger.info(
                f"  Train Weighted MAE: {train_weighted_mae:.2f} px | Val Weighted MAE: {val_weighted_mae:.2f} px")
            logger.info(
                f"  Train Binned: [{bin_labels[0]}: {train_bin_maes[0]:.2f}, {bin_labels[1]}: {train_bin_maes[1]:.2f}, {bin_labels[2]}: {train_bin_maes[2]:.2f}, {bin_labels[3]}: {train_bin_maes[3]:.2f}] px")
            logger.info(
                f"  Val Binned:   [{bin_labels[0]}: {val_bin_maes[0]:.2f}, {bin_labels[1]}: {val_bin_maes[1]:.2f}, {bin_labels[2]}: {val_bin_maes[2]:.2f}, {bin_labels[3]}: {val_bin_maes[3]:.2f}] px")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self._update_session_info('dme', epoch, val_loss)

            # Save checkpoint (with optimizer for proper resumption)
            if not self.save_only_best and (epoch % self.save_every == 0 or epoch == target_epoch):
                self._save_checkpoint(f'dme_epoch_{epoch}.pth', epoch, optimizer=optimizer)

            # Save best overall (global best across all training runs) - based on WEIGHTED MAE
            if val_weighted_mae < self.best_dme_mae_px:
                self.best_dme_mae_px = val_weighted_mae
                # keep best val_loss for reference
                self.best_val_loss = val_loss
                self._save_checkpoint('dme_best.pth', epoch, optimizer=optimizer,
                                      val_loss=val_loss, val_mae_px=val_weighted_mae)
                logger.info(f"  → New global best (weighted MAE): {val_weighted_mae:.4f} px")

                # Export training curves as PNG on best epochs (after warmup)
                if epoch >= 15:
                    self._save_training_curves(epoch, bins)

            # Save best for current training session (resets each time you start training)
            if val_weighted_mae < self.best_current_run_mae:
                self.best_current_run_mae = val_weighted_mae
                session_checkpoint_name = f'dme_best_session_{self.session_timestamp}.pth'
                self._save_checkpoint(
                    session_checkpoint_name, epoch, optimizer=optimizer, val_loss=val_loss,
                    val_mae_px=val_weighted_mae)
                logger.info(f"  → New session best (weighted MAE): {val_weighted_mae:.4f} px")

            self._save_training_history()

        # Finalize session when training ends
        self._finalize_session("DME training session completed")

    def _validate_dme(self, val_loader: DataLoader) -> Tuple[float, float, list, float]:
        """
        Validate DME-subnet with binned MAE tracking.

        Returns:
            (val_loss, val_mae_px, bin_maes, weighted_mae)
        """
        self.model.dme_subnet.eval()
        val_loss = 0.0
        mae_sum = 0.0
        n = 0

        # Binned MAE tracking (aligned with Beta distribution intervals)
        bins = self._get_bins()
        bin_errors = [[] for _ in bins]

        with torch.no_grad():
            for blur_img, blur_norm, blur_px in val_loader:
                blur_img = blur_img.to(self.device)
                blur_norm = blur_norm.to(self.device)
                blur_px = blur_px.to(self.device)

                # Forward — model returns (B, 1) scalar in [0, 1]
                pred_norm = self.model(blur_img)
                loss = self.dme_loss_fn(pred_norm, blur_norm)
                val_loss += loss.item()

                pred_px = pred_norm * self.max_blur
                gt_px = blur_px.view_as(pred_px)

                # Overall MAE
                mae_sum += torch.abs(pred_px - gt_px).sum().item()
                n += pred_px.numel()

                # Binned MAE
                errors = torch.abs(pred_px - gt_px).cpu().numpy().flatten()
                gt_values = gt_px.cpu().numpy().flatten()

                for i, (low, high) in enumerate(bins):
                    mask = (gt_values >= low) & (gt_values < high)
                    if mask.any():
                        bin_errors[i].extend(errors[mask].tolist())

        val_loss = val_loss / max(1, len(val_loader))
        val_mae_px = mae_sum / max(1, n)

        # Calculate binned MAEs and weighted MAE
        bin_maes = []
        weighted_mae = 0.0
        for i, (low, high) in enumerate(bins):
            if bin_errors[i]:
                bin_mae = np.mean(bin_errors[i])
                bin_maes.append(bin_mae)
                weighted_mae += self.bin_weights[i] * bin_mae
            else:
                bin_maes.append(0.0)

        return val_loss, val_mae_px, bin_maes, weighted_mae

    def _save_training_curves(self, epoch: int, bins: list):
        """Export training curves as publication-quality PNGs."""
        h = self.curve_history
        if len(h['epochs']) < 2:
            return

        epochs = h['epochs']
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Panel 1: Loss curves
        ax = axes[0, 0]
        ax.plot(epochs, h['train_loss'], label='Train', alpha=0.8)
        ax.plot(epochs, h['val_loss'], label='Val', alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Log-space MSE Loss')
        ax.set_title('Training & Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Panel 2: Weighted MAE curves
        ax = axes[0, 1]
        ax.plot(epochs, h['train_weighted_mae'], label='Train', alpha=0.8)
        ax.plot(epochs, h['val_weighted_mae'], label='Val', alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Weighted MAE (px)')
        ax.set_title('Weighted MAE')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Panel 3: Per-bin validation MAE
        ax = axes[1, 0]
        val_bins = np.array(h['val_bin_maes'])  # (n_epochs, 4)
        bin_labels = [f"{low:.1f}-{high:.1f}" for low, high in bins]
        for i in range(4):
            ax.plot(epochs, val_bins[:, i], label=f'{bin_labels[i]} px', alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MAE (px)')
        ax.set_title('Validation MAE by Blur Bin')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Panel 4: Learning rate
        ax = axes[1, 1]
        ax.plot(epochs, h['lr'], color='tab:orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(-4, -4))
        ax.grid(True, alpha=0.3)

        fig.suptitle(f'Training Curves — Best at Epoch {epoch} '
                     f'(Val WMAE: {h["val_weighted_mae"][-1]:.2f} px)',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()

        out_path = self.logs_dir / 'training_curves.png'
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"  → Saved training curves: {out_path}")

    def _save_checkpoint(
        self,
        filename: str,
        epoch: int,
        optimizer: Optional[optim.Optimizer] = None,
        val_loss: float = None,
        val_mae_px: float = None
    ) -> None:
        """Save model checkpoint with optimizer state."""
        train_cfg = self.config.get('training', {})
        training_mode = train_cfg.get('training_mode', 'optical')

        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'config': self.config,
            'max_blur': self.max_blur,
            'max_coc': self.max_blur,  # Backward compat — old code reads this key
            'log_eps': self.log_eps,
            'training_mode': training_mode,
            'dme_state_dict': self.model.dme_subnet.state_dict(),
        }

        if val_loss is not None:
            checkpoint['val_loss'] = val_loss
        if val_mae_px is not None:
            checkpoint['val_mae_px'] = float(val_mae_px)

        # Save optimizer state (critical for resuming training)
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        path = self.checkpoints_dir / filename
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(
        self,
        path: str,
        optimizer: Optional[optim.Optimizer] = None
    ) -> dict:
        """
        Load model from checkpoint.

        Returns:
            checkpoint dict (contains epoch, optimizer state, etc.)
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            # Old full-model checkpoint — try to extract DME weights
            state = checkpoint['model_state_dict']
            dme_state = {
                k.replace('dme_subnet.', ''): v for k, v in state.items()
                if k.startswith('dme_subnet.')}
            if dme_state:
                self.model.dme_subnet.load_state_dict(dme_state)
                logger.info(f"Loaded DME-subnet from full model checkpoint: {path}")
            else:
                logger.warning(f"WARNING:No DME weights found in full model checkpoint: {path}")
        elif 'dme_state_dict' in checkpoint:
            self.model.dme_subnet.load_state_dict(checkpoint['dme_state_dict'])
            logger.info(f"Loaded DME-subnet from {path}")
        else:
            raise KeyError(f"Checkpoint has unknown format. Keys: {list(checkpoint.keys())}")

        # Load optimizer state if provided and checkpoint is DME-compatible
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info(f"Restored optimizer state (epoch {checkpoint.get('epoch', '?')})")
            except Exception as e:
                logger.warning(f"WARNING:Could not restore optimizer state: {e}")
                logger.warning("  Continuing with fresh optimizer state")

        # Validate training mode consistency
        checkpoint_mode = checkpoint.get('training_mode', 'optical')
        current_mode = self.config.get('training', {}).get('training_mode', 'optical')
        if checkpoint_mode != current_mode:
            logger.warning(f"WARNING:Training mode mismatch!")
            logger.warning(f"  Checkpoint mode: {checkpoint_mode}")
            logger.warning(f"  Current config mode: {current_mode}")
            logger.warning(f"  Resuming with checkpoint mode: {checkpoint_mode}")

        return checkpoint

    def train(self, resume_from: Optional[str] = None) -> None:
        """Run DME training with optional resumption.

        Args:
            resume_from: Path to checkpoint to resume from.
        """
        logger.info("\nLoading data...")

        self._validate_data()

        logger.info(f"\nDME batch size: {self.batch_size}")
        logger.info(f"Data loader workers: {_NUM_WORKERS}")
        dme_train, dme_val = create_dme_dataloaders(
            self.data_dir,
            batch_size=self.batch_size,
            train_split=self.train_split,
            max_blur=self.max_blur,
            num_workers=_NUM_WORKERS,
            seed=self.seed,
            persistent_workers=_NUM_WORKERS > 0,
            pin_memory=torch.cuda.is_available(),
            stratified=self.stratified
        )

        # Save the resolved config and an initial run_metadata.json
        self._save_training_config_yaml()
        self.write_run_metadata(status='started')

        try:
            self.train_dme(dme_train, dme_val, self.epochs_dme, resume_from=resume_from)
            final_status = 'completed'
        except KeyboardInterrupt:
            final_status = 'interrupted'
            raise
        except Exception:
            final_status = 'failed'
            raise
        finally:
            try:
                self.write_run_metadata(status=final_status if 'final_status' in locals() else 'failed')
            except Exception as e:
                logger.warning(f"Failed to write run_metadata.json: {e}")

        logger.info("\n" + "=" * 60)
        logger.info("Training complete!")
        logger.info(f"Best model saved to: {self.checkpoints_dir / 'dme_best.pth'}")
        logger.info("=" * 60)

        self.writer.close()

    def _save_training_config_yaml(self) -> None:
        """Persist the full resolved config as <run_dir>/training_config.yaml."""
        try:
            with open(self.output_dir / 'training_config.yaml', 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            logger.warning(f"Failed to save training_config.yaml: {e}")

    def train_dme_only(
        self,
        checkpoint_preference: str = 'best',
        explicit_checkpoint: Optional[str] = None,
        force_fresh_start: bool = False
    ) -> None:
        """
        Train only the DME-subnet. Used by the GUI.

        Args:
            checkpoint_preference: 'best', 'latest', or 'fresh' (ignored if explicit_checkpoint provided)
            explicit_checkpoint: Explicit path to checkpoint file (overrides auto-detection)
            force_fresh_start: If True, train from scratch regardless of available checkpoints
        """
        logger.info("\nLoading data for DME training...")
        self._validate_data()

        logger.info(f"\nDME batch size: {self.batch_size}")
        logger.info(f"Data loader workers: {_NUM_WORKERS}")
        dme_train, dme_val = create_dme_dataloaders(
            self.data_dir,
            batch_size=self.batch_size,
            train_split=self.train_split,
            num_workers=_NUM_WORKERS,
            max_blur=self.max_blur,
            seed=self.seed,
            persistent_workers=_NUM_WORKERS > 0,
            pin_memory=torch.cuda.is_available()
        )

        logger.info("\n" + "=" * 60)
        logger.info("Training DME-subnet (scalar head)")
        logger.info("=" * 60)

        # Dump the resolved config so every model folder carries a readable
        # record of what produced it.
        self._save_training_config_yaml()

        # Use explicit checkpoint if provided, otherwise auto-detect
        if force_fresh_start:
            logger.info("Training from scratch (no checkpoint loaded)")
            resume_checkpoint = None
        elif explicit_checkpoint:
            checkpoint_path = Path(explicit_checkpoint)
            if checkpoint_path.exists():
                logger.info(f"Using specified checkpoint: {checkpoint_path.name}")
                resume_checkpoint = str(checkpoint_path)
            else:
                logger.warning(f"WARNING:Specified checkpoint not found: {explicit_checkpoint}")
                logger.info("Falling back to auto-detection")
                resume_checkpoint = self._find_latest_checkpoint(
                    'dme', preference=checkpoint_preference)
        else:
            # Auto-detect based on preference
            resume_checkpoint = self._find_latest_checkpoint(
                'dme', preference=checkpoint_preference)

        self.train_dme(dme_train, dme_val, self.epochs_dme, resume_from=resume_checkpoint)

    def _find_latest_checkpoint(self, mode: str = 'dme', preference: str = 'best') -> Optional[str]:
        """
        Find a checkpoint for resuming training.

        Strategy:
        1. If preference='best': Look for stage-specific "best" checkpoint
        2. If preference='latest': Look for most recent epoch checkpoint
        3. If preference='fresh': Return None (start from scratch)
        4. Fall back to generic best if no stage-specific best exists

        Args:
            mode: 'dme'
            preference: 'best', 'latest', or 'fresh'

        Returns:
            Path to checkpoint or None
        """
        # Handle fresh start request
        if preference == 'fresh':
            logger.info("Starting fresh training (no checkpoint loaded)")
            return None

        # Determine checkpoint paths
        best_checkpoint = self.checkpoints_dir / 'dme_best.pth'
        epoch_pattern = 'dme_epoch_*.pth'

        checkpoint_to_use = None

        # Find checkpoint based on preference
        if preference == 'latest':
            # Find most recent epoch checkpoint
            epoch_checkpoints = list(self.checkpoints_dir.glob(epoch_pattern))
            if epoch_checkpoints:
                # Sort by epoch number (extract from filename)
                def get_epoch_num(path):
                    try:
                        return int(path.stem.split('_')[-1])
                    except (ValueError, IndexError):
                        return 0
                checkpoint_to_use = max(epoch_checkpoints, key=get_epoch_num)
                logger.info(f"Resuming from latest epoch checkpoint: {checkpoint_to_use.name}")
            elif best_checkpoint.exists():
                checkpoint_to_use = best_checkpoint
                logger.info(
                    f"No epoch checkpoints found, using best checkpoint: {checkpoint_to_use.name}")
            else:
                logger.info(f"No checkpoints found, starting fresh")
                return None
        else:  # preference == 'best'
            if best_checkpoint.exists():
                checkpoint_to_use = best_checkpoint
                logger.info(f"Resuming from best checkpoint: {checkpoint_to_use.name}")
            else:
                backup_patterns = ['dme_best_old.pth', 'dme_best_backup.pth', 'dme_best_v*.pth']
                for pattern in backup_patterns:
                    matches = list(self.checkpoints_dir.glob(pattern))
                    if matches:
                        checkpoint_to_use = max(matches, key=lambda p: p.stat().st_mtime)
                        break

                if checkpoint_to_use:
                    logger.info(f"Using backup checkpoint: {checkpoint_to_use.name}")

        if checkpoint_to_use:
            # Load and display checkpoint info
            try:
                checkpoint = torch.load(checkpoint_to_use, map_location='cpu', weights_only=True)
                epoch = checkpoint.get('epoch', 'unknown')
                val_mae = checkpoint.get('val_mae_px', None)
                logger.info(f"  • Epoch: {epoch}")
                if val_mae is not None:
                    logger.info(f"  • Val MAE: {val_mae:.4f} px")
            except Exception as e:
                logger.warning(f"  WARNING:Could not read checkpoint metadata: {e}")

            return str(checkpoint_to_use)

        # Fallback: look for any epoch checkpoint
        stage_checkpoints = list(self.checkpoints_dir.glob(epoch_pattern))
        if stage_checkpoints:
            latest = max(stage_checkpoints, key=lambda p: p.stat().st_mtime)
            logger.info(f"No best checkpoint, using latest: {latest.name}")
            return str(latest)

        return None

    def _validate_data(self):
        """Check a sample of images for consistent sizes."""
        import cv2
        blur_dir = self.data_dir / 'blur'
        images = sorted(list(blur_dir.glob('*.png')))[:100]  # Check first 100

        sizes = set()
        for img_path in images:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                sizes.add(img.shape)

        if len(sizes) > 1:
            logger.warning(f"WARNING:Mixed image sizes detected: {sizes}")
        else:
            size = list(sizes)[0]
            logger.info(f"Data validated: {len(images)} samples checked, all {size[1]}×{size[0]}")

        # CUDA sanity check
        if self.device.type == 'cuda':
            logger.info("Running CUDA sanity check...")
            try:
                test_tensor = torch.randn(2, 1, size[0], size[1], device=self.device)
                test_out = self.model(test_tensor)
                out_shape = test_out.shape
                del test_tensor, test_out
                torch.cuda.empty_cache()
                logger.info(f"CUDA sanity check passed! Output shape: {out_shape}")
            except Exception as e:
                logger.error(f"WARNING:CUDA sanity check FAILED: {e}")
                logger.info("Falling back to CPU...")
                self.device = torch.device('cpu')
                self.model = self.model.to(self.device)

    def load_dme_checkpoint(self, checkpoint_path: str) -> None:
        """Load weights from checkpoint (handles various formats)."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            # Old full-model checkpoint — extract DME weights
            state = checkpoint['model_state_dict']
            dme_state = {
                k.replace('dme_subnet.', ''): v for k, v in state.items()
                if k.startswith('dme_subnet.')}
            if dme_state:
                self.model.dme_subnet.load_state_dict(dme_state)
                logger.info("Loaded DME-subnet from full model checkpoint (old format)")
            else:
                raise KeyError("No DME weights found in full model checkpoint")
        elif 'dme_state_dict' in checkpoint:
            self.model.dme_subnet.load_state_dict(checkpoint['dme_state_dict'])
            logger.info("Loaded DME-subnet checkpoint")
        else:
            raise KeyError(f"Checkpoint has unknown format. Keys: {list(checkpoint.keys())}")

        logger.info(f"  From epoch {checkpoint.get('epoch', '?')}")
        if 'val_loss' in checkpoint:
            self.best_val_loss = float(checkpoint.get('val_loss', float('inf')))
            self.best_dme_mae_px = float(checkpoint.get('val_mae_px', float('inf')))
            logger.info(f"  val_loss was: {checkpoint['val_loss']:.4f}")
            if 'val_mae_px' in checkpoint:
                logger.info(f"  val_mae_px was: {checkpoint['val_mae_px']:.2f}")

    def write_run_metadata(self, status: str, started_at: Optional[str] = None,
                           run_name: Optional[str] = None) -> None:
        """Write run_metadata.json describing this training run.

        Called at the start (status='started') and end (status='completed' or
        'interrupted' or 'failed') of training. Each call rewrites the file.
        """
        import json
        import platform as _platform
        from datetime import datetime

        run_dir = self.output_dir
        meta_path = self.logs_dir / 'run_metadata.json'

        # Preserve started_at across calls
        existing = {}
        if meta_path.is_file():
            try:
                with open(meta_path) as f:
                    existing = json.load(f)
            except Exception:
                existing = {}

        if started_at is None:
            started_at = existing.get('started_at') or datetime.now().isoformat(timespec='seconds')
        completed_at = datetime.now().isoformat(timespec='seconds') if status != 'started' else None
        duration_min = None
        if completed_at:
            try:
                t0 = datetime.fromisoformat(started_at)
                t1 = datetime.fromisoformat(completed_at)
                duration_min = round((t1 - t0).total_seconds() / 60.0, 2)
            except Exception:
                pass

        torch_version = getattr(torch, '__version__', 'unknown')
        cuda_available = bool(getattr(torch, 'cuda', None) and torch.cuda.is_available())

        # Hyperparameters are the source-of-truth in training_config.yaml next
        # to this file — don't duplicate them here. Keep only the fields that
        # make metadata useful without opening the config:
        #   * provenance / status / timing
        #   * training_mode + dataset summary (cheap to keep, enables jq-style
        #     filtering across many runs without parsing each config.yaml)
        #   * runtime environment (not in config.yaml at all)
        meta = {
            'run_name': run_name or existing.get('run_name') or run_dir.name,
            'run_id': run_dir.name,
            'started_at': started_at,
            'completed_at': completed_at,
            'duration_minutes': duration_min,
            'status': status,
            'training_mode': self.training_mode,
            'dataset_path': str(self.data_dir),
            'dataset_name': self.data_dir.name,
            'environment': {
                'python': _platform.python_version(),
                'torch': torch_version,
                'cuda_available': cuda_available,
                'platform': f"{_platform.system()}-{_platform.machine()}",
            },
        }

        # Pull dataset summary if present
        summary_path = self.data_dir / 'dataset_summary.json'
        if summary_path.is_file():
            try:
                with open(summary_path) as f:
                    ds = json.load(f)
                meta['dataset_n_samples'] = ds.get('n_samples')
                meta['dataset_blur_range_px'] = ds.get('blur_range_px')
            except Exception:
                pass

        if status != 'started':
            meta['results'] = {
                'best_val_mae_px': (
                    float(self.best_dme_mae_px)
                    if self.best_dme_mae_px != float('inf') else None
                ),
                'best_val_loss': (
                    float(self.best_val_loss)
                    if self.best_val_loss != float('inf') else None
                ),
                'epochs_trained': self.current_session.get('epochs_trained', 0),
                'best_epoch': self.current_session.get('best_epoch'),
            }
            ckpts = sorted(self.checkpoints_dir.glob('*.pth'))
            meta['checkpoints'] = [str(p.relative_to(run_dir)) for p in ckpts]

        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2, default=str)


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Train defocus estimation network (scalar head)")

    parser.add_argument(
        '--config', '-c',
        type=str,
        default='training_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--data-dir', '-d',
        type=str,
        default='data/synthetic',
        help='Path to training data directory'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='Output directory (default: checkpoints/TIMESTAMP)'
    )
    parser.add_argument(
        '--resume', '-r',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use'
    )

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path('checkpoints') / timestamp
    else:
        output_dir = Path(args.output_dir)

    trainer = Trainer(
        config=config,
        data_dir=args.data_dir,
        output_dir=output_dir,
        device=args.device
    )

    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()
