"""
Defocus Estimation Training Script — Scalar Head Architecture

Single-stage training: DME-subnet predicts scalar blur in [0, 1].
DD-subnet removed — scalar blur estimation only.

Usage:
    python train.py --config training_config.yaml --data-dir data/synthetic
"""

import platform
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np

# Windows + tkinter GUI causes multiprocessing issues; use workers on Linux/Mac only
_NUM_WORKERS = 0 if platform.system() == "Windows" else 4
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving PNGs
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import argparse
from datetime import datetime
from tqdm import tqdm
import shutil
from typing import Dict, Any, Optional, Tuple, List

from model import DefocusNet, model_summary
from dataset import create_dme_dataloaders, DMEDataset
from losses import DMELoss


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
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        print(f"Using device: {self.device}")

        # cuDNN settings for reproducibility
        if self.device.type == 'cuda':
            torch.backends.cudnn.deterministic = True  # Reproducible results
            # benchmark=True (default) allows cuDNN to find fastest convolution algorithms
            print("cuDNN: deterministic=True (reproducible mode)")

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
        self.writer = SummaryWriter(self.output_dir / 'logs')

        # Training history tracking
        self.history_file = self.output_dir / 'training_history.yaml'
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
        self.best_current_run_mae = float('inf')  # best MAE for current training session (resets on resume)
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
        """Save training history to file."""
        with open(self.history_file, 'w') as f:
            yaml.dump(self.training_history, f, default_flow_style=False, sort_keys=False)

    def _update_session_info(self, stage: str, epoch: int, val_loss: float = None):
        """Update current session info during training."""
        if self.current_session['start_epoch'] is None:
            self.current_session['start_epoch'] = epoch
            self.current_session['stage'] = stage

        self.current_session['end_epoch'] = epoch
        self.current_session['epochs_trained'] = epoch - self.current_session['start_epoch'] + 1

        # Track if this epoch improved the best model
        if val_loss is not None and (self.training_history['training_summary']['best_metrics']['val_loss'] is None or
                                     val_loss < self.training_history['training_summary']['best_metrics']['val_loss']):
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
        """Finalize current session and add to history."""
        if notes:
            self.current_session['notes'].append(notes)

        # Add session to history
        self.training_history['training_sessions'].append(self.current_session.copy())

        # Accumulate total epochs trained (cumulative across all sessions)
        self.training_history['training_summary']['total_epochs_trained'] += self.current_session['epochs_trained']

        # Save to file
        self._save_training_history()

    def _track_lr_change(self, epoch: int, old_lr: float, new_lr: float, reason: str):
        """Track learning rate changes."""
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
        """Calculate learning rate with decay."""
        if epoch < self.lr_decay_start:
            return base_lr
        else:
            decay_epochs = epoch - self.lr_decay_start
            return base_lr * (1.0 - self.lr_decay_rate) ** decay_epochs

    def _should_stop(self) -> bool:
        """Check if training should stop."""
        return self.stop_flag()

    def _calculate_bin_weights_from_beta(self) -> list:
        """Calculate bin weights from beta distribution parameters.

        Returns:
            List of 4 weights that sum to 1.0, representing the distribution
            of samples across equal-width blur bins.
        """
        data_cfg = self.config.get('data', {})
        blur_distribution = data_cfg.get('blur_distribution', data_cfg.get('coc_distribution'))
        beta_alpha = data_cfg.get('beta_alpha')
        beta_beta = data_cfg.get('beta_beta')

        # If no distribution type specified, infer from presence of beta parameters
        if blur_distribution is None:
            if beta_alpha is not None and beta_beta is not None:
                blur_distribution = 'weighted'
            else:
                # No distribution type and no beta params -> assume uniform
                print("ℹ️  No distribution type or beta parameters found in config")
                print("   Defaulting to uniform distribution: equal bin weights [0.25, 0.25, 0.25, 0.25]")
                return [0.25, 0.25, 0.25, 0.25]

        # If uniform distribution, use equal weights
        if blur_distribution == 'uniform':
            print("ℹ️  Using uniform distribution: equal bin weights [0.25, 0.25, 0.25, 0.25]")
            return [0.25, 0.25, 0.25, 0.25]

        # For weighted distribution, calculate from beta parameters
        if beta_alpha is None or beta_beta is None:
            print("⚠️  WARNING: Weighted distribution specified but beta parameters not found!")
            print("   Falling back to uniform weights [0.25, 0.25, 0.25, 0.25]")
            print("   To fix: Re-generate data with 'weighted' distribution and beta parameters")
            return [0.25, 0.25, 0.25, 0.25]

        try:
            from scipy import stats

            # Sample from beta distribution
            num_samples = 100000
            beta_samples = stats.beta.rvs(beta_alpha, beta_beta, size=num_samples)

            # Calculate distribution across 4 equal-width bins [0, 0.25), [0.25, 0.5), [0.5, 0.75), [0.75, 1.0]
            bin_edges = [0.0, 0.25, 0.5, 0.75, 1.0]
            weights = []

            for i in range(4):
                count = np.sum((beta_samples >= bin_edges[i]) & (beta_samples < bin_edges[i+1]))
                weight = count / num_samples
                weights.append(weight)

            # Normalize to ensure they sum to exactly 1.0
            total = sum(weights)
            weights = [w / total for w in weights]

            print(f"ℹ️  Calculated bin weights from β({beta_alpha:.3f}, {beta_beta:.3f}):")
            print(f"   Bin 1 (0-25%):   {weights[0]:.1%}")
            print(f"   Bin 2 (25-50%):  {weights[1]:.1%}")
            print(f"   Bin 3 (50-75%):  {weights[2]:.1%}")
            print(f"   Bin 4 (75-100%): {weights[3]:.1%}")

            return weights

        except ImportError:
            print("⚠️  WARNING: scipy not available, using default weights [0.40, 0.30, 0.20, 0.10]")
            return [0.40, 0.30, 0.20, 0.10]
        except Exception as e:
            print(f"⚠️  WARNING: Error calculating bin weights: {e}")
            print("   Using default weights [0.40, 0.30, 0.20, 0.10]")
            return [0.40, 0.30, 0.20, 0.10]

    def _update_lr(self, optimizer: optim.Optimizer, epoch: int, base_lr: float):
        """Update optimiser learning rate."""
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

        print(f"\n{'='*60}")
        print(f"{stage_name} Data Split Distribution")
        print(f"{'='*60}")
        print(f"Train: {len(train_loader.dataset):,} samples")
        print(f"Val:   {len(val_loader.dataset):,} samples")
        print(f"Seed:  {self.seed}")
        print(f"Method: {'Stratified (balanced by ' + self.blur_term + ' bins)' if self.stratified else 'Random (seed-based)'}")

        bins = self._get_bins()
        bin_labels = [f"{low:.1f}-{high:.1f}" for low, high in bins]

        # Analyze train distribution
        print(f"\nTrain Distribution by {self.blur_term} Bin:")
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
            print(f"  {label:6s} px: {count:6,d} samples ({pct:5.1f}%)")

        # Analyze val distribution
        print(f"\nVal Distribution by {self.blur_term} Bin:")
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
            print(f"  {label:6s} px: {count:6,d} samples ({pct:5.1f}%)")

        print(f"{'='*60}\n")

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
        print("\n" + "="*60)
        print("Training DME-subnet (scalar head)")
        print("="*60)
        print("\nCheckpoint Strategy:")
        print("  • dme_best.pth - Global best (never overwritten unless beaten)")
        print("  • dme_best_current_session.pth - Best in this training run")
        print("  • dme_epoch_X.pth - Recovery checkpoints (every epoch)")

        optimizer = optim.Adam(
            self.model.dme_subnet.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999)
        )

        # Resume from checkpoint if provided
        start_epoch = 1
        if resume_from and Path(resume_from).exists():
            print(f"\n⟳ Resuming from checkpoint: {resume_from}")
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
                    print(f"⚠ WARNING: max {_bt} mismatch!")
                    print(f"  Checkpoint max {_bt}: {ckpt_max:.4f} px")
                    print(f"  Current max {_bt}:    {self.max_blur:.4f} px")
                    print(f"  Using checkpoint value for consistency")
                    self.max_blur = ckpt_max
                    # Update loss function with checkpoint max_blur
                    self.dme_loss_fn = DMELoss(max_blur=self.max_blur, eps=self.log_eps)
                else:
                    print(f"✓ max {_bt} verified: {self.max_blur:.4f} px")

            # Handle LR: use checkpoint LR by default, or override if requested
            checkpoint_lr = optimizer.param_groups[0]['lr']
            if self.override_checkpoint_lr:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.lr
                print(f"Resuming from epoch {start_epoch}, training for {epochs} more epochs with LR={self.lr:.2e} (overriding checkpoint LR={checkpoint_lr:.2e})")
            else:
                self.lr = checkpoint_lr  # Use checkpoint LR
                print(f"Resuming from epoch {start_epoch}, training for {epochs} more epochs with LR={checkpoint_lr:.2e} (from checkpoint)")
        elif resume_from:
            print(f"⚠ Checkpoint not found: {resume_from}, starting from scratch")

        target_epoch = start_epoch + epochs - 1
        for epoch in range(start_epoch, target_epoch + 1):
            # Update learning rate
            lr = self._update_lr(optimizer, epoch, self.lr)

            # Training
            self.model.dme_subnet.train()
            train_loss = 0.0

            # Binned MAE tracking for training
            bins = self._get_bins()
            train_bin_errors = [[] for _ in bins]

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{target_epoch}")
            batch_idx = 0
            for blur_img, blur_norm, blur_px in pbar:
                # Check for stop request
                if self._should_stop():
                    print("\n⚠️  Training stopped by user")
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
                        print(f"\n⚠ CUDA Error at batch {batch_idx} (samples {sample_range})")
                        print(f"Error: {e}")
                        print("Try: 1) Reduce batch size, 2) Use CPU, 3) Check images around this index")
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

            # Validation
            val_loss, val_mae_px, val_bin_maes, val_weighted_mae = self._validate_dme(val_loader)

            # Logging
            self.writer.add_scalar('DME/train_loss', train_loss, epoch)
            self.writer.add_scalar('DME/train_weighted_mae', train_weighted_mae, epoch)
            self.writer.add_scalar('DME/val_loss', val_loss, epoch)
            self.writer.add_scalar('DME/val_mae_px', val_mae_px, epoch)
            self.writer.add_scalar('DME/val_weighted_mae', val_weighted_mae, epoch)
            self.writer.add_scalar('DME/learning_rate', lr, epoch)

            # Log binned MAEs for both train and val
            for i, ((low, high), train_mae, val_mae) in enumerate(zip(bins, train_bin_maes, val_bin_maes)):
                self.writer.add_scalar(f'DME/train_mae_bin_{low:.1f}-{high:.1f}px', train_mae, epoch)
                self.writer.add_scalar(f'DME/val_mae_bin_{low:.1f}-{high:.1f}px', val_mae, epoch)

            # Record curve history
            self.curve_history['epochs'].append(epoch)
            self.curve_history['train_loss'].append(train_loss)
            self.curve_history['val_loss'].append(val_loss)
            self.curve_history['train_weighted_mae'].append(train_weighted_mae)
            self.curve_history['val_weighted_mae'].append(val_weighted_mae)
            self.curve_history['lr'].append(lr)
            self.curve_history['train_bin_maes'].append(list(train_bin_maes))
            self.curve_history['val_bin_maes'].append(list(val_bin_maes))

            # Format bin labels dynamically
            bin_labels = [f"{low:.1f}-{high:.1f}" for low, high in bins]
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            print(f"  Train Weighted MAE: {train_weighted_mae:.2f} px | Val Weighted MAE: {val_weighted_mae:.2f} px")
            print(f"  Train Binned: [{bin_labels[0]}: {train_bin_maes[0]:.2f}, {bin_labels[1]}: {train_bin_maes[1]:.2f}, {bin_labels[2]}: {train_bin_maes[2]:.2f}, {bin_labels[3]}: {train_bin_maes[3]:.2f}] px")
            print(f"  Val Binned:   [{bin_labels[0]}: {val_bin_maes[0]:.2f}, {bin_labels[1]}: {val_bin_maes[1]:.2f}, {bin_labels[2]}: {val_bin_maes[2]:.2f}, {bin_labels[3]}: {val_bin_maes[3]:.2f}] px")

            # Clear cache between epochs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Update session tracking
            self._update_session_info('dme', epoch, val_loss)

            # Save checkpoint (with optimizer for proper resumption)
            if not self.save_only_best and (epoch % self.save_every == 0 or epoch == target_epoch):
                self._save_checkpoint(f'dme_epoch_{epoch}.pth', epoch, optimizer=optimizer)

            # Save best overall (global best across all training runs) - based on WEIGHTED MAE
            if val_weighted_mae < self.best_dme_mae_px:
                self.best_dme_mae_px = val_weighted_mae
                # keep best val_loss for reference
                self.best_val_loss = val_loss
                self._save_checkpoint('dme_best.pth', epoch, optimizer=optimizer, val_loss=val_loss, val_mae_px=val_weighted_mae)
                print(f"  → New global best (weighted MAE): {val_weighted_mae:.4f} px")

                # Export training curves as PNG on best epochs (after warmup)
                if epoch >= 15:
                    self._save_training_curves(epoch, bins)

            # Save best for current training session (resets each time you start training)
            if val_weighted_mae < self.best_current_run_mae:
                self.best_current_run_mae = val_weighted_mae
                session_checkpoint_name = f'dme_best_session_{self.session_timestamp}.pth'
                self._save_checkpoint(session_checkpoint_name, epoch, optimizer=optimizer, val_loss=val_loss, val_mae_px=val_weighted_mae)
                print(f"  → New session best (weighted MAE): {val_weighted_mae:.4f} px")

            # Save history after each epoch
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

        out_path = self.output_dir / 'training_curves.png'
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  → Saved training curves: {out_path}")

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

        path = self.output_dir / filename
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")

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
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            # Old full-model checkpoint — try to extract DME weights
            state = checkpoint['model_state_dict']
            dme_state = {k.replace('dme_subnet.', ''): v for k, v in state.items() if k.startswith('dme_subnet.')}
            if dme_state:
                self.model.dme_subnet.load_state_dict(dme_state)
                print(f"Loaded DME-subnet from full model checkpoint: {path}")
            else:
                print(f"⚠ No DME weights found in full model checkpoint: {path}")
        elif 'dme_state_dict' in checkpoint:
            self.model.dme_subnet.load_state_dict(checkpoint['dme_state_dict'])
            print(f"Loaded DME-subnet from {path}")
        else:
            raise KeyError(f"Checkpoint has unknown format. Keys: {list(checkpoint.keys())}")

        # Load optimizer state if provided and checkpoint is DME-compatible
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"Restored optimizer state (epoch {checkpoint.get('epoch', '?')})")
            except Exception as e:
                print(f"⚠ Could not restore optimizer state: {e}")
                print("  Continuing with fresh optimizer state")

        # Validate training mode consistency
        checkpoint_mode = checkpoint.get('training_mode', 'optical')
        current_mode = self.config.get('training', {}).get('training_mode', 'optical')
        if checkpoint_mode != current_mode:
            print(f"⚠ WARNING: Training mode mismatch!")
            print(f"  Checkpoint mode: {checkpoint_mode}")
            print(f"  Current config mode: {current_mode}")
            print(f"  Resuming with checkpoint mode: {checkpoint_mode}")

        return checkpoint

    def train(self, resume_from: Optional[str] = None) -> None:
        """Run DME training with optional resumption.

        Args:
            resume_from: Path to checkpoint to resume from.
        """
        # Create dataloaders
        print("\nLoading data...")

        # Validate data first
        self._validate_data()

        print(f"\nDME batch size: {self.batch_size}")
        print(f"Data loader workers: {_NUM_WORKERS}")
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

        self.train_dme(dme_train, dme_val, self.epochs_dme, resume_from=resume_from)

        print("\n" + "="*60)
        print("Training complete!")
        print(f"Best model saved to: {self.output_dir / 'dme_best.pth'}")
        print("="*60)

        self.writer.close()

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
        print("\nLoading data for DME training...")
        self._validate_data()

        print(f"\nDME batch size: {self.batch_size}")
        print(f"Data loader workers: {_NUM_WORKERS}")
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

        print("\n" + "=" * 60)
        print("Training DME-subnet (scalar head)")
        print("=" * 60)

        # Use explicit checkpoint if provided, otherwise auto-detect
        if force_fresh_start:
            print("ℹ Training from scratch (no checkpoint loaded)")
            resume_checkpoint = None
        elif explicit_checkpoint:
            checkpoint_path = Path(explicit_checkpoint)
            if checkpoint_path.exists():
                print(f"ℹ Using specified checkpoint: {checkpoint_path.name}")
                resume_checkpoint = str(checkpoint_path)
            else:
                print(f"⚠ Specified checkpoint not found: {explicit_checkpoint}")
                print("ℹ Falling back to auto-detection")
                resume_checkpoint = self._find_latest_checkpoint('dme', preference=checkpoint_preference)
        else:
            # Auto-detect based on preference
            resume_checkpoint = self._find_latest_checkpoint('dme', preference=checkpoint_preference)

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
            print("ℹ Starting fresh training (no checkpoint loaded)")
            return None

        # Determine checkpoint paths
        best_checkpoint = self.output_dir / 'dme_best.pth'
        epoch_pattern = 'dme_epoch_*.pth'

        checkpoint_to_use = None

        # Find checkpoint based on preference
        if preference == 'latest':
            # Find most recent epoch checkpoint
            epoch_checkpoints = list(self.output_dir.glob(epoch_pattern))
            if epoch_checkpoints:
                # Sort by epoch number (extract from filename)
                def get_epoch_num(path):
                    try:
                        return int(path.stem.split('_')[-1])
                    except (ValueError, IndexError):
                        return 0
                checkpoint_to_use = max(epoch_checkpoints, key=get_epoch_num)
                print(f"ℹ Resuming from latest epoch checkpoint: {checkpoint_to_use.name}")
            elif best_checkpoint.exists():
                checkpoint_to_use = best_checkpoint
                print(f"ℹ No epoch checkpoints found, using best checkpoint: {checkpoint_to_use.name}")
            else:
                print(f"ℹ No checkpoints found, starting fresh")
                return None
        else:  # preference == 'best'
            if best_checkpoint.exists():
                checkpoint_to_use = best_checkpoint
                print(f"ℹ Resuming from best checkpoint: {checkpoint_to_use.name}")
            else:
                backup_patterns = ['dme_best_old.pth', 'dme_best_backup.pth', 'dme_best_v*.pth']
                for pattern in backup_patterns:
                    matches = list(self.output_dir.glob(pattern))
                    if matches:
                        checkpoint_to_use = max(matches, key=lambda p: p.stat().st_mtime)
                        break

                if checkpoint_to_use:
                    print(f"ℹ Using backup checkpoint: {checkpoint_to_use.name}")

        if checkpoint_to_use:
            # Load and display checkpoint info
            try:
                checkpoint = torch.load(checkpoint_to_use, map_location='cpu', weights_only=False)
                epoch = checkpoint.get('epoch', 'unknown')
                val_mae = checkpoint.get('val_mae_px', None)
                print(f"  • Epoch: {epoch}")
                if val_mae is not None:
                    print(f"  • Val MAE: {val_mae:.4f} px")
            except Exception as e:
                print(f"  ⚠ Could not read checkpoint metadata: {e}")

            return str(checkpoint_to_use)

        # Fallback: look for any epoch checkpoint
        stage_checkpoints = list(self.output_dir.glob(epoch_pattern))
        if stage_checkpoints:
            latest = max(stage_checkpoints, key=lambda p: p.stat().st_mtime)
            print(f"ℹ No best checkpoint, using latest: {latest.name}")
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
            print(f"⚠ WARNING: Mixed image sizes detected: {sizes}")
        else:
            size = list(sizes)[0]
            print(f"Data validated: {len(images)} samples checked, all {size[1]}×{size[0]}")

        # CUDA sanity check
        if self.device.type == 'cuda':
            print("Running CUDA sanity check...")
            try:
                test_tensor = torch.randn(2, 1, size[0], size[1], device=self.device)
                test_out = self.model(test_tensor)
                out_shape = test_out.shape
                del test_tensor, test_out
                torch.cuda.empty_cache()
                print(f"CUDA sanity check passed! Output shape: {out_shape}")
            except Exception as e:
                print(f"⚠ CUDA sanity check FAILED: {e}")
                print("Falling back to CPU...")
                self.device = torch.device('cpu')
                self.model = self.model.to(self.device)

    def load_dme_checkpoint(self, checkpoint_path: str) -> None:
        """Load weights from checkpoint (handles various formats)."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            # Old full-model checkpoint — extract DME weights
            state = checkpoint['model_state_dict']
            dme_state = {k.replace('dme_subnet.', ''): v for k, v in state.items() if k.startswith('dme_subnet.')}
            if dme_state:
                self.model.dme_subnet.load_state_dict(dme_state)
                print("Loaded DME-subnet from full model checkpoint (old format)")
            else:
                raise KeyError("No DME weights found in full model checkpoint")
        elif 'dme_state_dict' in checkpoint:
            self.model.dme_subnet.load_state_dict(checkpoint['dme_state_dict'])
            print("Loaded DME-subnet checkpoint")
        else:
            raise KeyError(f"Checkpoint has unknown format. Keys: {list(checkpoint.keys())}")

        print(f"  From epoch {checkpoint.get('epoch', '?')}")
        if 'val_loss' in checkpoint:
            self.best_val_loss = float(checkpoint.get('val_loss', float('inf')))
            self.best_dme_mae_px = float(checkpoint.get('val_mae_px', float('inf')))
            print(f"  val_loss was: {checkpoint['val_loss']:.4f}")
            if 'val_mae_px' in checkpoint:
                print(f"  val_mae_px was: {checkpoint['val_mae_px']:.2f}")


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

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path('checkpoints') / timestamp
    else:
        output_dir = Path(args.output_dir)

    # Create trainer
    trainer = Trainer(
        config=config,
        data_dir=args.data_dir,
        output_dir=output_dir,
        device=args.device
    )

    # Train
    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()
