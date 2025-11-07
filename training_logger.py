"""
Training Logger Module for GEM Project

Provides structured logging for training runs with JSON Lines format
for crash-safe, append-only logging and automatic CSV export.
"""

import os
import json
import csv
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path


class TrainingLogger:
    """
    Structured logger for training runs.

    Logs training metrics to JSON Lines format (append-only, crash-safe)
    and provides CSV export for visualization.

    Features:
    - JSON Lines format for reliability
    - Automatic CSV export
    - Query interface for finding best checkpoints
    - Compatible with TensorBoard (no duplication)
    """

    def __init__(self, log_dir: str, resume: bool = False):
        """
        Initialize training logger.

        Args:
            log_dir: Directory to save log files
            resume: If True, append to existing log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.jsonl_path = self.log_dir / 'training_log.jsonl'
        self.csv_path = self.log_dir / 'training_log.csv'

        self.resume = resume
        self.history = []

        # Load existing history if resuming
        if resume and self.jsonl_path.exists():
            self._load_history()
        elif not resume and self.jsonl_path.exists():
            # Backup old log if starting fresh
            backup_path = self.log_dir / f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jsonl'
            os.rename(self.jsonl_path, backup_path)
            print(f"[TrainingLogger] Backed up old log to {backup_path}")

    def _load_history(self):
        """Load existing training history from JSON Lines file."""
        try:
            with open(self.jsonl_path, 'r') as f:
                for line in f:
                    if line.strip():
                        self.history.append(json.loads(line))
            print(f"[TrainingLogger] Loaded {len(self.history)} existing log entries")
        except Exception as e:
            print(f"[TrainingLogger] Warning: Could not load history: {e}")
            self.history = []

    def log_step(self, step: int, epoch: int, losses: Dict[str, float],
                 phase: str = 'train', **kwargs):
        """
        Log a training or validation step.

        Args:
            step: Global step number
            epoch: Current epoch
            losses: Dictionary of loss values (e.g., {'total': 0.5, 'l1': 0.3})
            phase: 'train' or 'val'
            **kwargs: Additional metrics to log (learning_rate, grad_norm, etc.)
        """
        entry = {
            'step': step,
            'epoch': epoch,
            'phase': phase,
            'timestamp': datetime.now().isoformat(),
            'losses': losses,
            **kwargs
        }

        self._write_entry(entry)
        self.history.append(entry)

    def log_validation(self, step: int, epoch: int, val_losses: Dict[str, float], **kwargs):
        """
        Log validation metrics.

        Args:
            step: Global step number
            epoch: Current epoch
            val_losses: Dictionary of validation loss values
            **kwargs: Additional validation metrics
        """
        self.log_step(step, epoch, val_losses, phase='val', **kwargs)

    def log_checkpoint(self, step: int, epoch: int, checkpoint_path: str,
                      val_loss: Optional[float] = None, **kwargs):
        """
        Log checkpoint save event.

        Args:
            step: Global step number
            epoch: Current epoch
            checkpoint_path: Path to saved checkpoint
            val_loss: Optional validation loss
            **kwargs: Additional metadata
        """
        entry = {
            'step': step,
            'epoch': epoch,
            'phase': 'checkpoint',
            'timestamp': datetime.now().isoformat(),
            'checkpoint_path': checkpoint_path,
            'val_loss': val_loss,
            **kwargs
        }

        self._write_entry(entry)
        self.history.append(entry)

    def _write_entry(self, entry: Dict[str, Any]):
        """Write a single entry to the JSON Lines file."""
        with open(self.jsonl_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def get_last_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recent checkpoint entry.

        Returns:
            Dictionary with checkpoint info or None if no checkpoints logged
        """
        checkpoints = [e for e in self.history if e.get('phase') == 'checkpoint']
        return checkpoints[-1] if checkpoints else None

    def get_best_checkpoint(self, metric: str = 'val_loss', minimize: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get the checkpoint with the best validation metric.

        Args:
            metric: Metric to use for comparison (default: 'val_loss')
            minimize: If True, lower is better; if False, higher is better

        Returns:
            Dictionary with best checkpoint info or None
        """
        checkpoints = [e for e in self.history
                      if e.get('phase') == 'checkpoint' and metric in e]

        if not checkpoints:
            return None

        if minimize:
            return min(checkpoints, key=lambda x: x[metric])
        else:
            return max(checkpoints, key=lambda x: x[metric])

    def get_training_history(self, phase: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get training history, optionally filtered by phase.

        Args:
            phase: Optional phase filter ('train', 'val', 'checkpoint')

        Returns:
            List of log entries
        """
        if phase is None:
            return self.history
        return [e for e in self.history if e.get('phase') == phase]

    def export_to_csv(self):
        """
        Export training history to CSV format.

        Creates a flattened CSV with columns for step, epoch, phase, timestamp,
        and all loss/metric values.
        """
        if not self.history:
            print("[TrainingLogger] No history to export")
            return

        # Collect all possible keys from all entries
        all_keys = set()
        for entry in self.history:
            all_keys.update(entry.keys())
            # Flatten nested 'losses' dict
            if 'losses' in entry and isinstance(entry['losses'], dict):
                all_keys.update([f"loss_{k}" for k in entry['losses'].keys()])

        # Remove 'losses' since we'll flatten it
        all_keys.discard('losses')

        # Sort keys for consistent column order
        fieldnames = ['step', 'epoch', 'phase', 'timestamp']
        remaining_keys = sorted(all_keys - set(fieldnames))
        fieldnames.extend(remaining_keys)

        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()

            for entry in self.history:
                # Flatten the entry
                flat_entry = {k: v for k, v in entry.items() if k != 'losses'}

                # Add flattened losses
                if 'losses' in entry and isinstance(entry['losses'], dict):
                    for loss_key, loss_val in entry['losses'].items():
                        flat_entry[f'loss_{loss_key}'] = loss_val

                writer.writerow(flat_entry)

        print(f"[TrainingLogger] Exported {len(self.history)} entries to {self.csv_path}")

    def get_last_step(self) -> int:
        """
        Get the last logged step number.

        Returns:
            Last step number or 0 if no entries
        """
        if not self.history:
            return 0

        train_entries = [e for e in self.history if e.get('phase') in ['train', 'val']]
        if not train_entries:
            return 0

        return max(e.get('step', 0) for e in train_entries)

    def get_last_epoch(self) -> int:
        """
        Get the last logged epoch number.

        Returns:
            Last epoch number or 0 if no entries
        """
        if not self.history:
            return 0

        train_entries = [e for e in self.history if e.get('phase') in ['train', 'val']]
        if not train_entries:
            return 0

        return max(e.get('epoch', 0) for e in train_entries)

    def __repr__(self):
        return f"TrainingLogger(log_dir={self.log_dir}, entries={len(self.history)})"
