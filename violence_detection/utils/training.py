"""Training utilities for violence detection models."""

import logging
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to prevent overfitting.

    Monitors a metric and stops training if it doesn't improve
    for a specified number of epochs.
    """

    def __init__(self, patience: int = 7, min_delta: float = 0.0, mode: str = "max"):
        """Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change in the monitored quantity to qualify as an improvement
            mode: 'min' or 'max' depending on whether we want to minimize or maximize the metric

        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode

    def __call__(self, score: float) -> bool:
        """Check if training should be stopped.

        Args:
            score: Current value of the monitored metric

        Returns:
            True if training should be stopped, False otherwise

        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            if score <= self.best_score + self.min_delta:
                self.counter += 1
                logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        else:  # mode == 'min'
            if score >= self.best_score - self.min_delta:
                self.counter += 1
                logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0

        return self.early_stop


class FocalLoss(nn.Module):
    """Focal loss for imbalanced datasets.

    Common in violence detection due to limited violent examples.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        """Initialize focal loss parameters.

        Args:
            alpha: Weighting factor for the rare class
            gamma: Focusing parameter
            reduction: Reduction method ('mean', 'sum', 'none')

        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate focal loss.

        Args:
            inputs: Model predictions
            targets: Ground truth labels

        Returns:
            torch.Tensor: Computed focal loss

        """
        # Convert to binary case for violence detection
        bce_loss = fn.binary_cross_entropy_with_logits(
            inputs, fn.one_hot(targets, num_classes=inputs.size(1)).float(), reduction="none"
        )

        pt = torch.exp(-bce_loss)  # prevents nans when probability 0
        f_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return f_loss.mean()
        elif self.reduction == "sum":
            return f_loss.sum()
        else:
            return f_loss


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    use_amp: bool = True,
    epoch: int = 0,
    log_interval: int = 10,
) -> Dict[str, float]:
    """Train for one epoch.

    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        scaler: GradScaler for mixed precision training
        scheduler: Learning rate scheduler
        use_amp: Whether to use mixed precision
        epoch: Current epoch number
        log_interval: How often to log progress

    Returns:
        Dictionary with training metrics

    """
    model.train()

    running_loss = 0.0
    all_preds = []
    all_targets = []

    start_time = time.time()
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass with mixed precision
        if use_amp:
            with autocast():
                output, _ = model(data)
                loss = criterion(output, target)

            # Backward and optimize with scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard forward pass
            output, _ = model(data)
            loss = criterion(output, target)

            # Backward and optimize
            loss.backward()
            optimizer.step()

        # Update metrics
        running_loss += loss.item()

        # Get predictions
        _, preds = torch.max(output, 1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

        # Update progress bar
        progress_bar.set_postfix({"loss": f"{running_loss / (batch_idx + 1):.4f}"})

        # Step scheduler if batch-wise
        if scheduler is not None and isinstance(
            scheduler, (optim.lr_scheduler.CyclicLR, optim.lr_scheduler.OneCycleLR)
        ):
            scheduler.step()

    # Calculate metrics
    train_loss = running_loss / len(train_loader)
    train_acc = accuracy_score(all_targets, all_preds)
    train_precision = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    train_recall = recall_score(all_targets, all_preds, average="macro", zero_division=0)
    train_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    # Step scheduler if epoch-wise
    if scheduler is not None and not isinstance(
        scheduler, (optim.lr_scheduler.CyclicLR, optim.lr_scheduler.OneCycleLR)
    ):
        scheduler.step()

    # Log metrics
    logger.info(
        f"Train Epoch: {epoch + 1} | "
        f"Loss: {train_loss:.4f} | "
        f"Acc: {train_acc:.4f} | "
        f"Prec: {train_precision:.4f} | "
        f"Rec: {train_recall:.4f} | "
        f"F1: {train_f1:.4f} | "
        f"Time: {time.time() - start_time:.2f}s"
    )

    return {
        "loss": train_loss,
        "accuracy": train_acc,
        "precision": train_precision,
        "recall": train_recall,
        "f1": train_f1,
    }


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True,
    epoch: int = 0,
) -> Dict[str, float]:
    """Validate the model.

    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        use_amp: Whether to use mixed precision
        epoch: Current epoch number

    Returns:
        Dictionary with validation metrics

    """
    model.eval()

    val_loss = 0.0
    all_preds = []
    all_probs = []
    all_targets = []

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc=f"Validation {epoch + 1}")

        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)

            # Forward pass with mixed precision
            if use_amp:
                with autocast():
                    output, _ = model(data)
                    loss = criterion(output, target)
            else:
                output, _ = model(data)
                loss = criterion(output, target)

            # Update metrics
            val_loss += loss.item()

            # Get predictions and probabilities
            probs = torch.softmax(output, dim=1)
            _, preds = torch.max(output, 1)
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    # Calculate metrics
    val_loss /= len(val_loader)
    val_acc = accuracy_score(all_targets, all_preds)
    val_precision = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    val_recall = recall_score(all_targets, all_preds, average="macro", zero_division=0)
    val_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    # ROC AUC score for binary classification
    val_auc = roc_auc_score(all_targets, all_probs) if len(np.unique(all_targets)) > 1 else 0.0

    # Log metrics
    logger.info(
        f"Validation Epoch: {epoch + 1} | "
        f"Loss: {val_loss:.4f} | "
        f"Acc: {val_acc:.4f} | "
        f"Prec: {val_precision:.4f} | "
        f"Rec: {val_recall:.4f} | "
        f"F1: {val_f1:.4f} | "
        f"AUC: {val_auc:.4f}"
    )

    metrics = {
        "loss": val_loss,
        "accuracy": val_acc,
        "precision": val_precision,
        "recall": val_recall,
        "f1": val_f1,
        "auc": val_auc,
    }

    return metrics
