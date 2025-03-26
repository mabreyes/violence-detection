"""Main training loop for violence detection models."""

import logging
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler

from .training import EarlyStopping, train_epoch, validate

logger = logging.getLogger(__name__)


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    num_epochs: int = 30,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    early_stopping: Optional[EarlyStopping] = None,
    use_amp: bool = True,
    save_dir: str = "checkpoints",
    save_best_only: bool = True,
    onnx_export: bool = False,
) -> Dict[str, List[float]]:
    """Train the model.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        num_epochs: Number of epochs to train for
        scheduler: Learning rate scheduler
        early_stopping: Early stopping callback
        use_amp: Whether to use mixed precision training
        save_dir: Directory to save checkpoints
        save_best_only: Whether to save only the best model
        onnx_export: Whether to export model to ONNX after training

    Returns:
        Dictionary with training history

    """
    # Initialize variables
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "val_auc": [],
    }

    # Create gradient scaler for mixed precision training
    scaler = GradScaler() if use_amp else None

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Best metric value for model saving
    best_val_f1 = 0.0

    # Training loop
    for epoch in range(num_epochs):
        # Train for one epoch
        train_metrics = train_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            scheduler=scheduler,
            use_amp=use_amp,
            epoch=epoch,
        )

        # Validate
        val_metrics = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
            epoch=epoch,
        )

        # Update history
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_auc"].append(val_metrics["auc"])

        # Save model
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_metrics["f1"],
                "val_acc": val_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
            }

            if save_best_only:
                # Save only the best model
                torch.save(checkpoint, os.path.join(save_dir, "best_model.pth"))
                logger.info(f"Saved best model with F1: {best_val_f1:.4f}")
            else:
                # Save all models
                torch.save(checkpoint, os.path.join(save_dir, f"model_epoch_{epoch + 1}.pth"))
                logger.info(f"Saved model from epoch {epoch + 1} with F1: {val_metrics['f1']:.4f}")

        # Early stopping
        if early_stopping is not None:
            if early_stopping(val_metrics["f1"]):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

    # Export to ONNX if requested
    if onnx_export:
        try:
            logger.info("Exporting model to ONNX format...")
            dummy_input = torch.randn(1, 16, 3, 112, 112).to(device)  # Adjust shape as needed
            torch.onnx.export(
                model,
                (dummy_input,),
                os.path.join(save_dir, "model.onnx"),
                export_params=True,
                opset_version=12,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            )
            logger.info(f"Model exported to {os.path.join(save_dir, 'model.onnx')}")
        except Exception as e:
            logger.error(f"Failed to export model to ONNX: {e}")

    return history
