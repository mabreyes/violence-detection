"""Command handlers for violence detection CLI."""

import glob
import json
import logging
import os
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import torch

from violence_detection.data import create_dataloaders
from violence_detection.inference import ViolenceDetector
from violence_detection.models import create_model
from violence_detection.utils import EarlyStopping, FocalLoss, train_model

logger = logging.getLogger(__name__)


def preprocess_command(args: Any) -> None:
    """Run data preprocessing.

    Args:
        args: Command-line arguments

    """
    from .preprocessing import analyze_dataset, process_dataset

    logger.info("Starting data preprocessing...")

    # Process dataset
    train_df, val_df = process_dataset(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        annotations=args.annotations,
        test_size=args.test_size,
        frame_rate=args.frame_rate,
        max_frames_per_video=args.max_frames,
        resize=(args.resize_width, args.resize_height),
    )

    # Analyze dataset
    analyze_dataset(train_df, val_df, args.output_dir)

    logger.info(f"Preprocessing complete! Data saved to {args.output_dir}")


def train_command(args: Any) -> None:
    """Run model training.

    Args:
        args: Command-line arguments

    """
    logger.info("Starting model training...")

    # Set up device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info(f"Using device: {device}")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    if use_cuda:
        torch.cuda.manual_seed_all(42)

    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        video_path=args.video_path,
        train_label_path=args.train_labels,
        val_label_path=args.val_labels,
        batch_size=args.batch_size,
        clip_length=args.clip_length,
        frame_stride=args.frame_stride,
        num_workers=args.num_workers,
        spatial_size=(args.spatial_size, args.spatial_size),
        pin_memory=use_cuda,
    )

    logger.info(
        f"Training with {len(train_loader.dataset)} samples, "
        f"validating with {len(val_loader.dataset)} samples"
    )

    # Create model
    model = create_model(
        num_classes=2,  # Binary classification for violence detection
        in_channels=3,  # RGB videos
        clip_length=args.clip_length,
        dropout=args.dropout,
        device=str(device),
        optimize_for_mobile=args.mobile,
    )
    model = model.to(device)

    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model has {total_params:,} parameters, {trainable_params:,} trainable")

    # Define loss function
    if args.focal_loss:
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        logger.info("Using Focal Loss")
    else:
        criterion = torch.nn.CrossEntropyLoss()
        logger.info("Using Cross Entropy Loss")

    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Define learning rate scheduler
    if args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.1, patience=5, verbose=True
        )
    else:
        scheduler = None

    if scheduler is not None:
        logger.info(f"Using {args.scheduler} learning rate scheduler")

    # Define early stopping
    early_stopping = EarlyStopping(patience=args.patience, mode="max")
    logger.info(f"Using early stopping with patience {args.patience}")

    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Train model
    logger.info("Starting training...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=args.num_epochs,
        scheduler=scheduler,
        early_stopping=early_stopping,
        use_amp=args.amp,
        save_dir=args.save_dir,
        save_best_only=True,
        onnx_export=args.export_onnx,
    )

    logger.info("Training complete!")

    # Save training history
    history_path = os.path.join(args.log_dir, "training_history.json")
    with open(history_path, "w") as f:
        # Convert numpy values to Python native types
        for k, v in history.items():
            history[k] = [float(val) for val in v]
        json.dump(history, f)

    logger.info(f"Training history saved to {history_path}")


def infer_command(args: Any) -> None:
    """Run inference on videos.

    Args:
        args: Command-line arguments

    """
    logger.info("Starting violence detection inference...")

    # Create detector
    detector = ViolenceDetector(
        model_path=args.model_path,
        device=args.device,
        clip_length=args.clip_length,
        frame_stride=args.frame_stride,
        threshold=args.threshold,
        mobile_optimized=args.mobile,
    )

    # Find videos to process
    if os.path.isdir(args.video_path):
        video_files = []
        for ext in ["mp4", "avi", "mov", "mkv"]:
            video_files.extend(glob.glob(os.path.join(args.video_path, f"*.{ext}")))

        logger.info(f"Found {len(video_files)} videos in {args.video_path}")
    else:
        video_files = [args.video_path]

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process videos
    results = []

    for video_path in video_files:
        logger.info(f"Processing {video_path}...")

        try:
            result = detector.predict_video(
                video_path=video_path,
                save_visualization=args.visualize,
                output_dir=args.output_dir,
                uniform_sampling=not args.dense_sampling,
            )

            # Add video name to result
            result["video"] = os.path.basename(video_path)
            results.append(result)

            logger.info(f"Result: {result['class']} (score: {result['score']:.4f})")

        except Exception as e:
            logger.error(f"Error processing {video_path}: {str(e)}")

    # Save overall results
    output = {
        "results": [
            {"video": r["video"], "score": r["score"], "class": r["class"]} for r in results
        ],
        "metadata": {
            "threshold": args.threshold,
            "model": args.model_path,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
    }

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(output, f, indent=4)

    logger.info(f"Inference complete! Results saved to {args.output_dir}")


def evaluate_command(args: Any) -> None:
    """Evaluate model on a validation set.

    Args:
        args: Command-line arguments

    """
    logger.info("Starting model evaluation...")

    # Load validation dataset
    val_df = pd.read_csv(args.val_labels)
    logger.info(f"Loaded {len(val_df)} validation samples")

    # Create detector
    detector = ViolenceDetector(
        model_path=args.model_path,
        device=args.device,
        clip_length=args.clip_length,
        frame_stride=args.frame_stride,
        threshold=args.threshold,
        mobile_optimized=args.mobile,
    )

    # Find video files
    if os.path.isdir(args.video_path):
        video_files = []
        for ext in ["mp4", "avi", "mov", "mkv"]:
            video_files.extend(glob.glob(os.path.join(args.video_path, f"*.{ext}")))
        logger.info(f"Found {len(video_files)} videos in {args.video_path}")
    else:
        video_files = [args.video_path]

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process validation videos
    all_scores = []
    all_preds = []
    all_labels = []

    for _, row in val_df.iterrows():
        video_path = os.path.join(args.video_path, row["video_id"])

        if not os.path.exists(video_path):
            logger.warning(f"Video not found: {video_path}")
            continue

        try:
            # Run inference
            result = detector.predict_video(
                video_path=video_path, save_visualization=False, uniform_sampling=True
            )

            # Store results
            score = result["score"]
            pred = 1 if score >= args.threshold else 0
            label = row["label"]

            all_scores.append(score)
            all_preds.append(pred)
            all_labels.append(label)

            logger.info(
                f"Processed {row['video_id']}: score={score:.4f}, pred={pred}, true={label}"
            )

        except Exception as e:
            logger.error(f"Error processing {video_path}: {str(e)}")

    # Calculate metrics
    if len(all_preds) == 0:
        logger.error("No valid predictions made. Check video paths and data.")
        return

    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    auc = roc_auc_score(all_labels, all_scores) if len(np.unique(all_labels)) > 1 else 0.0
    cm = confusion_matrix(all_labels, all_preds)

    # Print metrics
    logger.info("Evaluation Results:")
    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"Precision: {prec:.4f}")
    logger.info(f"Recall: {rec:.4f}")
    logger.info(f"F1-score: {f1:.4f}")
    logger.info(f"AUC-ROC: {auc:.4f}")

    # Create visualizations
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        # ROC curve
        from sklearn.metrics import roc_curve

        fpr, tpr, _ = roc_curve(all_labels, all_scores)

        # Create confusion matrix visualization
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Non-violent", "Violent"],
            yticklabels=["Non-violent", "Violent"],
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.savefig(os.path.join(args.output_dir, "confusion_matrix.png"))
        plt.close()

        # Create ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve (AUC = {auc:.4f})")
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(args.output_dir, "roc_curve.png"))
        plt.close()

        # Score distribution
        plt.figure(figsize=(10, 6))
        scores_df = pd.DataFrame(
            {
                "score": all_scores,
                "label": [
                    "Violent" if label_val == 1 else "Non-violent" for label_val in all_labels
                ],
            }
        )
        sns.histplot(data=scores_df, x="score", hue="label", bins=30, alpha=0.6)
        plt.axvline(
            x=args.threshold, color="red", linestyle="--", label=f"Threshold ({args.threshold})"
        )
        plt.xlabel("Violence Score")
        plt.ylabel("Count")
        plt.title("Score Distribution")
        plt.legend()
        plt.savefig(os.path.join(args.output_dir, "score_distribution.png"))
        plt.close()

    except ImportError as e:
        logger.error(f"Error creating visualizations: {str(e)}")

    # Save evaluation results
    output_dict = {
        "data": [
            {
                "video_id": row["video_id"],
                "label": row["label"],
                "pred": all_preds[i],
                "score": float(all_scores[i]),
            }
            for i, (_, row) in enumerate(val_df.iterrows())
            if row["video_id"] in [os.path.basename(p) for p in video_files]
        ],
        "config": {
            "model_path": args.model_path,
            "threshold": args.threshold,
            "clip_length": args.clip_length,
            "frame_stride": args.frame_stride,
        },
    }

    with open(os.path.join(args.output_dir, "evaluation_results.json"), "w") as f:
        json.dump(output_dict, f, indent=4)

    logger.info(f"Evaluation complete! Results saved to {args.output_dir}")
