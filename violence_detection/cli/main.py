"""Main CLI entry point for violence detection."""

import argparse
import logging
from typing import Optional

from .commands import evaluate_command, infer_command, preprocess_command, train_command


def setup_logging() -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("violence_detection.log")],
    )


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for the CLI.

    Returns:
        Argument parser

    """
    parser = argparse.ArgumentParser(
        description="Violence Detection in Philippine Context with Limited Compute Resources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(
        title="commands", description="valid commands", help="additional help", dest="command"
    )

    # Preprocess command
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess video dataset")
    preprocess_parser.add_argument(
        "--video_dir", type=str, required=True, help="Directory containing video files"
    )
    preprocess_parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save processed data"
    )
    preprocess_parser.add_argument("--annotations", type=str, help="Path to annotations CSV file")
    preprocess_parser.add_argument(
        "--frame_rate", type=int, default=5, help="Number of frames per second to extract"
    )
    preprocess_parser.add_argument(
        "--max_frames", type=int, default=300, help="Maximum frames to extract per video"
    )
    preprocess_parser.add_argument(
        "--resize_width", type=int, default=224, help="Width to resize frames"
    )
    preprocess_parser.add_argument(
        "--resize_height", type=int, default=224, help="Height to resize frames"
    )
    preprocess_parser.add_argument(
        "--test_size", type=float, default=0.2, help="Fraction of data for validation"
    )

    # Train command
    train_parser = subparsers.add_parser("train", help="Train violence detection model")
    train_parser.add_argument(
        "--video_path", type=str, required=True, help="Path to video directory"
    )
    train_parser.add_argument(
        "--train_labels", type=str, required=True, help="Path to training labels CSV"
    )
    train_parser.add_argument(
        "--val_labels", type=str, required=True, help="Path to validation labels CSV"
    )
    train_parser.add_argument(
        "--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints"
    )
    train_parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save logs")
    train_parser.add_argument(
        "--clip_length", type=int, default=16, help="Number of frames per clip"
    )
    train_parser.add_argument(
        "--frame_stride", type=int, default=2, help="Stride between consecutive frames"
    )
    train_parser.add_argument("--spatial_size", type=int, default=112, help="Frame resolution")
    train_parser.add_argument(
        "--mobile", action="store_true", help="Optimize model for mobile deployment"
    )
    train_parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    train_parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )
    train_parser.add_argument(
        "--num_epochs", type=int, default=30, help="Number of epochs to train for"
    )
    train_parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    train_parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    train_parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    train_parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    train_parser.add_argument("--amp", action="store_true", help="Use mixed precision training")
    train_parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    train_parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["step", "cosine", "plateau", "none"],
        help="Learning rate scheduler",
    )
    train_parser.add_argument("--focal_loss", action="store_true", help="Use focal loss")
    train_parser.add_argument(
        "--export_onnx", action="store_true", help="Export model to ONNX after training"
    )

    # Inference command
    infer_parser = subparsers.add_parser("infer", help="Run inference on videos")
    infer_parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the trained model"
    )
    infer_parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Path to a video file or directory containing videos",
    )
    infer_parser.add_argument(
        "--output_dir", type=str, default="results", help="Directory to save results"
    )
    infer_parser.add_argument(
        "--device", type=str, default="cuda", help="Device to run inference on"
    )
    infer_parser.add_argument(
        "--clip_length", type=int, default=16, help="Number of frames per clip"
    )
    infer_parser.add_argument(
        "--frame_stride", type=int, default=2, help="Stride between consecutive frames"
    )
    infer_parser.add_argument(
        "--threshold", type=float, default=0.5, help="Decision threshold for violence detection"
    )
    infer_parser.add_argument("--mobile", action="store_true", help="Use mobile-optimized model")
    infer_parser.add_argument(
        "--visualize", action="store_true", help="Save visualization of prediction results"
    )
    infer_parser.add_argument(
        "--dense_sampling", action="store_true", help="Use dense sampling strategy"
    )

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model on validation set")
    eval_parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the trained model"
    )
    eval_parser.add_argument(
        "--video_path", type=str, required=True, help="Directory containing video files"
    )
    eval_parser.add_argument(
        "--val_labels", type=str, required=True, help="Path to validation labels CSV"
    )
    eval_parser.add_argument(
        "--output_dir", type=str, default="evaluation", help="Directory to save evaluation results"
    )
    eval_parser.add_argument(
        "--device", type=str, default="cuda", help="Device to run evaluation on"
    )
    eval_parser.add_argument(
        "--clip_length", type=int, default=16, help="Number of frames per clip"
    )
    eval_parser.add_argument(
        "--frame_stride", type=int, default=2, help="Stride between consecutive frames"
    )
    eval_parser.add_argument(
        "--threshold", type=float, default=0.5, help="Decision threshold for violence detection"
    )
    eval_parser.add_argument("--mobile", action="store_true", help="Use mobile-optimized model")

    return parser


def main(args: Optional[argparse.Namespace] = None) -> None:
    """Execute the main entry point for the CLI.

    Args:
        args: Command line arguments (if None, parse from sys.argv)

    """
    # Set up logging
    setup_logging()

    # Parse arguments
    parser = create_parser()
    args = args or parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # Execute command
    if args.command == "preprocess":
        preprocess_command(args)
    elif args.command == "train":
        train_command(args)
    elif args.command == "infer":
        infer_command(args)
    elif args.command == "evaluate":
        evaluate_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
