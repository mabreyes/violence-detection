"""Factory functions for creating violence detection models."""

import torch.nn as nn

from .violence_detection import ViolenceDetectionModel


def create_model(
    num_classes: int = 2,
    in_channels: int = 3,
    clip_length: int = 16,
    dropout: float = 0.3,
    device: str = "cuda",
    optimize_for_mobile: bool = False,
) -> nn.Module:
    """Create an instance of the violence detection model.

    Args:
        num_classes: Number of output classes
        in_channels: Number of input channels
        clip_length: Number of frames per clip
        dropout: Dropout rate
        device: Device to run model on ('cuda' or 'cpu')
        optimize_for_mobile: Whether to optimize model for mobile deployment

    Returns:
        Instantiated model

    """
    # Create model with parameters optimized for limited compute
    if optimize_for_mobile:
        # Smaller model for mobile deployment
        model = ViolenceDetectionModel(
            num_classes=num_classes,
            in_channels=in_channels,
            base_filters=16,  # Reduced filters
            clip_length=clip_length,
            lstm_hidden_size=128,  # Smaller LSTM
            transformer_embed_dim=128,  # Smaller transformer
            transformer_num_heads=2,  # Fewer attention heads
            transformer_layers=1,  # Single transformer layer
            dropout=dropout,
            device=device,
        )
    else:
        # Standard model for server deployment
        model = ViolenceDetectionModel(
            num_classes=num_classes,
            in_channels=in_channels,
            base_filters=32,
            clip_length=clip_length,
            lstm_hidden_size=256,
            transformer_embed_dim=256,
            transformer_num_heads=4,
            transformer_layers=2,
            dropout=dropout,
            device=device,
        )

    return model
