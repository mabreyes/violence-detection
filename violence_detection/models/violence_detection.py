"""Main violence detection model implementation combining 4D CNN, LSTM, and Transformer."""

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from .layers import CNN4DBlock, LSTMWithAttention, SpatioTemporalAttention
from .transformer import TransformerEncoder


class ViolenceDetectionModel(nn.Module):
    """Violence Detection model optimized for Philippine context.

    and limited compute resources using 4D CNN + LSTM + Transformer architecture.
    """

    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 3,
        base_filters: int = 32,
        clip_length: int = 16,
        lstm_hidden_size: int = 256,
        transformer_embed_dim: int = 256,
        transformer_num_heads: int = 4,
        transformer_layers: int = 2,
        dropout: float = 0.3,
        use_attention: bool = True,
        device: str = "cuda",
    ):
        """Initialize violence detection model.

        Args:
            num_classes: Number of output classes
            in_channels: Number of input channels
            base_filters: Base number of filters for CNN
            clip_length: Number of frames per clip
            lstm_hidden_size: Hidden size for LSTM
            transformer_embed_dim: Embedding dimension for transformer
            transformer_num_heads: Number of attention heads for transformer
            transformer_layers: Number of transformer layers
            dropout: Dropout rate
            use_attention: Whether to use attention mechanism
            device: Device to use ('cuda' or 'cpu')

        """
        super().__init__()

        self.in_channels = in_channels
        self.clip_length = clip_length
        self.use_attention = use_attention
        self.device = device

        # 4D CNN Encoder for spatio-temporal feature extraction
        self.encoder = nn.Sequential(
            # Initial convolution
            nn.Conv3d(
                in_channels, base_filters, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False
            ),
            nn.BatchNorm3d(base_filters),
            nn.ReLU(inplace=True),
            # 4D CNN Blocks
            CNN4DBlock(base_filters, base_filters * 2, kernel_size=(3, 3, 3), stride=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            CNN4DBlock(base_filters * 2, base_filters * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            CNN4DBlock(base_filters * 4, base_filters * 8, kernel_size=(3, 3, 3), stride=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            # More efficient final block
            CNN4DBlock(
                base_filters * 8,
                base_filters * 8,
                kernel_size=(3, 3, 3),
                stride=(1, 1, 1),
                temporal_first=False,
            ),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            # Spatio-temporal attention
            SpatioTemporalAttention(
                embed_dim=base_filters * 8,
                num_heads=4,
                dropout=dropout,
                use_linear_projection=False,
            ),
        )

        # Calculate output size after CNN
        self._calc_cnn_output_size()

        # LSTM with attention for temporal modeling
        self.lstm = LSTMWithAttention(
            input_size=self.cnn_flat_size,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            bidirectional=True,
            dropout=dropout,
        )

        # Transformer for contextual understanding
        self.transformer = TransformerEncoder(
            embed_dim=lstm_hidden_size * 2,  # Bidirectional LSTM
            num_heads=transformer_num_heads,
            ff_dim=transformer_embed_dim * 2,
            dropout=dropout,
            num_layers=transformer_layers,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, lstm_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_size, num_classes),
        )

        # Initialize weights
        self._initialize_weights()

    def _calc_cnn_output_size(self) -> None:
        """Calculate the output size of the CNN encoder."""
        # Initialize dummy input: [B, C, T, H, W]
        dummy_input = torch.zeros(1, self.in_channels, self.clip_length, 112, 112)

        # Forward through encoder
        with torch.no_grad():
            output = self.encoder(dummy_input)

        # Store output shape
        self.cnn_output_shape = output.shape
        self.cnn_flat_size = output.shape[1] * output.shape[3] * output.shape[4]

    def _initialize_weights(self) -> None:
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass of the model.

        Args:
            x: Input video clips of shape [batch_size, time, channels, height, width]

        Returns:
            logits: Classification logits
            extras: Dictionary with additional outputs (attention maps, etc.)

        """
        batch_size = x.shape[0]

        # Reshape input to [B, C, T, H, W] for 3D convolution
        x = x.permute(0, 2, 1, 3, 4)

        # Extract spatio-temporal features with 4D CNN
        features = self.encoder(x)  # [B, C', T', H', W']

        # Reshape for temporal modeling: [B, T', C'*H'*W']
        t_dim = features.shape[2]
        features = features.permute(0, 2, 1, 3, 4).contiguous()
        features = features.view(batch_size, t_dim, -1)

        # Apply LSTM with attention
        context, attention_weights = self.lstm(features)

        # Reshape for transformer: [B, 1, hidden_size]
        context = context.unsqueeze(1)

        # Apply transformer encoder
        context = self.transformer(context)

        # Classification
        logits = self.classifier(context.squeeze(1))

        # Collect additional outputs for analysis/visualization
        extras = {"attention_weights": attention_weights, "features": features}

        return logits, extras
