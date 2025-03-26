"""Transformer modules for sequence modeling in violence detection."""

from typing import Optional

import torch
import torch.nn as nn

from .layers import PositionalEncoding


class TransformerEncoder(nn.Module):
    """Transformer encoder for temporal modeling.

    optimized for limited compute resources.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        num_layers: int = 2,
        activation: str = "relu",
        pre_norm: bool = True,
    ):
        """Initialize transformer encoder.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            dropout: Dropout rate
            num_layers: Number of transformer layers
            activation: Activation function ('relu' or 'gelu')
            pre_norm: Whether to use pre-layer normalization

        """
        super().__init__()

        self.embed_dim = embed_dim
        self.pre_norm = pre_norm

        # Add positional encoding
        self.pos_encoder = PositionalEncoding(embed_dim)

        # Create transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_layers
        )

        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply transformer encoder to input tensor.

        Args:
            x: Input tensor of shape [batch_size, seq_length, embed_dim]
            mask: Optional mask for padding

        Returns:
            Transformed output of shape [batch_size, seq_length, embed_dim]

        """
        # Add positional encoding
        x = self.pos_encoder(x)

        # Apply pre-normalization if specified
        if self.pre_norm:
            x = self.norm(x)

        # Apply transformer encoder
        output = self.transformer_encoder(x, src_key_padding_mask=mask)

        # Apply post-normalization if not pre-norm
        if not self.pre_norm:
            output = self.norm(output)

        return output
