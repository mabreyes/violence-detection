"""Basic neural network building blocks for violence detection models.

This module contains reusable layer components like attention mechanisms,
positional encoding, and custom convolutional blocks.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as fn


class PositionalEncoding(nn.Module):
    """Positional encoding for the transformer model.

    Adds positional information to input embeddings, allowing the model
    to understand the relative or absolute position of tokens in the sequence.
    """

    def __init__(self, d_model: int, max_seq_length: int = 200):
        """Initialize positional encoding.

        Args:
            d_model: Dimension of the model
            max_seq_length: Maximum sequence length

        """
        super().__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # Register as buffer (not a parameter but should be saved)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor.

        Args:
            x: Tensor of shape [batch_size, seq_length, embedding_dim]

        Returns:
            Output tensor with added positional encoding

        """
        return x + self.pe[:, : x.size(1)]


class SpatioTemporalAttention(nn.Module):
    """Self-attention module for spatio-temporal features.

    Optimized for limited compute resources.

    Can process either linear inputs or convolutional feature maps.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_linear_projection: bool = True,
    ):
        """Initialize spatio-temporal attention module.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            use_linear_projection: Whether to use linear projection (more parameter efficient)
                                  or convolutional projection (more compute efficient)

        """
        super().__init__()

        self.use_linear_projection = use_linear_projection
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        if use_linear_projection:
            # Use linear projections for Q, K, V (more parameter efficient)
            self.qkv = nn.Linear(embed_dim, embed_dim * 3)
            self.proj = nn.Linear(embed_dim, embed_dim)
        else:
            # Use conv projections for Q, K, V (more compute efficient)
            self.qkv_conv = nn.Conv3d(embed_dim, embed_dim * 3, kernel_size=1, bias=False)
            self.proj_conv = nn.Conv3d(embed_dim, embed_dim, kernel_size=1)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        # Initialize weights
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize weights with Xavier uniform distribution."""
        if self.use_linear_projection:
            nn.init.xavier_uniform_(self.qkv.weight)
            nn.init.xavier_uniform_(self.proj.weight)
            nn.init.constant_(self.qkv.bias, 0)
            nn.init.constant_(self.proj.bias, 0)
        else:
            nn.init.xavier_uniform_(self.qkv_conv.weight)
            nn.init.xavier_uniform_(self.proj_conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial temporal attention to input tensor.

        Args:
            x: Input tensor of shape [B, C, T, H, W] for convolutional feature maps
               or [B, L, C] for linear inputs

        Returns:
            torch.Tensor: Output tensor with same shape as input

        """
        batch_size = x.shape[0]

        if self.use_linear_projection:
            # Linear projection
            # Reshape: [B, C, T, H, W] -> [B, THW, C]
            if len(x.shape) == 5:
                c, t, h, w = x.shape[1], x.shape[2], x.shape[3], x.shape[4]
                x = x.permute(0, 2, 3, 4, 1).reshape(batch_size, t * h * w, c)

            # Calculate qkv: [B, N, 3*dim]
            qkv = self.qkv(x)

            # Reshape: [B, N, 3*C] -> [3, B, num_heads, N, C // num_heads]
            n = qkv.shape[1]
            qkv = qkv.reshape(batch_size, n, 3, self.num_heads, self.embed_dim // self.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)

            # Separate q, k, v
            q, k, v = qkv[0], qkv[1], qkv[2]

            # Scaled dot-product attention
            attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            attn = fn.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)

            # Apply attention weights to values
            x = (attn @ v).transpose(1, 2).reshape(batch_size, n, self.embed_dim)

            # Final projection
            x = self.proj(x)
            x = self.proj_dropout(x)

            # Reshape back to original format if needed
            if len(x.shape) == 3 and c != self.embed_dim:
                x = x.reshape(batch_size, t, h, w, self.embed_dim).permute(0, 4, 1, 2, 3)

        else:
            # Convolutional projection
            # Calculate qkv: [B, 3*C, T, H, W]
            qkv = self.qkv_conv(x)

            # Reshape: [B, 3*C, T, H, W] -> [3, B, num_heads, T*H*W, C // num_heads]
            c, t, h, w = x.shape[1], x.shape[2], x.shape[3], x.shape[4]
            qkv = qkv.reshape(batch_size, 3, self.num_heads, c // self.num_heads, t * h * w)
            qkv = qkv.permute(1, 0, 2, 4, 3)

            # Separate q, k, v
            q, k, v = qkv[0], qkv[1], qkv[2]

            # Scaled dot-product attention
            attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            attn = fn.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)

            # Apply attention weights to values
            x = (attn @ v).transpose(1, 2).reshape(batch_size, c, t, h, w)

            # Final projection
            x = self.proj_conv(x)
            x = self.proj_dropout(x)

        return x


class CNN4DBlock(nn.Module):
    """4D Convolutional Block (3D spatial + 1D temporal).

    Optimized for memory efficiency.

    Implements factorized convolutions to reduce computational cost
    while maintaining performance.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int, int] = (3, 3, 3),
        stride: Tuple[int, int, int] = (1, 1, 1),
        padding: Tuple[int, int, int] = (1, 1, 1),
        bias: bool = False,
        use_bn: bool = True,
        use_activation: bool = True,
        temporal_first: bool = True,
    ):
        """Initialize 4D CNN block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Kernel size as (temporal, height, width)
            stride: Stride as (temporal, height, width)
            padding: Padding as (temporal, height, width)
            bias: Whether to use bias in convolutions
            use_bn: Whether to use batch normalization
            use_activation: Whether to use activation function
            temporal_first: Whether to apply temporal convolution before spatial

        """
        super().__init__()

        self.temporal_first = temporal_first
        self.use_bn = use_bn
        self.use_activation = use_activation

        if temporal_first:
            # Temporal convolution followed by spatial
            self.temporal_conv = nn.Conv3d(
                in_channels,
                in_channels * 2,
                kernel_size=(kernel_size[0], 1, 1),
                stride=(stride[0], 1, 1),
                padding=(padding[0], 0, 0),
                bias=bias,
                groups=in_channels,  # Depthwise conv for efficiency
            )

            if use_bn:
                self.temporal_bn = nn.BatchNorm3d(in_channels * 2)

            self.spatial_conv = nn.Conv3d(
                in_channels * 2,
                out_channels,
                kernel_size=(1, kernel_size[1], kernel_size[2]),
                stride=(1, stride[1], stride[2]),
                padding=(0, padding[1], padding[2]),
                bias=bias,
            )
        else:
            # Spatial convolution followed by temporal
            self.spatial_conv = nn.Conv3d(
                in_channels,
                in_channels * 2,
                kernel_size=(1, kernel_size[1], kernel_size[2]),
                stride=(1, stride[1], stride[2]),
                padding=(0, padding[1], padding[2]),
                bias=bias,
            )

            if use_bn:
                self.spatial_bn = nn.BatchNorm3d(in_channels * 2)

            self.temporal_conv = nn.Conv3d(
                in_channels * 2,
                out_channels,
                kernel_size=(kernel_size[0], 1, 1),
                stride=(stride[0], 1, 1),
                padding=(padding[0], 0, 0),
                bias=bias,
            )

        if use_bn:
            self.bn = nn.BatchNorm3d(out_channels)

        if use_activation:
            # Using ReLU instead of more expensive activations
            self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the 4D CNN block.

        Args:
            x: Input tensor of shape [B, C, T, H, W]

        Returns:
            Output tensor with convolutions applied

        """
        if self.temporal_first:
            x = self.temporal_conv(x)
            if self.use_bn:
                x = self.temporal_bn(x)
            x = fn.relu(x, inplace=True)
            x = self.spatial_conv(x)
        else:
            x = self.spatial_conv(x)
            if self.use_bn:
                x = self.spatial_bn(x)
            x = fn.relu(x, inplace=True)
            x = self.temporal_conv(x)

        if self.use_bn:
            x = self.bn(x)

        if self.use_activation:
            x = self.activation(x)

        return x


class LSTMWithAttention(nn.Module):
    """LSTM module with attention mechanism for temporal modeling.

    Designed for efficiency on limited compute resources.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.1,
        attention_size: Optional[int] = None,
    ):
        """Initialize LSTM with attention.

        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            num_layers: Number of LSTM layers
            bidirectional: Whether to use bidirectional LSTM
            dropout: Dropout rate
            attention_size: Size of attention layer (defaults to hidden_size * directions)

        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.directions = 2 if bidirectional else 1

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Attention mechanism
        if attention_size is None:
            attention_size = hidden_size * self.directions

        self.attention = nn.Sequential(
            nn.Linear(hidden_size * self.directions, attention_size),
            nn.Tanh(),
            nn.Linear(attention_size, 1),
        )

    def forward(
        self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply LSTM with attention to input tensor.

        Args:
            x: Input tensor of shape [batch_size, seq_length, input_size]
            hidden: Initial hidden state tuple (h0, c0)

        Returns:
            context: Context vector with attention applied
            attention_weights: Attention weights for visualization

        """
        # LSTM forward pass
        outputs, _ = self.lstm(x, hidden)  # [batch_size, seq_length, hidden_size * directions]

        # Calculate attention scores
        attention_scores = self.attention(outputs).squeeze(-1)  # [batch_size, seq_length]
        attention_weights = fn.softmax(attention_scores, dim=1).unsqueeze(
            1
        )  # [batch_size, 1, seq_length]

        # Apply attention weights to sum up the context
        context = torch.bmm(attention_weights, outputs).squeeze(
            1
        )  # [batch_size, hidden_size * directions]

        return context, attention_weights
