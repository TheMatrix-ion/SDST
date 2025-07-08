"""Simplified single-channel transformer reference implementation.

This module provides a minimal transformer model that mirrors the
structure of the SDST transformer while operating on a single input
channel. Each input sample is embedded to a hidden dimension before
being processed by a small transformer block consisting of a
MultiheadAttention layer and a feed-forward network.
"""

import torch
from torch import nn


class SingleChannelTransformer(nn.Module):
    """Transformer layer for single-channel inputs."""

    def __init__(self, d_model: int = 16, ffn_size: int = 64,
                 dropout: float = 0.1, n_heads: int = 1) -> None:
        super().__init__()
        self.embedding = nn.Linear(1, d_model)

        self.att_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.attn_dropout = nn.Dropout(dropout)

        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_size),
            nn.GELU(),
            nn.Linear(ffn_size, d_model),
        )
        self.ffn_dropout = nn.Dropout(dropout)
        self.output = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``[batch, seq_len, 1]``.
        """
        x = self.embedding(x)

        y = self.att_norm(x)
        attn_out, _ = self.attention(y, y, y)
        attn_out = self.attn_dropout(attn_out)
        x = x + attn_out

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y

        return self.output(x)
