import torch.nn as nn
import torch

from Transformer.MHAttention import MultiHeadAttention
from Transformer.FeedForward import FeedForward

class Decoder(nn.Module):

    def __init__(self, num_heads :int, emb_size :int, head_size :int, max_seq_len: int, dropout: float = 0.1):

        super().__init__()

        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            emb_size=emb_size,
            head_size=head_size,
            max_seq_len=max_seq_len,
            dropout=dropout
        )

        self.feed_forward = FeedForward(
            emb_size=emb_size,
            dropout=dropout
        )

        self.layer_norm1 = nn.LayerNorm(emb_size)
        self.layer_norm2 = nn.LayerNorm(emb_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Выполняет прямой проход через Transformer

        Последовательно:
            1. Пропускает x через MultiHeadAttention.
            2. Складывает результат с исходным x.
            3. Применяет первый LayerNorm.
            4. Пропускает получившийся тензор через FeedForward.
            5. Складывает результат с входом в FeedForward.
            6. Применяет второй LayerNorm.

        Args:
            x (torch.Tensor): Входной тензор типа float размером
                (batch_size, seq_len, emb_size).

        Returns:
            torch.Tensor: Итоговый тензор размером
                (batch_size, seq_len, emb_size).
        """

        attention_output = self.attention(x)

        # Residual connection
        x = x + attention_output

        x = self.layer_norm1(x)
        feed_forward_output = self.feed_forward(x)

        # Residual connection
        x = x + feed_forward_output

        x = self.layer_norm2(x)

        return x