import torch.nn as nn
import torch

from Transformer.HeadAttention import HeadAttention


class MultiHeadAttention(nn.Module):
    """
        Args:
            num_heads (int): Количество голов self-attention
            emb_size (int): Размерность входных эмбеддингов
            head_size (int): Размерность Key, Query и Value внутри каждой головы
            max_seq_len (int): Максимальная длина входной последовательности
            dropout (float): Вероятность обнуления элементовв слое Dropout
    """

    def __init__(
        self,
        num_heads: int,
        emb_size: int,
        head_size: int,
        max_seq_len: int,
        dropout: float = 0.1):

        super().__init__()

        # Создаём HeadAttention такое количество, которому равно num_heads
        self.heads = nn.ModuleList([HeadAttention(emb_size=emb_size, head_size=head_size, max_seq_len=max_seq_len) for _ in range(num_heads)]) 

        # Создал линейный слой
        self.projection = nn.Linear(head_size * num_heads, emb_size )

        # Слой Dropout
        self.dropout = nn.Dropout(dropout)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Выполняет прямой проход через все головы механизмa внимания

        Args:
            x (torch.Tensor): Входной тензор типа float размером
                (batch_size, seq_len, emb_size).

        Returns:
            torch.Tensor: Выходной тензор размером
                (batch_size, seq_len, emb_size).
        """

        # Пропускаем x через все головы
        head_outputs = [head(x) for head in self.heads]

        # Объединяем выходы всех голов по последнему измерению
        concatenated = torch.cat(head_outputs, dim=-1) # dim=-1 - склеиваем по последней размерности

        # Пропускаем через линейный слой
        projected = self.projection(concatenated)

        # Применяем Dropout
        output = self.dropout(projected)

        return output

