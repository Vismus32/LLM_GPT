import torch.nn as nn
import torch


class FeedForward(nn.Module):

    def __init__(self, emb_size : int, dropout : float = 0.1):

        super().__init__()

        self.linear1 = nn.Linear(emb_size, emb_size * 4)
        self.relu = self.relu = nn.ReLU()
        self.linear2 = nn.Linear(emb_size * 4, emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Выполняет прямой проход через Feed Forward Network.

        Последовательно пропускает входной тензор через:
            1. Первый линейный слой
            2. Функцию активации ReLU
            3. Второй линейный слой, чтобы вернуть размерность
            4. Слой Dropout

        Args:
            x (torch.Tensor): Входной тензор типа float размером
                (batch_size, seq_len, emb_size).

        Returns:
            torch.Tensor: Итоговый тензор размером
                (batch_size, seq_len, emb_size).
        """

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.dropout(x)

        return x

    

