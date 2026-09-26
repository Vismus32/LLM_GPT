import torch.nn as nn
import torch

class HeadAttention(nn.Module):

    def __init__(self, emb_size : int, head_size : int, max_seq_len : int):
        super().__init__()

        self.Wk = nn.Linear(emb_size, head_size)
        self.Wq = nn.Linear(emb_size, head_size)
        self.Wv = nn.Linear(emb_size, head_size)

        self.mask = torch.tril(torch.ones(max_seq_len, max_seq_len))

        self.head_size = head_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Выполняет прямой проход через одну голову self-attention.

        Последовательно: 

            1. Получает Key, Query и Value с помощью линейных слоёв. 
            2. Вычисляет матрицу внимания как Q @ K^T и масштабирует её на sqrt(head_size). 
            3. Накладывает маску, скрывая будущие позиции. 
            4. Применяет Softmax к матрице внимания. 
            5. Умножает матрицу внимания на Value и возвращает результат. 

        Args:
             x (torch.Tensor): Входной тензор размером (batch_size, seq_len, emb_size). 

        Returns: 
             torch.Tensor: Выходной тензор размером (batch_size, seq_len, head_size).
        """

        # Получаем Key, Query и Value
        k = self.Wk(x)
        q = self.Wq(x)
        v = self.Wv(x)

        # Вычисляем матрицу внимания, умножаем Quary на транспонированную Key
        attention = q @ k.transpose(-2, -1)

        # Делим на sqrt(head_size)
        attention = attention / torch.sqrt(torch.tensor(self.head_size, dtype=torch.float32))


        # Делаем маску размера seq_len × seq_len
        seq_len = x.shape[1]
        mask = self.mask[:seq_len, :seq_len]

        # Скрываем будущие позиции с помощью этой маски
        attention = attention.masked_fill(mask == 0, float("-inf"))

        #Софтмакс построчно
        attention = torch.softmax(attention, dim=-1)

        # Умножаем матрицу внимания на Value
        output = attention @ v

        return output