import torch.nn as nn
import torch

class TokenEmbeddings(nn.Module):

    def __init__(self, vocab_size : int, emb_size : int):
        super().__init__() # вызов конструктора родительского класса
        self.embeddings = nn.Embedding(vocab_size, emb_size) # создание матрицы ембеддингов


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход через слой эмбеддингов.

        Args:
            x (torch.Tensor): Входной тензор с индексами токенов.
            Каждый элемент является индексом токена в словаре.

        Returns:
            torch.Tensor: Тензор эмбеддингов размером
            (batch_size, seq_len, emb_size), содержащий вектор эмбеддинга
            для каждого токена в том же порядке, в котором токены
            переданы во входном тензоре.

        """
        embeddings = self.embeddings(x)

        return embeddings

class PositionalEmbeddings(nn.Module):

    def __init__(self,max_seq_len : int, emb_size : int):
        super().__init__()
        self.pos_embeddings = nn.Embedding(max_seq_len , emb_size)

    def forward(self, seq_len: int) -> torch.Tensor:
        """Получение позиционных эмбеддингов

        Args:
            seq_len (int): Длина последовательности. Определяет количество
                позиций, для которых необходимо получить эмбеддинги.

        Returns:
            torch.Tensor: Тензор позиционных эмбеддингов размером
                (seq_len, emb_size), содержащий эмбеддинги позиций
                от 0 до seq_len - 1.
        """
        positions = torch.arange(seq_len)
        embeddings = self.pos_embeddings(positions)

        return embeddings
