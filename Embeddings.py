import torch.nn as nn
import torch

class TokenEmbeddings(nn.Module):

    def __init__(self, vocab_size : int, emb_size : int):
        super().__init__() # вызов конструктора родительского класса
        self.embeddings = nn.Embedding(vocab_size, emb_size) # создание матрицы ембеддингов


    def forward(self, x: torch.Tensor) :
        """Прямой проход через слой эмбеддингов.

        Args:
            x (torch.Tensor): Входной тензор с индексами токенов.

        Returns:
            
        """
        