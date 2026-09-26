import torch.nn as nn
import torch

from Embeddings_process.Embeddings import PositionalEmbeddings, TokenEmbeddings
from Transformer import Decoder


class GPT(nn.Module):

    def __init__(self,
                vocab_size : int,
                max_seq_len : int,
                emb_size : int,
                num_heads : int,
                head_size : int,
                num_layers : int, 
                dropout : float = 0.1,
                device : str = "cpu"
                ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device

        self.token_embeddings = TokenEmbeddings(vocab_size = vocab_size, emb_size = emb_size)
        self.positional_embeddings = PositionalEmbeddings(max_seq_len = max_seq_len, emb_size = emb_size)
        self.dropout_layer = nn.Dropout(dropout)

        self.decoders = nn.ModuleList([Decoder(
                num_heads=num_heads,
                emb_size=emb_size,
                head_size=head_size,
                max_seq_len=max_seq_len
            )
            for _ in range(num_layers)
        ])

        self.linear = nn.Linear(emb_size, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через модель GPT.

        Args:
            x: Входная последовательность токенов размером
                (batch_size, seq_len).

        Returns:
            Логиты размером (batch_size, seq_len, vocab_size).
        """
        token_embeddings = self.token_embeddings(x)
        positional_embeddings = self.positional_embeddings(x.size(1))

        embeddings = token_embeddings + positional_embeddings # Получаем полноценный эмбединг

        embeddings = self.dropout_layer(embeddings)

        for decoder in self.decoders:
            embeddings = decoder(embeddings)

        logits = self.linear(embeddings)

        return logits