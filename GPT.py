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

    def generate(self, x: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        Генерирует новые токены на основе входной последовательности.

        Args:
            x: Входная последовательность токенов размером
                (batch_size, seq_len).
            max_new_tokens: Количество токенов, которые необходимо сгенерировать.

        Returns:
            Последовательность размером
            (batch_size, seq_len + max_new_tokens).
        """
        for _ in range(max_new_tokens):
            # Оставляем только последние max_seq_len токенов
            x_context = x[:, -self.max_seq_len:]

            # Получаем логиты
            logits = self.forward(x_context)

            # Берём логиты только последнего токена
            logits = logits[:, -1, :]

            # Преобразуем логиты в вероятности через софтмакс
            probabilities = torch.softmax(logits, dim=-1)

            # Выбираем токен с максимальной вероятностью
            next_token = torch.argmax(probabilities, dim=-1, keepdim=True)

            # Добавляем новый токен в конец последовательности
            x = torch.cat((x, next_token), dim=1)

        return x
    
    def save(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'vocab_size': self.vocab_size,
            'max_seq_len': self.max_seq_len,
            'emb_size': self.emb_size,
            'num_heads': self.num_heads,
            'head_size': self.head_size,
            'num_layers': self.num_layers
        }, path)
        
    @classmethod
    def load(cls, path, device):
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            vocab_size=checkpoint['vocab_size'],
            max_seq_len=checkpoint['max_seq_len'],
            emb_size=checkpoint['emb_size'],
            num_heads=checkpoint['num_heads'],
            head_size=checkpoint['head_size'],
            num_layers=checkpoint['num_layers']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        return model


if __name__ == "__main__":
    # Пример сохранения и загрузки модели GPT
    gpt = GPT(
    vocab_size=vocab_size,
    max_seq_len=max_seq_len,
    emb_size=emb_size,
    num_heads=num_heads,
    head_size=head_size,
    num_layers=num_layers
    )


    gpt.save("data/gpt_model.pth")

    gpt = GPT.load("data/gpt_model.pth", device=device)