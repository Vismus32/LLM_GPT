import torch.nn as nn
import torch

from tqdm import tqdm

from Embeddings_process.Embeddings import PositionalEmbeddings, TokenEmbeddings
from Transformer.Decoder import Decoder

from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn.functional import cross_entropy


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

    def generate(self, x: torch.Tensor, max_new_tokens: int, do_sample: bool = False,
                temperature: float = 1.0, top_k: int = None, top_p: float = None) -> torch.Tensor:
        """
        Генерирует новые токены на основе входной последовательности.

        Args:
            x: Входная последовательность токенов размером
                (batch_size, seq_len).
            max_new_tokens: Количество токенов, которые необходимо сгенерировать.
            do_sample: Флаг, указывающий на необходимость выбора токена с учетом вероятностей.
            temperature: Температура для регулирования случайности выбора токенов.

        Returns:
            Последовательность размером
            (batch_size, seq_len + max_new_tokens).
        """
        for _ in range(max_new_tokens):
            # Оставляем только последние max_seq_len токенов
            x_context = x[:, -self.max_seq_len:]

            # Получаем логиты
            logits = self.forward(x_context)

            # Делим на температуру
            logits = logits / temperature

            # Берём логиты только последнего токена
            logits = logits[:, -1, :]

            if do_sample:
                # top_k
                if top_k is not None:
                    top_k_values, _ = torch.topk(logits, top_k, dim=-1)

                    # Значение k-го по величине логита
                    min_top_k = top_k_values[:, -1].unsqueeze(-1)

                    # Всё, что меньше k-го логита, заменяем на -inf
                    logits = torch.where(logits < min_top_k, torch.full_like(logits, -float("Inf")), logits)

                # top_p
                if top_p is not None:
                    # Получаем вероятности
                    probabilities = torch.softmax(logits, dim=-1)

                    # Сортируем вероятности по убыванию
                    sorted_probs, sorted_indices = torch.sort(
                        probabilities,
                        descending=True,
                        dim=-1
                    )

                    # Считаем кумулятивные вероятности
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                    # Определяем токены, которые нужно удалить
                    sorted_indices_to_remove = cumulative_probs > top_p

                    sorted_indices_to_remove[:, 0] = 0

                    # Возвращаем маску из отсортированного порядка
                    # в исходный порядок токенов
                    indices_to_remove = torch.zeros_like(sorted_indices_to_remove)

                    indices_to_remove.scatter_(
                        dim=-1,
                        index=sorted_indices,
                        src=sorted_indices_to_remove
                    )

                    # Заменяем логиты удаляемых токенов на -inf
                    logits = logits.masked_fill(
                        indices_to_remove.byte(),
                        -float("Inf")
                    )


            # Преобразуем логиты в вероятности через софтмакс
            probabilities = torch.softmax(logits, dim=-1)

            if do_sample:
                # Случайно выбираем токен согласно его вероятности
                next_token = torch.multinomial(
                    probabilities,
                    num_samples=1
                )
            else:
                # Выбираем токен с максимальной вероятностью
                next_token = torch.argmax(
                    probabilities,
                    dim=-1,
                    keepdim=True
                )

            # Добавляем новый токен в конец последовательности
            x = torch.cat((x, next_token), dim=1)

        return x


    def fit(self, train_loader: DataLoader, valid_loader: DataLoader, num_epoch: int, learning_rate: float):
        """
        Обучает модель GPT на тренировочной выборке и оценивает её
        качество на валидационной выборке.

        Args:
            train_loader: DataLoader для тренировочной выборки.
                Каждый элемент должен содержать пару (inputs, targets).
            valid_loader: DataLoader для валидационной выборки.
                Каждый элемент должен содержать пару (inputs, targets).
            num_epoch: Количество эпох обучения.
            learning_rate: Скорость обучения для оптимизатора Adam.

        В процессе обучения модель переводится в режим train(),
        а при валидации — в режим eval() с отключением вычисления
        градиентов.

        Атрибуты:
            train_loss: Последнее значение функции потерь на тренировочной
                выборке.
            valid_loss: Последнее значение функции потерь на валидационной
                выборке.
        """


        # Переводим модель на нужное устройство
        self.to(self.device)

        # Создаём оптимизатор
        optimizer = Adam(self.parameters(), lr=learning_rate)

        for epoch in range(num_epoch):
            # Режим обучения
            self.train()

            train_losses = []

            for inputs, targets in train_loader:
                # Переносим данные на device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                logits = self.forward(inputs)

                # Переобразуем [batch_size, seq_len, vocab_size] в [batch_size * seq_len, vocab_size] для вычисления потерь
                logits = logits.view(-1, self.vocab_size)

                # [batch_size, seq_len] переобразуем в [batch_size * seq_len] для вычисления потерь
                targets = targets.view(-1)

                # Cross entropy loss
                loss = cross_entropy(logits, targets)

                # Сохраняем loss внутри класса
                self.train_loss = loss

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Шаг оптимизатора
                optimizer.step()

                train_losses.append(loss.item())

            # Средний training loss
            mean_train_loss = sum(train_losses) / len(train_losses)

            print("Epoch {}/{} - train loss: {}".format(epoch + 1, num_epoch, mean_train_loss))

            # Режим оценки
            self.eval()

            valid_losses = []

            # Отключаем вычисление градиентов
            with torch.no_grad():
                for inputs, targets in valid_loader:
                    # Переносим данные на device
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    # Forward pass
                    logits = self.forward(inputs)

                    # Переобразуем [batch_size, seq_len, vocab_size] в [batch_size * seq_len, vocab_size] для вычисления потерь
                    logits = logits.view(-1, self.vocab_size)

                    # [batch_size, seq_len] переобразуем в [batch_size * seq_len] для вычисления потерь
                    targets = targets.view(-1)

                    # Validation loss
                    loss = cross_entropy(logits, targets)

                    # Сохраняем loss внутри класса
                    self.valid_loss = loss

                    valid_losses.append(loss.item())

            # Средний validation loss
            mean_valid_loss = sum(valid_losses) / len(valid_losses)

            print("Epoch {}/{} - valid loss: {}".format(epoch + 1, num_epoch, mean_valid_loss))

            # Локальное сохранение текущей версии модели
            self.save("data/gpt_model_epoch_{}.pth".format(epoch + 1))


    
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
    def load(cls, path, device: str = "cpu"):
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

    gpt = GPT.load("data/gpt_model.pth")