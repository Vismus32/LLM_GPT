import dill
import torch

from torch.utils.data import DataLoader

from data.DataLoad import GetData
from GPT import GPT


def main():
    # Параметры обучения
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seq_len = 64
    batch_size = 32

    vocab_size = 2000
    max_seq_len = seq_len

    emb_size = 128
    num_heads = 4
    head_size = 32
    num_layers = 2
    dropout = 0.1

    learning_rate = 2.5e-4
    num_epoch = 2

    print("Device:", device)

    # Загружаем токены
    with open("data/tokens.dill", "rb") as file:
        token_ids = dill.load(file)

    print("Количество токенов:", len(token_ids))

    # Делим корпус на train и validation
    n = int(0.9 * len(token_ids))

    train_token_ids = token_ids[:n]
    valid_token_ids = token_ids[n:]

    print("Train токенов:", len(train_token_ids))
    print("Valid токенов:", len(valid_token_ids))

    # Создаём тренировочный датасет
    train_dataset = GetData(data=train_token_ids, seq_len=seq_len, device=device)

    # Создаём валидационный датасет
    valid_dataset = GetData(data=valid_token_ids, seq_len=seq_len, device=device)

    # Создаём DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    print("Train batches:", len(train_loader))
    print("Valid batches:", len(valid_loader))

    # Создаём модель
    model = GPT(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        emb_size=emb_size,
        num_heads=num_heads,
        head_size=head_size,
        num_layers=num_layers,
        dropout=dropout,
        device=device
    )

    # Обучаем модель
    model.fit(
        train_loader=train_loader,
        valid_loader=valid_loader,
        num_epoch=num_epoch,
        learning_rate=learning_rate
    )


if __name__ == "__main__":
    main()