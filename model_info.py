
import os
import torch

from GPT import GPT


def main():
    model_path = "data/gpt_model_epoch_2.pth"
    device = "cpu"

    model = GPT.load(
        model_path,
        device=device
    )

    # Общее количество параметров
    total_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    # Обучаемые параметры
    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    # Размер checkpoint
    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)

    print()
    print("========== MODEL INFO ==========")
    print("Model:", model_path)
    print()
    print("vocab_size:", model.vocab_size)
    print("max_seq_len:", model.max_seq_len)
    print("emb_size:", model.emb_size)
    print("num_heads:", model.num_heads)
    print("head_size:", model.head_size)
    print("num_layers:", model.num_layers)
    print()
    print("Total parameters:", total_params)
    print("Trainable parameters:", trainable_params)
    print("Parameters (millions): {:.2f}M".format(
        total_params / 1000000.0
    ))
    print()
    print("Checkpoint size: {:.2f} MB".format(file_size_mb))
    print("================================")


if __name__ == "__main__":
    main()
