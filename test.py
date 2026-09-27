import dill
import torch

from GPT import GPT
from Embeddings_process.BPE import BPE


def main():
    device = "cpu"

    model = GPT.load(
        "data/gpt_model_epoch_2.pth",
        device=device
    )

    model.eval()

    bpe = BPE.load("data/bpe.dill")

    prompt = "Я вас любил"

    token_ids = bpe.encode(prompt)

    x = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device
    )

    generated = model.generate(
        x,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.8,
        top_k=20
    )

    generated_token_ids = generated[0].tolist()

    text = bpe.decode(generated_token_ids)

    print()
    print("PROMPT:")
    print(prompt)
    print()
    print("GENERATED:")
    print(text)


if __name__ == "__main__":
    main()
