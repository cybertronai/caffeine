import torch

from model import VanillaSelfAttention


def main() -> None:
    torch.manual_seed(42)

    batch_size = 3
    sequence_length = 5
    embed_dim = 8

    model = VanillaSelfAttention(embed_dim)
    inputs = torch.randn(batch_size, sequence_length, embed_dim)
    outputs = model(inputs)

    print("Random vanilla self-attention demo")
    print(f"seed: 42")
    print(f"input shape:  {tuple(inputs.shape)}")
    print(f"output shape: {tuple(outputs.shape)}")
    print("\ninputs:")
    print(inputs)
    print("\noutputs:")
    print(outputs)


if __name__ == "__main__":
    main()
