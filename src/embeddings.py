import torch
import torch.nn as nn


class TokenAndPositionEmbedding(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        max_sequence_length
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.position_embedding = nn.Embedding(
            max_sequence_length,
            embedding_dim
        )

    def forward(self, input_ids):

        batch_size, sequence_length = input_ids.shape

        token_vectors = self.token_embedding(input_ids)

        positions = torch.arange(
            sequence_length,
            device=input_ids.device
        )

        position_vectors = self.position_embedding(positions)

        x = token_vectors + position_vectors

        return x


if __name__ == "__main__":

    # Example token IDs from:
    # "I enjoy eating shawarma"
    input_ids = torch.tensor([
        [101, 1045, 5959, 5983, 8233, 27292, 2050, 102]
    ])

    embedding = TokenAndPositionEmbedding(
        vocab_size=30522,
        embedding_dim=4,
        max_sequence_length=8
    )

    output = embedding(input_ids)

    print("Input IDs shape:", input_ids.shape)
    print("Embedding output shape:", output.shape)