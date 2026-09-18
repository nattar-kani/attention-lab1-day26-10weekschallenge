import torch
import torch.nn as nn

from src.embeddings import TokenAndPositionEmbedding
from src.transformer_block import TransformerBlock

class AttentionLab(nn.Module):

    def __init__(
            self,
            vocab_size,
            embedding_dim,
            num_heads,
            ff_hidden_dim,
            max_sequence_length
        ):
            super().__init__()

            self.embedding = TokenAndPositionEmbedding(
                  vocab_size=vocab_size,
                  embedding_dim=embedding_dim,
                  max_sequence_length=max_sequence_length
            )

            self.transformer_block = TransformerBlock(
                  embedding_dim=embedding_dim,
                  num_heads=num_heads,
                  ff_hidden_dim=ff_hidden_dim
            )

            self.lm_head = nn.Linear(
                  embedding_dim,vocab_size
            )

    def forward(self, input_ids):

          x = self.embedding(input_ids)

          x = self.transformer_block(x)

          logits = self.lm_head(x)

          return logits

if __name__ == "__main__":
    input_ids = torch.tensor([
        [101, 1045, 5959, 5983, 8233, 27292, 2050, 102]
    ])

    model = AttentionLab(
        vocab_size=30522,
        embedding_dim=4,
        num_heads=2,
        ff_hidden_dim=16,
        max_sequence_length=8
    )

    output = model(input_ids)

    print("Input IDs shape:", input_ids.shape)
    print("logits shape:", output.shape)
