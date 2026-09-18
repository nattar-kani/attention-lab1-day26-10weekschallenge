import torch 
import torch.nn as nn

from src.multi_head_attention import MultiHeadAttention
from src.feed_forward import FeedForward

class TransformerBlock(nn.Module):

    def __init__(
            self,
            embedding_dim,
            num_heads,
            ff_hidden_dim
    ):
        super().__init__()

        self.attention = MultiHeadAttention(
            embedding_dim, num_heads
        )

        self.feed_forward = FeedForward(
            embedding_dim, ff_hidden_dim
        )

        self.norm1 = nn.LayerNorm(embedding_dim)

        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self,x):

        attention_output = self.attention(x)

        x = self.norm1(x + attention_output)

        ff_output = self.feed_forward(x)

        x = self.norm2(x + ff_output)

        return x

if __name__ == "__main__":
    x = torch.randn(1,8,4)

    block = TransformerBlock(
        embedding_dim=4,
        num_heads=2,
        ff_hidden_dim=16
    )

    output = block(x)

    print("Input shape:",x.shape)
    print("Output shape:",output.shape)