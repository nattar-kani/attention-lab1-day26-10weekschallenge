import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):

    def __init__(self,embedding_dim,num_heads):
        super().__init__()

        assert embedding_dim % num_heads == 0

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.q_projection = nn.Linear(
            embedding_dim,embedding_dim
        )

        self.k_projection = nn.Linear(
            embedding_dim,embedding_dim
        )

        self.v_projection = nn.Linear(
            embedding_dim,embedding_dim
        )

        self.output_projection = nn.Linear(
            embedding_dim,embedding_dim
        )

    def forward(self,x):
        batch_size,sequence_length,_ = x.shape

        Q = self.q_projection(x)
        K = self.k_projection(x)
        V = self.v_projection(x)

        Q = Q.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        )
        K = K.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        )
        V = V.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        )

        Q = Q.transpose(1,2)
        K = K.transpose(1,2)
        V = V.transpose(1,2)

        scores = torch.matmul(Q,K.transpose(-2, -1))

        scores = scores / torch.sqrt(
            torch.tensor(
                self.head_dim,
                dtype=torch.float32,
                device=scores.device
            )
        )

        mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=scores.device
            ),
            diagonal=1
        )

        scores = scores.masked_fill(
            mask == 1,
            float("-inf")
        )


        attention_weights = torch.softmax(scores, dim=-1)

        attention_output = torch.matmul(
            attention_weights,
            V
        )

        attention_output = attention_output.transpose(1, 2)

        attention_output = attention_output.contiguous().view(
            batch_size,
            sequence_length,
            self.embedding_dim
        )

        output = self.output_projection(
            attention_output
        )

        return output
    

'''if __name__ == "__main__":
    x = torch.randn(1,8,4)

    attention = MultiHeadAttention(
        embedding_dim=4, num_heads=2
    )

    output = attention(x)'''