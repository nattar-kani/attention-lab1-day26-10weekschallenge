import torch
import torch.nn as nn

class FeedForward(nn.Module):

    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(embedding_dim,hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim,embedding_dim)
        )

    def forward(self,x):
        return self.network(x)
    
'''
if __name__ == "__main__":
    x = torch.randn(1,8,4)

    feed_forward = FeedForward(
        embedding_dim=4, hidden_dim=16
    )

    output = feed_forward(x)

    print("Input shape:", x.shape)
    print("Output shape:",output.shape)'''