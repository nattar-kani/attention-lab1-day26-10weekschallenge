import torch
import torch.nn as nn

# transformer input
x = torch.randn(8,4)
embedding_dim = 4

# linear projections
Wq = nn.Linear(embedding_dim, embedding_dim)
Wk = nn.Linear(embedding_dim,embedding_dim)
Wv = nn.Linear(embedding_dim,embedding_dim)

# create q,k,v
Q = Wq(x)
K = Wk(x)
V = Wv(x)

print(f"X shape: {x.shape}")
print(f"Q shape: {Q.shape}")
print(f"K shape: {K.shape}")
print(f"V shape: {V.shape}")
