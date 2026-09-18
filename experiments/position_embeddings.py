import torch
import torch.nn as nn

vocab_size = 30522
embedding_dim = 4
max_sequence_length = 8

# token embedding
token_embedding = nn.Embedding(
        num_embeddings=vocab_size,
        embedding_dim=embedding_dim
)

# positional embedding
position_embedding = nn.Embedding(
    num_embeddings=max_sequence_length,
    embedding_dim= embedding_dim
)

# our token ids - i love shawarma
token_ids = torch.tensor([
    101,1045,5959,5983,8233,27292,2050, 102
])
# position ids: 0,1,2,...7
positions = torch.arange(len(token_ids))
# converting ids to vectors
token_vectors = token_embedding(token_ids)
position_vectors = position_embedding(positions)

x = token_vectors+position_vectors

print(f"Token embedding shape: {token_vectors.shape}")
print(f"Position embedding shape: {position_vectors.shape}")
print(f"Output shape: {x.shape}")
