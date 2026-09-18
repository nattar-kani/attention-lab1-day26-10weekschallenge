import torch

batch_size = 1
sequence_len = 8
embedding_dim = 4
num_heads = 2

head_dim = embedding_dim // num_heads

Q = torch.randn(batch_size, sequence_len, embedding_dim)
K = torch.randn(batch_size, sequence_len, embedding_dim)
V = torch.randn(batch_size, sequence_len, embedding_dim)

print("Before splitting")
print("Q:", Q)

# splitting embedding dimensions across 2 heads

Q = Q.view(
    batch_size,sequence_len,
    num_heads,head_dim
)
K = K.view(
    batch_size,sequence_len,
    num_heads,head_dim
)
V = V.view(
    batch_size,sequence_len,
    num_heads,head_dim
)

print("After splitting")
print("Q:", Q.shape)
print("K:", K.shape)
print("V:", V.shape)

Q = Q.transpose(1, 2)
K = K.transpose(1, 2)
V = V.transpose(1, 2)

print("After transpose")
print("Q:", Q.shape)
print("K:", K.shape)
print("V:", V.shape)

# attention scores for every head
scores = torch.matmul(
    Q, K.transpose(-2,-1)
)

print("Attention scores shape:", scores.shape)

# scaling
scores = scores / torch.sqrt(
    torch.tensor(head_dim, dtype=torch.float32)
)

print("Scaled scores shape:", scores.shape)

# causal masking in all the heads
mask = torch.triu(
    torch.ones(sequence_len, sequence_len), diagonal=1
)

print("Causal mask:",mask)

scores = scores.masked_fill(
    mask == 1, float("-inf")
)

attention_weights = torch.softmax(
    scores,dim=-1
)

print("Attention weights shape:", attention_weights.shape)
print("Head 1 attention weights:", attention_weights[0,0])
print("Head 2 attention weights:", attention_weights[0,1])

attention_ouput = torch.matmul(
    attention_weights,V
)

print("Attention output shape (still with individual heads so 2 dimensions):",attention_ouput.shape)

attention_ouput = attention_ouput.transpose(1,2)

print("Attention outputs shape (after transpose):",attention_ouput.shape)

attention_ouput = attention_ouput.contiguous().view(
    batch_size,sequence_len,embedding_dim
)

print("Attention outputs shape(after combining heads):", attention_ouput.shape)

output_projection = torch.nn.Linear(
    embedding_dim,embedding_dim
)
final_output = output_projection(attention_ouput)

print("Final multi-head attention output shape:",final_output.shape)