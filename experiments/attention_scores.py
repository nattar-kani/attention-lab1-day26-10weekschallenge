import torch
import torch.nn as nn

# 8 tokens, each represented by 4 numbers
x = torch.randn(8,4)

# q,k,v projections
Wq = nn.Linear(4,4)
Wk = nn.Linear(4,4)
Wv = nn.Linear(4,4)

Q = Wq(x)
K = Wk(x)
V = Wv(x)

# attention scores
scores = Q @ K.T

print("Q shape:", Q.shape)
print("K shape:", K.shape)
print("V shape:", V.shape)

print("RAw Attention scores: ",scores)
print("Raw Attention scores shape: ",scores.shape)


# scaling the scores
d_k = K.size(-1)
scaled_scores = scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

# causal mask
mask = torch.triu(
    torch.ones(8,8), diagonal=1
)

scaled_scores = scaled_scores.masked_fill(
    mask == 1, float("-inf")
)
print("d_k:", d_k)
print("Scaled Attention scores: ",scaled_scores)

# softmax
attention_weights = torch.softmax(
    scaled_scores, dim=-1
)

print("Attention weights:",attention_weights)
print("Attention weights shape:",attention_weights.shape)

print("Row sum:",attention_weights.sum(dim=-1))

# attention output
attention_output = attention_weights @ V

print("Attention output:",attention_output)
print("Attention output shape:",attention_output.shape)
