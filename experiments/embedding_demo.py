import torch
import torch.nn as nn
from transformers import pipeline
'''
vocab_size = 10
embedding_dim = 4

embedding = nn.Embedding(
    num_embeddings= vocab_size,
    embedding_dim= embedding_dim
)

token_ids = torch.tensor([2,5,7])

vectors = embedding(token_ids)

print(f"Token IDs: {token_ids}")
print(f"Embeddings: {vectors}")
print(f"Shape: {vectors.shape}")
'''
# with an example
classifier = pipeline("sentiment-analysis")
tokenizer = classifier.tokenizer

text = "I enjoy eating shawarma"

encoded = tokenizer(text)

token_ids = torch.tensor(encoded["input_ids"])

print(f"Token IDs: {token_ids}")
print(f"Total tokens: {len(token_ids)}")

vocab_size = tokenizer.vocab_size
print(f"Vocab size: {vocab_size}")

embedding = nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=4
)

vectors = embedding(token_ids)

print(f"Embeddings: {vectors}")
print(f"Shape: {vectors.shape}")

vocab = tokenizer.get_vocab()

print(len(vocab))
print(list(vocab.items())[:20])
# shows first 20 letters from vocab dict

for token_id in token_ids:
    print(token_id.item(), tokenizer.convert_ids_to_tokens(token_id.item()))
