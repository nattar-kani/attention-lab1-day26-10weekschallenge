import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer 

from src.attention_lab import AttentionLab

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Training data
input_ids = torch.tensor([
    [101, 1045, 5959, 5983, 8233, 27292, 2050]
])

target_ids = torch.tensor([
    [1045, 5959, 5983, 8233, 27292, 2050, 102]
])


# Model
model = AttentionLab(
    vocab_size=30522,
    embedding_dim=4,
    num_heads=2,
    ff_hidden_dim=16,
    max_sequence_length=7
)

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001
)

# Loss function
loss_function = torch.nn.CrossEntropyLoss()

# Training loop
epochs = 100
losses = []

for epoch in range(epochs):

    optimizer.zero_grad()

    # Forward pass
    logits = model(input_ids)

    # shape assertions since model is very tiny
    assert input_ids.shape == (1, 7)

    assert target_ids.shape == (1, 7)

    assert logits.shape == (1, 7, 30522)
    
    # Calculate loss
    loss = loss_function(
        logits.view(-1, logits.size(-1)),
        target_ids.view(-1)
    )

    losses.append(loss.item())

    # Backpropagation
    loss.backward()

    # Update weights
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- Loss: {loss.item():.4f}"
        )

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("AttentionLab Training Loss")

plt.show()

model.eval()

with torch.no_grad():
    logits = model(input_ids)

    # select highest-scoring token
    predicted_ids = torch.argmax(logits,dim=-1)
    
input_tokens = tokenizer.convert_ids_to_tokens(
    input_ids[0].tolist()
)

target_tokens = tokenizer.convert_ids_to_tokens(
    target_ids[0].tolist()
)

predicted_tokens = tokenizer.convert_ids_to_tokens(
    predicted_ids[0].tolist()
)

print("Predictions:")

for input_token, target_token, predicted_token in zip(
    input_tokens,
    target_tokens,
    predicted_tokens
):

    print(
        f"{input_token:10} → "
        f"Target: {target_token:10} "
        f"Prediction: {predicted_token}"
    )

accuracy = (
    predicted_ids == target_ids
).float().mean().item()


print(f"\nToken Accuracy: {accuracy:.2%}")