import torch
import matplotlib.pyplot as plt

from src.attention_lab import AttentionLab
from data.toy_dataset import input_ids, target_ids, vocab, id_to_token


# --------------------------------------------------
# 1. Model configuration
# --------------------------------------------------

vocab_size = len(vocab)

embedding_dim = 4
num_heads = 2
ff_hidden_dim = 16
max_sequence_length = 7


# --------------------------------------------------
# 2. Create model
# --------------------------------------------------

model = AttentionLab(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    ff_hidden_dim=ff_hidden_dim,
    max_sequence_length=max_sequence_length
)


# --------------------------------------------------
# 3. Optimizer and loss
# --------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001
)

loss_function = torch.nn.CrossEntropyLoss()


# --------------------------------------------------
# 4. Training
# --------------------------------------------------

epochs = 500

losses = []


for epoch in range(epochs):

    # Clear previous gradients
    optimizer.zero_grad()

    # Forward pass
    logits = model(input_ids)

    # Shape assertions
    assert input_ids.shape == (1, 7)
    assert target_ids.shape == (1, 7)
    assert logits.shape == (1, 7, vocab_size)

    # Calculate loss
    loss = loss_function(
        logits.view(-1, logits.size(-1)),
        target_ids.view(-1)
    )

    # Store loss
    losses.append(loss.item())

    # Backpropagation
    loss.backward()

    # Update weights
    optimizer.step()

    # Print progress
    if (epoch + 1) % 50 == 0:

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- Loss: {loss.item():.4f}"
        )


# --------------------------------------------------
# 5. Plot training curve
# --------------------------------------------------

plt.plot(losses)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("AttentionLab Toy Training Loss")

plt.show()


# --------------------------------------------------
# 6. Generate predictions
# --------------------------------------------------

model.eval()

with torch.no_grad():

    logits = model(input_ids)

    predicted_ids = torch.argmax(
        logits,
        dim=-1
    )


# --------------------------------------------------
# 7. Decode predictions
# --------------------------------------------------

input_tokens = [
    id_to_token[token_id]
    for token_id in input_ids[0].tolist()
]

target_tokens = [
    id_to_token[token_id]
    for token_id in target_ids[0].tolist()
]

predicted_tokens = [
    id_to_token[token_id]
    for token_id in predicted_ids[0].tolist()
]


# --------------------------------------------------
# 8. Display predictions
# --------------------------------------------------

print("\nPredictions:")

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


# --------------------------------------------------
# 9. Calculate accuracy
# --------------------------------------------------

accuracy = (
    predicted_ids == target_ids
).float().mean().item()


print(f"\nToken Accuracy: {accuracy:.2%}")