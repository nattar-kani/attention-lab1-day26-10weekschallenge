import torch
import matplotlib.pyplot as plt

from src.attention_lab import AttentionLab
from data.small_text_dataset import (
    input_ids,
    target_ids,
    vocab,
    id_to_token
)


# --------------------------------------------------
# 1. Model configuration
# --------------------------------------------------

vocab_size = len(vocab)

embedding_dim = 8
num_heads = 2
ff_hidden_dim = 32
max_sequence_length = 5


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
    assert input_ids.shape == (4, 5)
    assert target_ids.shape == (4, 5)
    assert logits.shape == (4, 5, vocab_size)

    # Calculate loss
    loss = loss_function(
        logits.view(-1, logits.size(-1)),
        target_ids.view(-1)
    )

    # Store loss
    losses.append(loss.item())

    # Backpropagation
    loss.backward()

    # Update model weights
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
plt.figure(figsize=(8, 5))
plt.plot(losses)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("AttentionLab Small Text Training Loss")

plt.savefig(
    "training_curve.png",
    dpi=150,
    bbox_inches="tight"
)

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
# 7. Calculate token accuracy
# --------------------------------------------------

accuracy = (
    predicted_ids == target_ids
).float().mean().item()


print(f"\nToken Accuracy: {accuracy:.2%}")


# --------------------------------------------------
# 8. Decode predictions
# --------------------------------------------------

print("\nPredictions:")

for row in range(input_ids.size(0)):

    input_tokens = [
        id_to_token[token_id]
        for token_id in input_ids[row].tolist()
    ]

    target_tokens = [
        id_to_token[token_id]
        for token_id in target_ids[row].tolist()
    ]

    predicted_tokens = [
        id_to_token[token_id]
        for token_id in predicted_ids[row].tolist()
    ]

    print(f"\nSequence {row + 1}:")

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