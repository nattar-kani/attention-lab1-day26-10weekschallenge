import torch


# Tiny vocabulary for the toy experiment
vocab = {
    "[CLS]": 0,
    "i": 1,
    "enjoy": 2,
    "eating": 3,
    "shaw": 4,
    "##arm": 5,
    "##a": 6,
    "[SEP]": 7
}


# Reverse vocabulary for decoding predictions
id_to_token = {
    index: token
    for token, index in vocab.items()
}


# Original sequence:
# [CLS] i enjoy eating shaw ##arm ##a [SEP]

input_ids = torch.tensor([
    [0, 1, 2, 3, 4, 5, 6]
])

target_ids = torch.tensor([
    [1, 2, 3, 4, 5, 6, 7]
])


print("Vocabulary size:", len(vocab))

print("Input shape:", input_ids.shape)
print("Target shape:", target_ids.shape)

print("Inputs:")
print(input_ids)

print("Targets:")
print(target_ids)

print("Vocabulary:")
for token, token_id in vocab.items():
    print(f"{token:8} → {token_id}")