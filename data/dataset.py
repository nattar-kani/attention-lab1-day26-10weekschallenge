import torch


input_ids = torch.tensor([
    [101, 1045, 5959, 5983, 8233, 27292, 2050]
])

target_ids = torch.tensor([
    [1045, 5959, 5983, 8233, 27292, 2050, 102]
])


print("Input shape:", input_ids.shape)
print("Target shape:", target_ids.shape)

print("Inputs:")
print(input_ids)

print("Targets:")
print(target_ids)