from datasets import Dataset
from transformers import AutoTokenizer


sentences = [
    "i enjoy eating shawarma",
    "i enjoy drinking coffee",
    "i like eating pizza",
    "i like drinking tea",
]


dataset = Dataset.from_dict({
    "text": sentences
})


tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)


def tokenize_function(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        max_length=10,
        truncation=True
    )


tokenized_dataset = dataset.map(
    tokenize_function, batched=True
)


print("Tokenized dataset:")
print(tokenized_dataset)

print("\nAll tokenized rows:")

for row in tokenized_dataset:
    print("\nText:", row["text"])
    print("Input IDs:", row["input_ids"])
    print("Attention mask:", row["attention_mask"])