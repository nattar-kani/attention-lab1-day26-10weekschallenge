import torch

sentences = [
    "i enjoy eating shawarma",
    "i enjoy drinking coffee",
    "i like eating pizza",
    "i like drinking tea",
]

special_tokens = [
    "[CLS]",
    "[SEP]",
    "[PAD]"
]

words =[]

for sentence in sentences:
    words.extend(
        sentence.split()
    )

unique_words = sorted(set(words))

vocab_tokens = (special_tokens + unique_words)

vocab = {
    token:index
    for index,token in enumerate(vocab_tokens)
}

id_to_token = {
    index: token 
    for token,index in vocab.items()
}

tokenized_sentences = []

for sentence in sentences:
    tokens = (
        ["[CLS]"] + sentence.split()+ ["[SEP]"]
    )
    token_ids = [vocab[token] for token in tokens]
    tokenized_sentences.append(token_ids)

input_sequences = []
target_sequences = []

for token_ids in tokenized_sentences:
    input_sequences.append(token_ids[:-1])
    target_sequences.append(token_ids[1:])

input_ids = torch.tensor(
    input_sequences,
    dtype=torch.long
)

target_ids = torch.tensor(
    target_sequences,
    dtype=torch.long
)

print("Vocabulary size:", len(vocab))

print("\nVocabulary:")

for token, token_id in vocab.items():

    print(
        f"{token:10} → {token_id}"
    )


print("\nInput shape:")
print(input_ids.shape)


print("\nTarget shape:")
print(target_ids.shape)


print("\nInput IDs:")
print(input_ids)


print("\nTarget IDs:")
print(target_ids)
