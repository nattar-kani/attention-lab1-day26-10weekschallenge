from transformers import pipeline


text = "I enjoy eating shawarma"

classifier = pipeline("sentiment-analysis")
# The pipeline isn't itself the Transformer model
# It's an orchestrator that connects the tokenizer and model together

print(f"Model: {classifier.model}")
print(f"Tokenizer: {classifier.tokenizer}")

tokenizer = classifier.tokenizer

tokens = tokenizer.tokenize(text)
token_ids = tokenizer.encode(text)

print(f"Tokens: {tokens}")
print(f"Token IDs: {token_ids}")

result = classifier(text)

print(f"Result: {result}")

encoded = tokenizer(text)
print(f"Encoded: {encoded}")
