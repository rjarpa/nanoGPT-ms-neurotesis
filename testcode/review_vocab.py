
#https://huggingface.co/docs/transformers/fast_tokenizers

from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast(tokenizer_file=f"data/tokenizer-output-{sufix}-2sE.json")

sufix="4maps"


# Tokenize the dataset and count the occurrences
vocab_count = {}
for sentence in dataset:
    encoding = tokenizer.encode(sentence)
    tokens = encoding.tokens
    for token in tokens:
        vocab_count[token] = vocab_count.get(token, 0) + 1

# Optionally, you can sort the vocabulary by count
sorted_vocab_count = {k: v for k, v in sorted(vocab_count.items(), key=lambda item: item[1], reverse=True)}

# Print the result
for token, count in sorted_vocab_count.items():
    print(f"{token}: {count}")
    