from tokenizers import Tokenizer
#from tokenizers.models import BPE
from tokenizers import pre_tokenizers, models

tokenizer = Tokenizer(models.BPE())

def custom_pre_tokenizer(text):
    # Remove punctuation
    text = ''.join(char for char in text if char.isalnum())
    # Split into individual characters
    return list(text)

# Create a pre-tokenizer instance with the custom function
pre_tok = pre_tokenizers.PreTokenizer.custom(custom_pre_tokenizer)


tokenizer.pre_tokenizer = pre_tok



dataset = [
    "This is the first sentence.",
    "And this is the second sentence.",
    "Here's the third sentence.",
]
tokenizer.train_from_iterator(dataset)

vocabulary = tokenizer.get_vocab()


for token, frequency in vocabulary.items():
    print(f"Token: {token}, Frequency: {frequency}")

