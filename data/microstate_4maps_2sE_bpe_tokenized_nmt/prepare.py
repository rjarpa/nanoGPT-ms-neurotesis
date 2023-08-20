import os
import requests
import tiktoken
import numpy as np
import json

# download the tiny shakespeare dataset
sufix="4maps"
input_file_path = os.path.join(os.path.dirname(__file__), f'output_{sufix}_2sE.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r') as f:
    data = f.read()
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken gpt2 bpe
#enc = tiktoken.get_encoding("gpt2")

# Step 1: Load the local BPE JSON vocabulary
with open(f"data/tokenizer-output-{sufix}-2sE-NMT.json", "r", encoding="utf-8") as f:
    enc2 = json.load(f)

# Step 2: Create a custom encoder using the loaded BPE vocabulary
def custom_encoder(text):
    tokens = tiktoken.tokenize(text, enc2)
    return tiktoken.bpe_to_text(tokens)

#https://huggingface.co/docs/transformers/fast_tokenizers

from transformers import PreTrainedTokenizerFast
enc = PreTrainedTokenizerFast(tokenizer_file=f"data/tokenizer-output-{sufix}-2sE-NMT.json")


#train_ids = enc.encode_ordinary(train_data)
#val_ids = enc.encode_ordinary(val_data)

train_ids = enc.encode(train_data)
val_ids = enc.encode(val_data)


print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))


#vocab size 1337
#train has 1,499,612 tokens
#val has 165,443 tokens
