from tokenizers import Tokenizer
from tokenizers.models import BPE # from hugging face 
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

from tokenizers.trainers import BpeTrainer

trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],vocab_size=223)
sufix="4maps_alpha"

files = [f"data/microstate_{sufix}_2sE_dataset_pretokenize/output_{sufix}_2sE.{split}.txt" for split in ["test", "train", "valid"]]

tokenizer.train(files, trainer,)

tokenizer.save(f"data/tokenizer-output-{sufix}-2sE-NMT.json")

print(tokenizer.get_vocab())
print(len(tokenizer.get_vocab()))

#by default 30K lenght vocabulary RUBEN review how to calculate the optimize lenght of vocabulary