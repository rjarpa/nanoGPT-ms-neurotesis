# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

out_dir = 'out-microstate-final2'
wandb_log = False #allow to create accunt to log results
wandb_project = 'owt'
#wandb_run_name='gpt2-124M'
wandb_run_name='microstate-M'

dataset = 'microstate'



# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 24
#block_size = 1024
batch_size=128
#gradient_accumulation_steps = 5 * 8
gradient_accumulation_steps = 5 * 1

# this makes total number of tokens be 300B
#max_iters = 600000
max_iters = 100000 #Cambiar a 10000
#lr_decay_iters = 600000
lr_decay_iters = 100000 #Cambiar a 10000

# eval stuff
#eval_interval = 1000
eval_interval = 10
#eval_iters = 200
eval_iters = 2
#log_interval = 10
log_interval = 2

# weight decay
weight_decay = 1e-1

eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

gradient_accumulation_steps = 1
batch_size = 12
block_size = 96 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 12
n_head = 12
n_embd = 384
dropout = 0.2

min_lr = 1e-2 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially