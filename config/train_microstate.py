# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

sufix="32maps_nonalpha"
out_dir =f'out-microstate-{sufix}-2sE-nmt_log_2'
wandb_log = True #allow to create accunt to log results
wandb_project = 'tesis-microstate'
#wandb_run_name = '4maps-1000' # 'run' + str(time.time())
#wandb_run_name = 'gpt2' # 'run' + str(time.time())
wandb_run_name=f'microstate-{sufix}-2sE_delete'

dataset = f'microstate_{sufix}_2sE_bpe_tokenized_nmt' # name of folder



# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 24
#block_size = 1024
batch_size=128
#gradient_accumulation_steps = 5 * 8
gradient_accumulation_steps = 5 * 1

# this makes total number of tokens be 300B
#max_iters = 600000
max_iters = 10000 #Cambiar a 10000
#lr_decay_iters = 600000
lr_decay_iters = 10000 #Cambiar a 10000
#Dracula used 100000 move info below to document
"""
saving checkpoint to out-microstate-99maps-2sE-nmt
iter 99900: loss 0.5921, time 8419.77ms, mfu 2.28%
iter 99920: loss 0.5672, time 248.43ms, mfu 2.35%
iter 99940: loss 0.6621, time 248.96ms, mfu 2.42%
iter 99960: loss 0.6289, time 248.03ms, mfu 2.48%
iter 99980: loss 0.7324, time 248.96ms, mfu 2.53%
step 100000: train loss 0.3431, val loss 8.8511
"""
#Orgullo y prejuicio
"""
step 900: train loss 5.4748, val loss 6.4939
saving checkpoint to out-microstate-99maps-2sE-nmt
iter 900: loss 5.6145, time 8579.71ms, mfu 2.31%
iter 920: loss 5.6985, time 245.50ms, mfu 2.39%
iter 940: loss 5.8711, time 246.04ms, mfu 2.45%
iter 960: loss 5.7648, time 246.22ms, mfu 2.51%
iter 980: loss 5.8566, time 246.42ms, mfu 2.56%
step 1000: train loss 5.7018, val loss 6.6340
"""

# weight decay
weight_decay = 1e-1
#eval_interval = 250 # keep frequent because we'll overfit
#eval_iters = 200
#log_interval = 10 # don't print too too often

# eval stuff
#eval_interval = 1000
eval_interval = 100
#eval_iters = 200
eval_iters = 10
#log_interval = 10
log_interval = 20


gradient_accumulation_steps = 1
batch_size = 24 #
block_size = 128 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.2
 
min_lr = 1e-2 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially