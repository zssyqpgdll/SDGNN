import os

root_dir = os.path.expanduser("~")

#train_data_path = os.path.join(root_dir, "ptr_nw/cnn-dailymail-master/finished_files/train.bin")
train_data_path = os.path.join(root_dir, "code/pointer_summarizer/data/finished_files/chunked/train_*")
eval_data_path = os.path.join(root_dir, "code/pointer_summarizer/data/finished_files/chunked/val_*")
decode_data_path = os.path.join(root_dir, "code/pointer_summarizer/data/finished_files/chunked/test_*")
vocab_path = os.path.join(root_dir, "code/pointer_summarizer/data/finished_files/vocab")
log_root = os.path.join(root_dir, "code/pointer_summarizer/log")

# Hyperparameters
hidden_dim= 256 #256
emb_dim= 256 #128
batch_size= 32
max_enc_steps=300
max_dec_steps=80
beam_size= 4
min_dec_steps=1
vocab_size=5000 #50000

lr=0.001 #0.15
adagrad_init_acc=0.1
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4
max_grad_norm=2.0

pointer_gen = True
is_coverage = False
cov_loss_wt = 1.0

eps = 1e-12
max_iterations = 1800 #500000

use_gpu=True

lr_coverage=0.15

gru_times_step = 3

# wikisql: 5; conala: 27; atis: 17
max_layer = 27
