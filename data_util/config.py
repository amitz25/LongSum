import os

root_dir = os.path.dirname(__file__)

arxiv_dataset = True
dataset_name = 'arxiv-release' if arxiv_dataset else 'PubMed'

train_data_path = os.path.join(root_dir, dataset_name, "chunked", "train_*")
eval_data_path = os.path.join(root_dir, dataset_name, "chunked", "val_*")
decode_data_path = os.path.join(root_dir, dataset_name, "chunked", "test_*")
vocab_path = os.path.join(root_dir, dataset_name, "vocab")
log_root = os.path.join(root_dir, "log")

# Hyperparameters
hidden_dim= 256
emb_dim= 128
batch_size= 4
max_article_size=2000
max_section_size=500
max_num_sections=4
max_dec_steps=210
beam_size=4
min_dec_steps=35
vocab_size=50000

lr=0.15
adagrad_init_acc=0.1
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4
max_grad_norm=2.0

pointer_gen = True
is_coverage = False
cov_loss_wt = 1.0

eps = 1e-12
max_iterations = 500000

use_gpu=True

lr_coverage=0.15

use_maxpool_init_ctx = False
