import cs
import os
from models.ann_bert import AnnBertConfig


# ==================================================================================================
# Task Config

task_name = 'bert_metric'
task_comment = 'comment'


# ==================================================================================================
# Training Config

log_interval = 20 * 60          # seconds
checkpoint_interval = 60 * 60   # seconds
dataloader_workers = 2
last_training_time = 1623682295
last_step = 69690

# Relative path of pretrain model file from cs.DATASET_DIR
# This value can be None, it means training from scratch
pretrained_model = 'masked_lm_1622528325/model_70209.pt'


# ==================================================================================================
# Ann-bert Model Config

# If last_training_time == 0，it means train form scratch, the model configuration is as follows
# If last_training_time != 0，it means further-pretrain from a checkpoint, it will use the model configuration from checkpoint directionary

model_conf = AnnBertConfig(
    vocab_size=67,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=768 * 4,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_char_position_embeddings=256,
    max_word_position_embeddings=64,
    initializer_range=0.02,
    layer_norm_eps=1e-12,
    pad_token_id=0,
    gradient_checkpointing=False,
    add_pooling_layer='tanh',
    tie_word_embeddings=True,
)


# ==================================================================================================
# Dataset Config

# dataset_name = '210422'
training_set_file = cs.save_pkl_root + 'remain_train.pkl'
val_set_file = cs.save_pkl_root + 'dev.pkl'
test_set_file = cs.save_pkl_root + 'test.pkl'
id_to_cls_file = f'{cs.DATASET_DIR}/id_to_cls.pkl'
nor2len_file = cs.save_pkl_root + '210422_nor2len_dict.pkl'
aff_id_to_nor_file = cs.save_pkl_root + 'afid2nor.pkl'
manager_host = '10.10.10.3'
manager_port = 50003
manager_password = b'something'

anchors_file = cs.save_pkl_root + 'anchor.pkl'
# anchors_file = os.path.join(cs.DATASET_DIR, dataset_name, 'anchors.pkl')


# ==================================================================================================
# Hyper-parameters

init_lr = 2e-5
batch_size = 16
samples_per_step = 64
batches_per_step = samples_per_step // batch_size // cs.num_devices
steps_per_validation = 10

# The learning rate schedule 
# First： it increases to inin_lr from 0 linearly
num_warmup_steps = 20000
# Second：it decreases to lr_decay_mul*init_lr from inin_lr linearly
lr_decay_mul = 0
num_decay_steps = 1000000

contrastive_loss_margin = 2.0
samples_per_class = 2
resample_exp = 0.3
