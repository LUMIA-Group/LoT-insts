import cs
from models.ann_bert import AnnBertConfig


# ==================================================================================================
# Task Config

task_name = 'masked_lm'
task_comment = 'comment'


# ==================================================================================================
# Training Config

log_interval = 10 * 60          # seconds
checkpoint_interval = 60 * 60   # seconds
dataloader_workers = 2
last_training_time = 0
last_step = -1


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
    add_pooling_layer='none',
    tie_word_embeddings=True,
)


# ==================================================================================================
# Dataset Config

corpus_file = cs.save_pkl_root + 'corpus.pkl'
manager_host = '10.10.10.8'
manager_port = 50009
manager_password = b'something'


# ==================================================================================================
# Hyper-parameters

init_lr = 5e-5
batch_size = 32
num_training_steps = 160000
num_warmup_steps = num_training_steps // 100
samples_per_step = 3840
batches_per_step = samples_per_step // batch_size // cs.num_devices

n_batches_in_validating = 4
