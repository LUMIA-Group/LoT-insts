import io

import torch
import torch.nn as nn
import torch.nn.functional as nnf

from transformers import PretrainedConfig
from transformers.models.bert.modeling_bert import BertEncoder, BertPreTrainedModel, BertPooler


class AnnBertConfig(PretrainedConfig):
    model_type = "bert"

    def __init__(
            self,
            vocab_size=67,
            hidden_size=256,
            num_hidden_layers=6,
            num_attention_heads=4,
            intermediate_size=256*4,
            hidden_act="gelu",
            hidden_dropout_prob=0.03,
            attention_probs_dropout_prob=0.03,
            max_char_position_embeddings=256,
            max_word_position_embeddings=64,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=0,
            gradient_checkpointing=False,
            add_pooling_layer='none',  # 'none', 'tanh', 'l2-norm'
            **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.kwargs = kwargs

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_char_position_embeddings = max_char_position_embeddings
        self.max_word_position_embeddings = max_word_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.gradient_checkpointing = gradient_checkpointing
        self.add_pooling_layer = add_pooling_layer

    def __str__(self):
        buf = io.StringIO()
        buf.write('<AnnBertConfig:\n')
        buf.write(f'vocab_size={self.vocab_size},\n')
        buf.write(f'hidden_size={self.hidden_size},\n')
        buf.write(f'num_hidden_layers={self.num_hidden_layers},\n')
        buf.write(f'num_attention_heads={self.num_attention_heads},\n')
        buf.write(f'intermediate_size={self.intermediate_size},\n')
        buf.write(f'hidden_act={self.hidden_act},\n')
        buf.write(f'hidden_dropout_prob={self.hidden_dropout_prob},\n')
        buf.write(f'attention_probs_dropout_prob={self.attention_probs_dropout_prob},\n')
        buf.write(f'max_char_position_embeddings={self.max_char_position_embeddings},\n')
        buf.write(f'max_word_position_embeddings={self.max_word_position_embeddings},\n')
        buf.write(f'initializer_range={self.initializer_range},\n')
        buf.write(f'layer_norm_eps={self.layer_norm_eps},\n')
        buf.write(f'pad_token_id={self.pad_token_id},\n')
        buf.write(f'gradient_checkpointing={self.gradient_checkpointing},\n')
        buf.write(f'add_pooling_layer={self.add_pooling_layer},\n')
        for k, v in self.kwargs.items():
            buf.write(f'{k}={v},\n')
        buf.write('>')
        return buf.getvalue()


class AnnBertEmbeddings(nn.Module):

    def __init__(self, config: AnnBertConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.char_position_embeddings = nn.Embedding(config.max_char_position_embeddings, config.hidden_size)
        self.word_position_embeddings = nn.Embedding(config.max_word_position_embeddings, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, char_position_ids=None, word_position_ids=None):
        bs, seq_length = input_ids.size()
        device = input_ids.device

        if char_position_ids is None:
            char_position_ids = torch.arange(seq_length, device=device).expand((bs, -1))
        if word_position_ids is None:
            word_position_ids = torch.zeros(bs, seq_length, device=device, dtype=torch.long)

        char_embeddings = self.word_embeddings(input_ids)
        char_position_embeddings = self.char_position_embeddings(char_position_ids)
        word_position_embeddings = self.word_position_embeddings(word_position_ids)
        embeddings = char_embeddings + char_position_embeddings + word_position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertPoolerWithL2Norm(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = nnf.normalize(pooled_output, dim=-1)
        return pooled_output


class AnnBert(BertPreTrainedModel):

    def __init__(self, config: AnnBertConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = AnnBertEmbeddings(config)
        self.encoder = BertEncoder(config)

        if config.add_pooling_layer == 'tanh':
            self.pooler = BertPooler(config)
        elif config.add_pooling_layer == 'l2-norm':
            self.pooler = BertPoolerWithL2Norm(config)
        else:
            self.pooler = None

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
            self,
            input_ids,
            attention_mask=None,
            char_position_ids=None,
            word_position_ids=None,
            head_mask=None,
            output_attentions=None,
            output_hidden_states=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        input_shape = input_ids.size()
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, char_position_ids=char_position_ids, word_position_ids=word_position_ids
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        # sequence_output, pooled_output, (all_hidden_states), (all_attentions)
        return (sequence_output, pooled_output) + encoder_outputs[1:]
