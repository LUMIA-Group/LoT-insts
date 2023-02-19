from transformers.models.bert.modeling_bert import BertPreTrainedModel
import torch.nn as nn

from models.ann_bert import AnnBertConfig, AnnBert


class AnnBertForClassification(BertPreTrainedModel):
    def __init__(self, config: AnnBertConfig):
        super().__init__(config)

        self.bert = AnnBert(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

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
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            char_position_ids=char_position_ids,
            word_position_ids=word_position_ids,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        output = (logits,) + outputs[2:]
        return output
