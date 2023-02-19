from transformers.models.bert.modeling_bert import BertLMPredictionHead, BertPreTrainedModel

from models.ann_bert import AnnBertConfig, AnnBert


class AnnBertForMaskedLM(BertPreTrainedModel):

    def __init__(self, config: AnnBertConfig):
        super().__init__(config)
        self.config = config

        self.bert = AnnBert(config)
        self.cls = BertLMPredictionHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.decoder

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

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        return prediction_scores
