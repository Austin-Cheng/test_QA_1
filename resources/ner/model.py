from transformers.modeling_bert import *
from torch.nn.utils.rnn import pad_sequence
from torchcrf import CRF
from torch import nn
from transformers import BertModel, BertTokenizer

class BertNER(nn.Module):
    def __init__(self, config):
        super(BertNER, self).__init__()
        num_labels = len(config.label2id)
        self.num_labels = num_labels

        self.bert = BertModel.from_pretrained(config.bert_model)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.bilstm = nn.LSTM(
            input_size=config.lstm_embedding_size,  # 1024
            hidden_size=config.hidden_size // 2,  # 1024
            batch_first=True,
            num_layers=2,
            dropout=config.lstm_dropout_prob,  # 0.5
            bidirectional=True
        )
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

        # self.init_weights()

    def forward(self, input_data, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, inputs_embeds=None, head_mask=None):
        input_ids, input_token_starts = input_data
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)
        sequence_output = outputs[0]

        # 去除[CLS]标签等位置，获得与label对齐的pre_label表示
        origin_sequence_output = [layer[starts.nonzero().squeeze(1)]
                                  for layer, starts in zip(sequence_output, input_token_starts)]
        # 将sequence_output的pred_label维度padding到最大长度
        padded_sequence_output = pad_sequence(origin_sequence_output, batch_first=True)
        # dropout pred_label的一部分feature
        padded_sequence_output = self.dropout(padded_sequence_output)
        lstm_output, _ = self.bilstm(padded_sequence_output)
        # 得到判别值
        logits = self.classifier(lstm_output)
        outputs = (logits,)
        if labels is not None:
            loss_mask = labels.gt(-1)
            out = self.crf.decode(emissions=logits,
                                  mask=loss_mask)

            loss = self.crf(logits, labels, loss_mask)
            score = self.crf._compute_score(emissions=logits, tags=labels, mask=loss_mask)
            outputs = (loss,) + outputs

        # contain: (loss), scores
        return outputs
