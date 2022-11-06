
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
from torch.nn import CrossEntropyLoss
from torch import nn
import torch.nn.functional as F
SENTENCE_LEN = 384

class ModelA(nn.Module):
    # Plain conv net
    # TODO: Add QANet Attention mechanism in separate model
    def __init__(self):
        super(ModelA, self).__init__()
        # BERT Model in eval mode, not training, hence bert weights are frozen and remains unchanged during training of our model
        self.default_config = AutoConfig.from_pretrained('bert-base-uncased', output_hidden_states=False)
        self.bert_model = AutoModel.from_pretrained("bert-base-uncased", config=self.default_config)
        self.dropout = nn.Dropout(self.default_config.hidden_dropout_prob)
        d_word = SENTENCE_LEN
        self.hidden1 = nn.Linear(self.default_config.hidden_size, self.default_config.hidden_size)
        self.hidden2 = nn.Linear(self.default_config.hidden_size, self.default_config.hidden_size)
        self.hidden3 = nn.Linear(self.default_config.hidden_size, self.default_config.hidden_size)
        self.batchnorm = nn.BatchNorm1d(d_word)
        self.qa_outputs = nn.Linear(self.default_config.hidden_size, 2)

    def forward(self, encoded_input, start_positions=None, end_positions=None):
        sequence_output = self.bert_model(**encoded_input)[0]
        i1 = self.hidden2(self.batchnorm(self.hidden1(sequence_output)))
        logits = self.qa_outputs(i1)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # ignore terms when start and end is out of context range
            ignored_index = start_logits.size(1)
            start_positions = torch.clamp(start_positions, min=0, max=ignored_index)
            end_positions = torch.clamp(end_positions, min=0, max=ignored_index)
            loss_func = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_func(start_logits, start_positions)
            end_loss = loss_func(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        else:
            return start_logits, end_logits