import torch
from transformers import *

class NeurTxt(torch.nn.Module):
    def __init__(self):
        super(NeurTxt, self).__init__()
        BERT_DIM = 768
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.bert_model.train()
        fcc = torch.nn.Linear(BERT_DIM, 1)

    def forward(self, X, mask):
        '''
        X: tokens tensor, [N x max_tokens]
        Multiple utterance tokens should be concatenated

        mask: [N x max_tokens], with 1 in relevant positions
        and 0s otherwise
        '''
        outputs = self.bert_model(X, mask)
        sentence_embedding = outputs.pooler_output
        y = self.fcc(sentence_embedding)
        return y.squeeze()
