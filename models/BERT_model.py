import torch
import os
import random
from transformers import BertTokenizer, BertModel

class GeneralBERTclassifier(torch.nn.Module):
    
    def __init__(self,softmax=False):
        super().__init__() # just call the parent module inits.
        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.dropout = torch.nn.Dropout(0.5) # do we use dropout?
        if softmax:
            # self.classifier = torch.nn.Linear(768,2) # 0=noise, 1 = rumour
            self.ce = True
            self.classifier = torch.nn.Sequential(torch.nn.Linear(768,8),torch.nn.Linear(8,32),torch.nn.Linear(32,2)) # 0=noise, 1 = rumour
            self.activation_fn = torch.nn.Softmax(dim=1)
        else:
            # self.classifier = torch.nn.Linear(768,1) # 0=noise, 1 = rumour
            self.classifier = torch.nn.Sequential(torch.nn.Linear(768,8),torch.nn.Linear(8,32),torch.nn.Linear(32,1)) # 0=noise, 1 = rumour
            self.activation_fn = torch.nn.Sigmoid()
            self.ce = False

    
    def forward(self, inputs,dropout=False,rawdata=""):
        # raw data is never used for this model.
        with torch.no_grad():
            outputs = self.bert(**inputs)
        pooled_output = outputs[1]
        # print(pooled_output.shape)
        if dropout:
            pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        outputs = self.activation_fn(logits)

        return logits, outputs