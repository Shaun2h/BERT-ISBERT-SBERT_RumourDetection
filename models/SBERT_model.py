import csv
import torch
import os
import random
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer

class SentenceBERTclassifier(torch.nn.Module):
    def __init__(self,softmax=False):
        """
        raw sentences as input please.
        """
        super().__init__() # just call the parent module inits.
        self.bert = SentenceTransformer('distiluse-base-multilingual-cased-v2')
        self.dropout = torch.nn.Dropout(0.5) # do we use dropout?
        if softmax:
            # self.classifier = torch.nn.Linear(768,2) # 0=noise, 1 = rumour
            self.ce = True
            self.classifier = torch.nn.Sequential(torch.nn.Linear(512,8),torch.nn.Linear(8,32),torch.nn.Linear(32,2)) # 0=noise, 1 = rumour
            self.activation_fn = torch.nn.Softmax(dim=1)
        else:
            # self.classifier = torch.nn.Linear(768,1) # 0=noise, 1 = rumour
            self.classifier = torch.nn.Sequential(torch.nn.Linear(512,8),torch.nn.Linear(8,32),torch.nn.Linear(32,1)) # 0=noise, 1 = rumour
            self.activation_fn = torch.nn.Sigmoid()
            self.ce = False
        
    
    def forward(self, inputs,dropout=False,rawdata = []):
        # print(inputs)
        
        with torch.no_grad():
            outputs = self.bert.encode(rawdata,convert_to_tensor = True)
        # print(pooled_output.shape)
        if dropout:
            pooled_output = self.dropout(outputs)
        else:
            pooled_output = outputs
        logits = self.classifier(pooled_output)
        outputs = self.activation_fn(logits)
        # print(logits)
        # print(outputs)
        return logits, outputs
        
        