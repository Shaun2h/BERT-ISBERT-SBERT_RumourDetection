import csv
import torch
import os
import random
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from .CNN_sentence_transformers_cleaned import CNN
from .pooling_sentence_transformers_cleaned import Pooling
from .sentence_transformers_mutual_info_loss import MutualInformationLoss
# Original ISBERT code.
# Essentially, one Transformers Bert model for tokenisation,
# one CNN module that they wrote
# one pooling
# word_embedding_model = models.Transformer(model_name)
# cnn = models.CNN(in_word_embedding_dimension=word_embedding_model.get_word_embedding_dimension())
# pooling_model = models.Pooling(cnn.get_word_embedding_dimension(), pooling_mode_mean_tokens=True, pooling_mode_cls_token=False, pooling_mode_max_tokens=False)
# model = SentenceTransformer(modules=[word_embedding_model, cnn, pooling_model])


class IS_BERT(torch.nn.Module):
    
    def __init__(self,softmax=False, mi_loss_multiplier=0.1):
        """
        raw sentences as input please.
        """
        super().__init__() # just call the parent module inits.

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
        
        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.cnn = CNN(in_word_embedding_dimension=768)#hardcoded.
        self.pooling = Pooling(self.cnn.get_word_embedding_dimension(), pooling_mode_mean_tokens=True, pooling_mode_cls_token=False, pooling_mode_max_tokens=False)
        self.dropout = torch.nn.Dropout(0.5) # do we use dropout?
        self.loss = MutualInformationLoss(model=None, sentence_embedding_dimension=768)
        self.mi_loss_multiplier = mi_loss_multiplier
        
        
        
    
    def forward(self, inputdata,dropout=False,rawdata=[]):
        logitslist = []
        mi_loss = []
        outputs_bert = self.bert(inputdata["input_ids"]) # you want to GRAD this, else there's no shared space between MI and normal task.
        outputs_cnn = self.cnn(outputs_bert,inputdata) # note that we must break it apart here to reshape outputs to their needs.
        outputs_pooling = self.pooling(outputs_cnn)
        mi_loss = self.loss(outputs_pooling)  # ???  it doesn't use the original "labels" argument at all within the code... Labels was removed.
        
        if dropout:
            dropped_bert_embeds = self.dropout(outputs_bert["last_hidden_state"][:,0,:]).squeeze()
        else:
            dropped_bert_embeds = outputs_bert["last_hidden_state"][:,0,:].squeeze()
        logits = self.classifier(dropped_bert_embeds)
        outputs = self.activation_fn(logits)
        return logits, outputs, mi_loss
        