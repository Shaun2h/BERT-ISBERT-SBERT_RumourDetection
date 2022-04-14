import csv
import torch
import os
import random
import json
from torch.utils.data import Dataset, DataLoader



class indo_dataset_class(Dataset):
    def __init__(self, idxes, data,label,backref_dict,tokeniser,device):
        self.idxes = idxes
        self.data = data
        self.label = label
        self.backref_dict = backref_dict
        self.device = device
        self.tokenizer = tokeniser
        
    def __len__(self):
        return len(self.idxes)

    def __getitem__(self, idx):
        data = self.data[self.idxes[idx]]
        label = self.label[self.idxes[idx]]
        textlist = [data]
        with torch.no_grad():
            tokenized_data = self.tokenizer(textlist, return_tensors="pt", padding="max_length", truncation=True)
            tokenized_data["input_ids"] = tokenized_data["input_ids"].squeeze()
            tokenized_data['token_type_ids'] = tokenized_data['token_type_ids'].squeeze()
            tokenized_data['attention_mask'] = tokenized_data['attention_mask'].squeeze()
        tokenized_data.to(self.device)
        return tokenized_data, label,idx, data
    
        
    def backref(self,targetidx):
        return self.data[self.idxes[int(targetidx)]], self.backref_dict[self.data[self.idxes[int(targetidx)]]][0]


    
    
class dataset_class_PHEME(Dataset):
    def __init__(self, data,tokeniser,device,targetdumpfile):
        # 0=noise, 1 = rumour
        with open(targetdumpfile,"rb") as dumpfile:
            loaded_threads = json.load(dumpfile)
        self.tokenizer = tokeniser
        self.allthreads = {}
        self.rootitems = []
        self.device = device
        for thread in loaded_threads:
            threadtextlist,tree,rootlabel,source_id = thread
            if str(source_id) in data:
                self.allthreads[source_id] = thread
                self.rootitems.append(source_id)
        
    def __len__(self):
        return len(self.rootitems)

    def __getitem__(self, idx):
        targettree = self.rootitems[idx]
        thread = self.allthreads[targettree]
        threadtextlist,tree,rootlabel,source_id = thread
        textlist = ""
        for item in threadtextlist:
            textlist = textlist+ item[0]+" [SEP]"
        textlist = textlist[:-6] # remove last sep.
        with torch.no_grad():
            tokenized_data = self.tokenizer(textlist, return_tensors="pt", padding="max_length", truncation=True)
            tokenized_data["input_ids"] = tokenized_data["input_ids"].squeeze()
            tokenized_data['token_type_ids'] = tokenized_data['token_type_ids'].squeeze()
            tokenized_data['attention_mask'] = tokenized_data['attention_mask'].squeeze()
        tokenized_data.to(self.device)
        return tokenized_data, rootlabel[0],idx, " ".join(textlist)
        
    def backref(self,idx):
        return self.allthreads[self.rootitems[idx]], self.rootitems[idx]
        
        