import csv
import torch
import os
import random
import json
import torch.optim
import train_loops
from ast import literal_eval
from models.BERT_model import GeneralBERTclassifier
from models.SBERT_model import SentenceBERTclassifier
from models.ISBERT_model import IS_BERT
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from dataset_classes import dummy_dataset

    
# Works with all 3!
    
if __name__=="__main__":
    # sbert calculated sentence loss modifier = 0.0001
    output_all_predictions = False
    randomstate = 0
    test_only_last =True
    cut_datapercentage = 0.4
    testpercentage = 0.3
    max_epoch = 1
    batch_size = 8
    start_learn_rate=5e-5
    scheduler_step_size = 3
    momentum = 0.9 # if not Adam.
    scheduler_gamma = 0.9
    pheme_directory = "Pheme_dataset"
    indo_directory = "Indo_dataset"
    bert_softmax_style = True
    decision_threshold = 0.5
    mi_loss_multiplier = 0.1

    loadpath = "SBERT"
    # attempt to infer model type via loadpath.
        
    if "ISBERT" in loadpath:
        modeltype = IS_BERT
        nametype = "ISBERT"
        
    elif "SBERT" in loadpath:
        modeltype = SentenceBERTclassifier
        nametype = "SBERT"
        
    else:
        modeltype = GeneralBERTclassifier
        nametype = "BERT"
        
    if "nosoftmax" in loadpath:
        bert_softmaxstyle = False
    else:   
        bert_softmaxstyle=True
        
        
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    mainpath = os.path.join("resource_creation","all-rnr-annotated-threads")
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

    
    
    # This is the input portion.
    threadtextlist = ["First Statement","Second Statement afterwards","oh no test haha lol"]
    # note how structure doesn't even matter. Just ensure that your entire thread is contained within a list.
    
    
    
    
    
    dummyset = dummy_dataset(threadtextlist, tokenizer, device)
    dummy_dataloader = DataLoader(dummyset, batch_size=batch_size, shuffle=True, num_workers=0)    

    

    
    classifier_model = modeltype(bert_softmax_style).to(device)
    classifier_model.load_state_dict(torch.load(loadpath))
    classifier_model.eval()
    eventwrap = "running_sample"
    if not bert_softmax_style:
        modelname="BCE_"+nametype+"general_nosoftmax_"+eventwrap
    else:
        modelname="CE_"+nametype+"general_nosoftmax_"+eventwrap
    
    currentepoch = 0 
        
    
    with torch.no_grad():
        for _,(tokenized_data,idxlist,raw_data) in enumerate(dummy_dataloader):
            if nametype!="ISBERT":
                logits,outputs = classifier_model(tokenized_data,rawdata=raw_data) # run model
            else:
                logits,outputs,mi_loss = classifier_model(tokenized_data,rawdata=raw_data) # run model

            if classifier_model.ce:
                prediction_output = int(outputs[0].cpu().tolist().index(max(outputs[0])))
            else:
                prediction_output = int(outputs[0].cpu()>threshold)
        
    prediction = "rumour" if prediction_output==0 else "non-rumour"
    print(prediction)
    # 0=noise, 1 = rumour

            
            