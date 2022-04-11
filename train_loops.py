import csv
import torch
import os
import random
import json
import torch.optim
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from dataset_classes import indo_dataset_class, dataset_class_PHEME




def baseline_bert_loop(model,dataloader,batch_size=12,device="cpu",loss_fn=None,name="",epoch=0,backprop=False,threshold=0.75,optimizer=None):
    model.eval()
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    totalsubjects = len(dataloader)*batch_size
    correctcount = 0 
    totalloss = 0
    total_comparisonsubjects = 0
    disinfocorrect = 0
    pairings = []

    class nothing: # do... nothing with a with statement.
        def __init__(self):
            pass
        def __enter__(self):
            pass
        def __exit__(self, exception_type, exception_value, traceback):
            pass
    
    if backprop:
        grad_decider = nothing
        # print("Using Grad")
    else:
        grad_decider = torch.no_grad
        # print("No Grad.")
    
    
    
    with grad_decider():
        for _,(tokenized_data,label,idlist) in enumerate(dataloader):

            logits,outputs = model(tokenized_data,dropout=backprop) # run model
            disinfolosslist = []
            losslist = []
            for i in range(len(label)):
                targetlabel = label[i]
                if model.ce:
                    calculated_sampleloss = loss_fn(logits[i].reshape(1,len(logits[i])),torch.tensor([targetlabel]).to(device))
                    prediction_output = int(outputs[i].cpu().tolist().index(max(outputs[i])))
                else:
                    calculated_sampleloss = loss_fn(outputs[i],torch.tensor([targetlabel]).float().to(device))
                    prediction_output = int(outputs[i].cpu()>threshold)
                    
                    
                if prediction_output == targetlabel:
                    correctcount+=1
                    if targetlabel==1:  # 0=noise, 1 = rumour
                        TP+=1
                    else:
                        TN+=1
                else:
                    if prediction_output==0:
                        FP+=1
                    else:
                        FN+=1
                
                losslist.append(calculated_sampleloss)
                thread, root_tweet_id = dataloader.dataset.backref(idlist[i]) 
                pairings.append([root_tweet_id, prediction_output, int(label[i])])
                
            try: # if the entire thing is an empty list somehow from pytorch, catch error and continue
                final_loss = torch.sum(torch.stack(losslist))
                totalloss+=final_loss.item()
                if backprop:
                    final_loss.backward()
                    optimizer.step()
            except Exception as e: 
                print(e)
                pass


        
        print("Percentage Correct:", (correctcount/totalsubjects*100),"%")
        print("Total loss:",totalloss)
        print("Average loss:",totalloss/totalsubjects)
        print("TP:",TP)
        print("FP:",FP)
        print("FN:",FN)
        print("TN:",TN)
        try:
            precision = TP/(TP+FP)
            recall = TP/(TP+FN)
        except ZeroDivisionError:
            precision = "Error"
            recall = "Error"
        print("Precision:",precision)
        print("Recall:",recall)
        report = ["Percentage Correct:", (correctcount/totalsubjects*100),"%","\n",
        "Total loss:",totalloss,"\n",
        "Average loss:",totalloss/totalsubjects,
        "TP:",TP,"\n",
        "FP:",FP,"\n",
        "FN:",FN,"\n",
        "TN:",TN,"\n",
        "Precision",precision,"\n",
        "Recall",recall,"\n",]
        
        pairings.insert(0,report)

        with open(name+str(epoch)+"_epoch.json","w",encoding="utf-8") as jsondumpfile:
            json.dump(pairings,jsondumpfile,indent=4)
        
    return totalsubjects,totalloss,correctcount
            
        