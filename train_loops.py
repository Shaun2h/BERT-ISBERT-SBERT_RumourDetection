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
        for _,(tokenized_data,label,idlist,raw_data) in enumerate(dataloader):

            logits,outputs = model(tokenized_data,dropout=backprop,rawdata=raw_data) # run model
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
                    
                # print(prediction_output,outputs[i],targetlabel)
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
        if backprop:
            insertion="_trainpart_"
        else:
            insertion="_testpart_"
        with open(name+str(epoch)+insertion+"_epoch.json","w",encoding="utf-8") as jsondumpfile:
            json.dump(pairings,jsondumpfile,indent=4)
        
    return totalsubjects,totalloss,correctcount
            
            
            



def baseline_ISBERT_loop(model,dataloader,batch_size=12,device="cpu",loss_fn=None,name="",epoch=0,backprop=False,threshold=0.75,optimizer=None):
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
        for _,(tokenized_data,label,idlist,raw_data) in enumerate(dataloader):

            logits,outputs,mi_loss = model(tokenized_data,dropout=backprop,rawdata=raw_data) # run model
            
            
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
                    
                # print(prediction_output,outputs[i],targetlabel)
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
                    mi_loss = mi_loss * model.mi_loss_multiplier
                    final_loss = final_loss+mi_loss
                    final_loss.backward()
                    optimizer.step()
                totalloss+=mi_loss*model.mi_loss_multiplier

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
        if backprop:
            insertion="_trainpart_"
        else:
            insertion="_testpart_"
        with open(name+str(epoch)+insertion+"_epoch.json","w",encoding="utf-8") as jsondumpfile:
            json.dump(pairings,jsondumpfile,indent=4)
        
    return totalsubjects,totalloss,correctcount


















def doloop(model,dataloader,loss_fn,activation_fn,model_optimizer,is_ce = False,batch_size=12,device="cpu",tokenizer=None,mutualinfo=False,linearcomparatormodel=None,linearoptimiser=None,name="",epoch=0):
    model.train()
    totalsubjects = len(dataloader)*batch_size
    correctcount =0
    disinfocorrect = 0 
    total_metaloss = 0
    totalloss = 0
    total_comparisonsubjects=0
    disinfoloss_total = 0
    pairings = []
    rngsample = True

    # if mutualinfo:
        # MI_trainloss = MutualInformationLoss(model=model.bert, sentence_embedding_dimension=model.bert.get_sentence_embedding_dimension())

    
    for idx,(data,label) in enumerate(dataloader):
        model_optimizer.zero_grad()
        if not tokenizer is None:
            tokenized_data = tokenizer(data, return_tensors="pt",padding=True,truncation=True)
            tokenized_data.to(device)
        else:
            tokenized_data = data # just raw data eh?
        logits = model(tokenized_data) # run model
        if mutualinfo:
            linearoptimiser.zero_grad()
            loss_summation = 0
            losslist = []
            disinfolosslist = []                    
            
            
            for label_idx in range(len(label)):
                nextidx = label_idx+1
                if label_idx==len(label)-1:
                    continue # no one to compare to...
                basetarget = logits[label_idx]
                baselabel = label[label_idx]
                basetarget_dropped_output_token, basetarget_dropped_output_sentence, basetarget_conv1out_token, basetarget_conv2out_token, basetarget_conv3out_token = basetarget
                relevants = logits[nextidx:]
                relevantlabels = label[nextidx:]
                for relevantidx in range(len(relevants)):
                    dropped_output_token, dropped_output_sentence, conv1out_token, conv2out_token, conv3out_token = relevants[relevantidx]
                    if baselabel==relevantlabels[relevantidx]:
                        targetlabel = torch.tensor(1).to(device)
                    else:
                        targetlabel = torch.tensor(0).to(device)
                    try:
                        for conv_piece in range(len(conv1out_token)):
                            # print(conv1out_token.shape)
                            # print(conv2out_token.shape)
                            # print(conv3out_token.shape)
                            cat_convs = torch.cat([conv1out_token[conv_piece],conv2out_token[conv_piece],conv3out_token[conv_piece]])
                            base_convs = torch.cat([basetarget_conv1out_token[conv_piece],basetarget_conv2out_token[conv_piece],basetarget_conv3out_token[conv_piece]])
                            # print(cat_convs.shape)
                            resultantprediction,_ = linearcomparatormodel(basetarget_dropped_output_sentence, cat_convs,base_convs)
                            
                            
                            # MI_trainloss(logits[])

                            
                            
                            if is_ce:
                                activated_prediction = activation_fn(resultantprediction.unsqueeze(0))
                                calculated_sampleloss = loss_fn(activated_prediction.reshape(1,-1),targetlabel.unsqueeze(0))
                                if torch.argmax(activated_prediction) ==targetlabel:
                                    correctcount+=1
                                
                            else:
                                activated_prediction,_ = activation_func_bce(resultantprediction[0])
                                calculated_sampleloss = loss_fn(activated_prediction.unsqueeze(0), targetlabel.unsqueeze(0).float())
                                if activated_prediction == targetlabel:
                                    correctcount+=1
                            losslist.append(calculated_sampleloss*0.0001) # naturally, about 400k mutual info is to 3000 samples of disinfo.
                            total_comparisonsubjects+=1
                    except IndexError:
                        continue
                _, matchy_prediction = linearcomparatormodel(basetarget_dropped_output_sentence,cat_convs)
                disinfo_targetlabel = torch.tensor([baselabel]).to(device)
                # print("unactivated prediction:",matchy_prediction)
                if is_ce:

                    disinfo_sampleloss = loss_fn(matchy_prediction.reshape(1,-1),torch.tensor([baselabel]).to(device))
                    activated_prediction = activation_fn(matchy_prediction)
                    # print("targetlabel:",disinfo_targetlabel)
                    # print("prediction:",activated_prediction)
                    if torch.argmax(activated_prediction) ==torch.tensor([baselabel]).to(device):
                        disinfocorrect+=1
                    prediction_output = int(torch.argmax(activated_prediction))
                    
                else:
                    activated_prediction,_ = activation_func_bce(matchy_prediction[0])
                    disinfo_sampleloss = loss_fn(activated_prediction.unsqueeze(0), torch.tensor([baselabel]).to(device))
                    if activated_prediction == torch.tensor([baselabel]).to(device):
                        disinfocorrect+=1
                    prediction_output = int(activated_prediction)
                # refid = dataloader.dataset.backref[data[label_idx]]
                # pairings.append([refid, prediction_output, int(baselabel)])
                disinfolosslist.append(disinfo_sampleloss)
                # print("sampleloss:", disinfo_sampleloss)
                disinfo_sampleloss.backward()
             


             
        else: # not mutual information
            outputs = activation_fn(logits).cpu()
            # print(outputs)
            losslist = []
            for i in range(len(label)):
                if is_ce:
                    calculated_sampleloss = loss_fn(logits[i].reshape(1,len(logits[i])),torch.tensor([label[i]]).to(device))
                    # print(outputs[i],label[i],i)
                    if torch.argmax(outputs[i])==label[i]:
                        correctcount+=1
                else:
                    outputs = activation_fn(logits)
                    calculated_sampleloss = loss_fn(outputs[i],torch.tensor([label[i]]).float().to(device))
                    if outputs[i]>0.75 and label[i]==1:
                        correctcount+=1
                losslist.append(calculated_sampleloss)
        final_loss = torch.mean(torch.stack(losslist))
        totalloss+=torch.sum(torch.stack(losslist)).item()                
        final_loss.backward()
        
        # for name, param in linearcomparatormodel.named_parameters():
            # print(name, param.grad)
        # quit()
        if mutualinfo:
            linearoptimiser.step()
        model_optimizer.step() # forward the normal model.
        
        if mutualinfo:
            disinfoloss_total+=torch.sum(torch.stack(disinfolosslist)).item()
            # torch.mean(torch.stack(disinfolosslist)).backward()
            # final_loss += torch.mean(torch.stack(disinfolosslist))
            # if rngsample:
                # print(final_loss)
                # rngsample=False
            
        
        # if random.randint(0,1000)>750:
            # print("Random sample loss:",final_loss)
            # print("Random Label sample:",label)
            # print("random correctcount:",correctcount,"/",batch_size)
    if mutualinfo:
        print("Percentage Correct:", (correctcount/total_comparisonsubjects*100),"%")
        print("Total MI Loss:",totalloss)
        print("Average MI loss:",totalloss/total_comparisonsubjects)
        print("Disinfo Correct:",disinfocorrect)

        print("Total disinfo loss:",disinfoloss_total)
        print("Average disinfo loss:",disinfoloss_total/totalsubjects)
        correct_disinfo_percentage  = disinfocorrect/totalsubjects*100
        print("Average disinfo correct percentage",correct_disinfo_percentage,"%")
        report = ["Percentage Correct:", (correctcount/total_comparisonsubjects*100),"%","\n",
        "Total MI Loss:",totalloss,"\n",
        "Average MI loss:",totalloss/total_comparisonsubjects,"\n",
        "Disinfo Correct:",disinfocorrect,"\n",
        "Total disinfo loss:",disinfoloss_total,"\n",
        "Average disinfo loss:",disinfoloss_total/totalsubjects,"\n",
        "Average disinfo correct percentage",correct_disinfo_percentage,"%"]
        
        # pairings.insert(0,report)
        # with open(name+str(epoch)+"_epoch.json","w",encoding="utf-8") as jsondumpfile:
            # json.dump(pairings,jsondumpfile,indent=4)
        
        return totalsubjects,totalloss,correctcount,disinfoloss_total,total_comparisonsubjects, correct_disinfo_percentage

    else:
        print("Percentage Correct:", (correctcount/totalsubjects*100),"%")
        print("Total loss:",totalloss)
        print("Average loss:",totalloss/totalsubjects)
        report = ["Percentage Correct:", (correctcount/totalsubjects*100),"%","\n",
        "Total loss:",totalloss,"\n",
        "Average loss:",totalloss/totalsubjects]
        pairings.insert(0,report)

        with open(name+str(epoch)+"_epoch.json","w",encoding="utf-8") as jsondumpfile:
            json.dump(pairings,jsondumpfile,indent=4)
        
        return correctcount,totalsubjects,totalloss
                
            

