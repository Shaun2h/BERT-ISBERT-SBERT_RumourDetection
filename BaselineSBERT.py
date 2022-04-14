import csv
import torch
import os
import random
import json
import torch.optim
import train_loops
from ast import literal_eval
from models.SBERT_model import SentenceBERTclassifier
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from dataset_classes import indo_dataset_class, dataset_class_PHEME

    
if __name__=="__main__":
    # sbert calculated sentence loss modifier = 0.0001
    output_all_predictions = False
    randomstate = 0
    test_only_last =True
    cut_datapercentage = 0.4
    testpercentage = 0.3
    max_epoch = 30
    batch_size = 8
    start_learn_rate=5e-5
    scheduler_step_size = 3
    momentum = 0.9 # if not Adam.
    scheduler_gamma = 0.9
    pheme_directory = "Pheme_dataset"
    indo_directory = "Indo_dataset"
    bert_softmax_style = True
    decision_threshold = 0.5
    """
    Batch size: 16, 32
    Learning rate (Adam): 5e-5, 3e-5, 2e-5
    Number of epochs: 2, 3, 4
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    mainpath = os.path.join("resource_creation","all-rnr-annotated-threads")
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

    if not os.path.exists("pheme_traintest_splits.json"):
        combined_data = []
        labels = []
        with open(os.path.join(pheme_directory,"PHEME_labelsplits.txt"),"r",encoding="utf-8") as labelfile:
            for line in labelfile:
                if line:
                    root = line.split()[0]
                    labellist = literal_eval(" ".join(line.split()[1:]))
                    label = labellist[0]    
                    combined_data.append(root)
                    labels.append(label)
        
        
        sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=randomstate) # save random state for reproducibility
        traintestindexes = []
        for train_index, test_index in sss.split(combined_data, labels):    
            train = []
            test = []
            for trainsubject in train_index:
                train.append(combined_data[trainsubject])
            for testsubject in test_index:
                test.append(combined_data[testsubject])
            traintestindexes.append([train,test])
            
        with open(os.path.join(pheme_directory,"pheme_traintest_splits.json"),"w",encoding="utf-8") as dumpfile:
            json.dump(traintestindexes,dumpfile)
    else:
        with open(os.path.join(pheme_directory,"pheme_traintest_splits.json"),"r",encoding="utf-8") as dumpfile:
            traintestindexes = json.load(dumpfile)
    
    with open(os.path.join(pheme_directory,"Eventsplit_details.txt"),"r",encoding="utf-8") as dumpfile:
        eventrefdict = json.load(dumpfile)
    
    
    
    ##################################################################################################################################
    indo_data = []
    indo_label = []
    indo_backref_dict = {}
    with open(os.path.join(indo_directory,"daniel_subsettweets.csv"),"r",encoding="utf-8") as disinfofile:
        disinfo_csv_reader = csv.DictReader(disinfofile)
        rowcount=0
        for row in disinfo_csv_reader:
            # "tweetid,userid,user_display_name,user_screen_name,user_reported_location,user_profile_description,user_profile_url,follower_count,following_count,account_creation_date,account_language,tweet_language,tweet_text,tweet_time,tweet_client_name,in_reply_to_userid,in_reply_to_tweetid,quoted_tweet_tweetid,is_retweet,retweet_userid,retweet_tweetid,latitude,longitude,quote_count,reply_count,like_count,retweet_count,hashtags,urls,user_mentions,poll_choices"
            tweetid = row["tweetid"]
            tweet_text = row["tweet_text"]
            userid = row["userid"]
            indo_data.append(tweet_text)
            indo_label.append(0)
            indo_backref_dict[tweet_text] = [tweetid,userid,0]
            
    with open(os.path.join(indo_directory,"noisetweets.csv"),"r",encoding="utf-8") as noisefile:
        nosie_csv_reader = csv.DictReader(noisefile)
        rowcount=0
        for row in nosie_csv_reader:
            # "tweetid,userid,user_display_name,user_screen_name,user_reported_location,user_profile_description,user_profile_url,follower_count,following_count,account_creation_date,account_language,tweet_language,tweet_text,tweet_time,tweet_client_name,in_reply_to_userid,in_reply_to_tweetid,quoted_tweet_tweetid,is_retweet,retweet_userid,retweet_tweetid,latitude,longitude,quote_count,reply_count,like_count,retweet_count,hashtags,urls,user_mentions,poll_choices"
            tweetid = row["tweetid"]
            tweet_text = row["tweet_text"]
            userid = row["userid"]
            indo_data.append(tweet_text)
            indo_label.append(1)
            indo_backref_dict[tweet_text] = [tweetid,userid,1]    


    if not os.path.exists(os.path.join(indo_directory,"indo_traintest_splits.json")):
        indo_traintestindexes = []
        sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
        splitter = StratifiedShuffleSplit(n_splits=5,random_state=0) # save random state for reproducibility
        for train_index, test_index in sss.split(indo_data, indo_label):    
            indo_traintestindexes.append([train_index.tolist(),test_index.tolist(),"INDO"])
        with open(os.path.join(indo_directory,"indo_traintest_splits.json"),"w",encoding="utf-8") as dumpfile:
            json.dump(indo_traintestindexes,dumpfile)
    else:
        with open(os.path.join(indo_directory,"indo_traintest_splits.json"),"r",encoding="utf-8") as dumpfile:
            indo_traintestindexes = json.load(dumpfile)
    
    ##################################################################################################################################
    
    
    
    
    
    
    for eventwrap in indo_traintestindexes+traintestindexes+list(eventrefdict.keys()): # isolate a single event as the TEST case, or run it as a split.
        
        if type(eventwrap)==str: #withheld one event as test.
            train = []
            for internal_event in eventrefdict:
                if internal_event!=eventwrap:
                    train.extend(eventrefdict[internal_event])
                else:
                    test = eventrefdict[eventwrap]
        
            train_dataset = dataset_class_PHEME(train, tokenizer, device, os.path.join(pheme_directory,"phemethreaddump.json"))
            test_dataset = dataset_class_PHEME(test, tokenizer, device, os.path.join(pheme_directory,"phemethreaddump.json"))
            print("Done loading Datasets")
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        elif len(eventwrap)==3: # it's the indonesian dataset split.
            train = eventwrap[0]
            test = eventwrap[1]
            train_dataset = indo_dataset_class(train,indo_data,indo_label,indo_backref_dict, tokenizer, device)
            test_dataset = indo_dataset_class(test,indo_data,indo_label,indo_backref_dict, tokenizer, device)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)    
            eventwrap = "_indosplit_"+str(indo_traintestindexes.index(eventwrap))
        
        else: # it's a pheme Split.
            train = eventwrap[0]
            test = eventwrap[1]
            train_dataset = dataset_class_PHEME(train, tokenizer, device, os.path.join(pheme_directory,"phemethreaddump.json"))
            test_dataset = dataset_class_PHEME(test, tokenizer, device, os.path.join(pheme_directory,"phemethreaddump.json"))
            
            print("Done loading Datasets")
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)    
            eventwrap = "_split_"+str(traintestindexes.index(eventwrap))
            
            if output_all_predictions:
                combined_dataset = dataset_class_PHEME(train+test)
                combined_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
                        
                    
        classifier_model = SentenceBERTclassifier(softmax=True).to(device)
        if not bert_softmax_style:
            loss_fn = torch.nn.BCELoss().to(device)
            modelname="BCE_SBERTgeneral_nosoftmax_"+eventwrap
        else:
            loss_fn = torch.nn.CrossEntropyLoss().to(device)
            modelname="CE_SBERTgeneral_nosoftmax_"+eventwrap
            
        classifier_optimizer = torch.optim.Adam(classifier_model.parameters(),lr=start_learn_rate)
        classifier_scheduler = torch.optim.lr_scheduler.StepLR(classifier_optimizer, scheduler_step_size, gamma=scheduler_gamma)
                
        train_correctcount = []
        train_totalloss = []
        train_totalsubjects = []
        test_correctcount = []
        test_totalloss = []
        test_totalsubjects = []
        train_percentagecorrect = []
        test_percentagecorrect = []

        

        modelname="SBERT_"+eventwrap

        for currentepoch in range(max_epoch):
            print("-"*20)
            print("SBERT Train:", currentepoch)
            correctcount,totalsubjects,totalloss = train_loops.baseline_bert_loop(classifier_model, train_dataloader,loss_fn = loss_fn, optimizer=classifier_optimizer,batch_size=batch_size,device=device,name=modelname,epoch=currentepoch,backprop=True,threshold=decision_threshold)

            
            train_correctcount.append(correctcount)
            train_totalloss.append(totalloss)
            train_totalsubjects.append(totalsubjects)
            train_percentagecorrect.append(correctcount/totalsubjects*100)
            print("Test")
            totalsubjects,totalloss,correctcount = train_loops.baseline_bert_loop(classifier_model,test_dataloader,batch_size=batch_size,device=device,loss_fn=loss_fn,name=modelname,epoch=currentepoch,threshold=decision_threshold)
            test_correctcount.append(correctcount)
            test_totalloss.append(totalloss)
            test_totalsubjects.append(totalsubjects)
            test_percentagecorrect.append(correctcount/totalsubjects*100)
            classifier_scheduler.step()
            
            
            torch.save(classifier_model.state_dict(),modelname+str(currentepoch)+str(eventwrap)+".torch")
            with open("training_details_"+modelname+"_"+eventwrap+".txt","w",encoding="utf-8") as trainingout:
                json.dump(
                {"BCE":{"Train": {"Train Correct Count":train_correctcount, "Train Total Loss": train_totalloss, "Total Subjects": train_totalsubjects, "Percentage correct":train_percentagecorrect} ,
                "Test":{"Test Correct Count":test_correctcount, "Test Total Loss":test_totalloss, "Test Total Subjects":test_totalsubjects,"Percentage correct":test_percentagecorrect}},
                "cut_datapercentage":cut_datapercentage, "testpercentage":testpercentage, "max_epoch":max_epoch, "batch_size":batch_size, "start_learn_rate":start_learn_rate, "scheduler_step_size":scheduler_step_size, "momentum":momentum}
                , trainingout,indent=4)
