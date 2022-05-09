import csv
import torch
import os
import random
import json
from datetime import datetime
import torch.optim
import train_loops
from ast import literal_eval
from models.BERT_model import GeneralBERTclassifier
from models.SBERT_model import SentenceBERTclassifier
from models.ISBERT_model import IS_BERT
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from dataset_classes import dataset_class_PHEME, indo_dataset_class



def load_pheme_labels():
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
    return combined_data, labels

def load_indo_labels():
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
    
    return indo_data, indo_label, indo_backref_dict, indo_traintestindexes[0][0] + indo_traintestindexes[0][1] 


    
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
    
    target_is_pheme = True # true for pheme, false for indo
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
    if target_is_pheme:
        combined_data, labels = load_pheme_labels()
        mass_dataset = dataset_class_PHEME(combined_data, tokenizer, device, os.path.join(pheme_directory,"phemethreaddump.json"))
        appending_word = "pheme"
    else:        
        indo_data, indo_label, indo_backref_dict ,combined_data = load_indo_labels()
        mass_dataset = indo_dataset_class(combined_data,indo_data,indo_label,indo_backref_dict, tokenizer, device)
        appending_word = "indo"
        
    mass_dataloader = DataLoader(mass_dataset, batch_size=1, shuffle=True, num_workers=0)

    classifier_model = modeltype(bert_softmax_style).to(device)
    # classifier_model.load_state_dict(torch.load(loadpath))
    classifier_model.eval()
    eventwrap = "running_sample"
    if not bert_softmax_style:
        modelname="BCE_"+nametype+"general_nosoftmax_"+eventwrap
    else:
        modelname="CE_"+nametype+"general_nosoftmax_"+eventwrap
            
    
    with open("phemethreaddump.json","rb") as dumpfile:
        loaded_threads = json.load(dumpfile)
    allthreads = []


    with torch.no_grad():
        for _, (tokenized_data, label,idx, raw_data) in enumerate(mass_dataloader):
            threaditem, _ = mass_dataset.backref(idx)
            threadtextlist,tree,rootlabel,source_id = threaditem
            if nametype!="ISBERT":
                logits,outputs = classifier_model(tokenized_data,rawdata=raw_data) # run model
            else:
                logits,outputs,mi_loss = classifier_model(tokenized_data,rawdata=raw_data) # run model

            if classifier_model.ce:
                prediction_output = int(outputs[0].cpu().tolist().index(max(outputs[0])))
                savable_pred = outputs[0].cpu().tolist()
            else:
                prediction_output = int(outputs[0].cpu()>threshold)
                saveable_pred = outputs[0].cpu()
            
    
            prediction = "rumour" if prediction_output==0 else "non-rumour"
            actual_label = "rumour" if label[0]==0 else "non-rumour"
            print([source_id,prediction,actual_label,outputs[0].cpu().tolist()])
            allthreads.append([source_id,prediction,rootlabel[0],outputs[0].cpu().tolist()])

    mainpath = os.path.join("all-rnr-annotated-threads")
    path_reference_dict = {}
    for eventwrap in os.listdir(mainpath):
        if eventwrap[0] == ".":
            continue
        for item in os.listdir(os.path.join(mainpath,eventwrap,"rumours")):
            if item[0]==".":
                continue
            path_reference_dict[item] = os.path.join(mainpath,eventwrap,"rumours",item)
        for item in os.listdir(os.path.join(mainpath,eventwrap,"non-rumours")):
            if item[0]==".":
                continue
            path_reference_dict[item] = os.path.join(mainpath,eventwrap,"non-rumours",item)

    treelist = []
    for i in allthreads:
        treeid = i[0]
        predicted = i[1]
        actual = i[2]
        prediction_value = i[3]
        readable = ['false', 'true', 'unverified']
        tree_path = path_reference_dict[str(treeid)]
        list_of_reactions = os.listdir(os.path.join(tree_path,"reactions"))
        tree_dict = {}
        with open(os.path.join(tree_path,"source-tweets",str(treeid)+".json"),"r",encoding="utf-8") as opened_source:
            loaded_source = json.load(opened_source)
            text = loaded_source["text"]
            source_id = loaded_source["id"]
            links = []
            tree_dict[source_id] = [text,source_id,links,predicted,actual,loaded_source["created_at"],loaded_source["user"]["screen_name"],prediction_value]
            
        for item in list_of_reactions:
            if item[0] == ".":
                continue
            with open(os.path.join(tree_path,"reactions",item),"r",encoding="utf-8") as opened_reaction:
                
                reaction_dict = json.load(opened_reaction)
                reactiontext = reaction_dict["text"]
                reactionid = reaction_dict["id"]
                links = []
                reaction_target = reaction_dict["in_reply_to_status_id"]
                retweetedornot = reaction_dict["retweeted"]
                
                if not reactionid in tree_dict:
                    tree_dict[reactionid] = [reactiontext,reactionid,links,predicted,actual,reaction_dict["created_at"],reaction_dict["user"]["screen_name"],prediction_value]
                else:
                    tree_dict[reactionid] = [reactiontext,reactionid,tree_dict[reactionid][2],predicted,actual,reaction_dict["created_at"],reaction_dict["user"]["screen_name"],prediction_value]
                
                if reaction_target!="null":
                    if not reaction_target in tree_dict:
                        tree_dict[reaction_target] = [None,reaction_target,[[reactionid,reaction_target,"Reply"]],None,None,None,None,None]
                    else:
                        tree_dict[reaction_target][2].append([reactionid,reaction_target,"Reply"])
                    tree_dict[reactionid][2].append([reactionid,reaction_target,"Reply"])
                        
                        
                if retweetedornot:
                    if not reaction_target in tree_dict:
                        tree_dict[reaction_target] = [None,reaction_target,[[reactionid,reaction_target,"Retweet"]],None,None,None,None,None]
                    else:
                        tree_dict[reaction_target][2].append([reactionid,reaction_target,"Retweet"])
                    tree_dict[reactionid][2].append([reactionid,reaction_target,"Retweet"])
                    
        # print("handlin")
        treelist.append(tree_dict)
    
    # 0=noise, 1 = rumour

    # print("dumpin")
    with open(appending_word+"_"+nametype+"_all_predictions_dump.json","w",encoding="utf-8") as treedumpfile:
        # csvwriter = csv.writer(treedumpfile)
        fieldnames = ["tweet_id","handle","text","tweet_type","is_misinformation","tweet_time","edges","actual","scoring"]
        csvwriter = csv.DictWriter(treedumpfile, fieldnames=fieldnames)
        csvwriter.writeheader()
        for treeid in treelist:
            for node in treeid:
                timestampval = str(treeid[node][5])
                if timestampval!="None":
                # "Wed Jan 07 11:11:33 +0000 2015" -> 2012-02-23 09:15:26 +00:00
                    date = datetime.strptime(timestampval,"%a %b %d %H:%M:%S %z %Y").strftime("%Y-%m-%d %H:%M:%S %z")
                else:
                    date = "None"
                csvwriter.writerow({"text":treeid[node][0], "tweet_id":treeid[node][1], "edges":treeid[node][2],"is_misinformation":treeid[node][3],"actual":treeid[node][4],"tweet_time":treeid[node][5],"handle":treeid[node][6],"scoring":treeid[node][7]})
    print("Dumped:",appending_word+"_"+nametype+"_all_predictions_dump.json")            