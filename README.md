# BERT-ISBERT-SBERT_RumourDetection
BERT, ISBERT, SBERT as rumour detection baselines on PHEME dataset. Also on Indonesian Disinfo dataset.  README for details.

# Datasets
1. Pheme Default dataset, available [Here.](https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078)
    - Is the 9 event version, but only uses Rumour/Nonrumour as the actual label, ignoring veracity annotations.
2. Indonesian Disinformation Dataset [Here](https://transparency.twitter.com/en/reports/information-operations.html)
    - Is the dataset released under April 2020, Indonesia (February 2020)
    - Contains ALL TWEETS from the related bot/misinformation/disinformation accounts, regardless of timestamp, from creation until deletion.
    - Is frankly, a terrible dataset to really benchmark with because of a large majority of tweets being unrelated directly to a specific campaign.
    - Lacks pointers on which tweets are related to the campaign itself.
    - Visualisations made available later.

# Baselines
1. Bert - BERT + Linear(768,8) -> Linear(8,32) -> Linear(32,2) (SOFTMAX)
2. Bert - BERT + Linear(768,8) -> Linear(8,32) -> Linear(32,1) (SIGMOID)
    