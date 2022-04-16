# BERT-ISBERT-SBERT_RumourDetection
BERT, ISBERT, SBERT as rumour detection baselines on PHEME dataset. Also on Indonesian Disinfo dataset.  README for details.

# ISBERT Source
This repository uses files from ISBERT's original repository, with edits to ensure they work.

- Repository here: [https://github.com/yanzhangnlp/IS-BERT/tree/ea6622c81a2d732798d974db74cc83f5398ac4d5](https://github.com/yanzhangnlp/IS-BERT/tree/ea6622c81a2d732798d974db74cc83f5398ac4d5)
- Files used:
    1. [MutualInformationLoss.py](https://github.com/yanzhangnlp/IS-BERT/blob/ea6622c81a2d732798d974db74cc83f5398ac4d5/sentence_transformers/losses/MutualInformationLoss.py)
    2. [CNN](https://github.com/yanzhangnlp/IS-BERT/blob/ea6622c81a2d732798d974db74cc83f5398ac4d5/sentence_transformers/models/CNN.py)
    3. [Pooling](https://github.com/yanzhangnlp/IS-BERT/blob/ea6622c81a2d732798d974db74cc83f5398ac4d5/sentence_transformers/models/Pooling.py)
- As is the custom, they have been edited to be compatible with a more recent huggingface version/sentence-transformers version.
- transformers - 4.12.3
- torch - 1.10.0
- sentence-transformers - 2.1.0
- Requirements.txt will contain most of the required statements, but pytorch might require searching in previous versions list accordingly.




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
    