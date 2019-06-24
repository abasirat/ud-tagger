# UD-Tagger
A multi-task neural network model for multi-tagging of natural languages 

This repository is an implementation of a multi-task neural network model to tag natural languages sentence based on the features of conllu training corpora such as the corpus of Universal Dependencies. Three bidirectional long short-term memory (BiLSTM) models are used to represent a word in a sentence. The BiLSTMs are used to encode:
- the characters forming the word
- the context of the word based on the BiLSTM character representation of words 
- the context of the word based on a set of word embeddings 
The two context representations are concatenated and fed to different activation layers related to multiple tags of the word. 

# Usage
In ```driver.py```, set the paths to the training, validation, and test conllu files. In case of using pre-trained word-embeddings set the path to the embeddings too. Some other parameters such as the maximum number of words for which individual word embeddings will be considered can also be set in this file. 

# Requirements
