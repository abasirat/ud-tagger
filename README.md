# UD-Tagger
A multi-task neural network model for multi-tagging of natural languages 

This repository is an implementation of a multi-task neural network model to tag natural languages sentence based on the features of conllu training corpora such as the corpus of Universal Dependencies. Three bidirectional long short-term memory (BiLSTM) models are used to represent a word in a sentence. The BiLSTMs are used to encode:
- the characters forming the word
- the context of the word based on the BiLSTM character representation of words 
- the context of the word based on a set of word embeddings 
The two context representations are concatenated and fed to different activation layers related to multiple tags of the word. 

# Usage
In ```driver.py```, set the paths to the training, validation, and test conllu files. 
In case of using pre-trained word-embeddings set the path to the embeddings too (```emb_file```). Some other parameters such as the maximum number of words for which individual word embeddings will be considered can also be set in this file. 
The list ````tags```` holds the target tags to be predicted. 

# Requirements
- python3
- numpy
- keras

# Performance
The accuracy of the ud-tagger on the development sets of the corpus of universal dependencies. Individual tagging models are trained to predict each tag set.

|corpus|UDTAG|
|------|-----|
|Afrikaans-AfriBooms|0.9364|
|Ancient_Greek-Perseus|0.8496|
|Arabic-PADT|0.9378|
|Basque-BDT|0.8868|
|Belarusian-HSE|0.8109|
|Bulgarian-BTB|0.9577|
|Catalan-AnCora|0.9520|
|Coptic-Scriptorium|0.9112|
|Croatian-SET|0.9296|
|Czech-CLTT|0.9589|
|Danish-DDT|0.9234|
|Dutch-LassySmall|0.9134|
|English-LinES|0.9296|
|Estonian-EDT|0.8966|
|Finnish-TDT|0.8637|
|French-FTB|0.3092|
|Gothic-PROIEL|0.9265|
|Greek-GDT|0.9201|
|Hebrew-HTB|0.9426|
|Hindi_English-HIENCS|0.3127|
|Hungarian-Szeged|0.8062|
|Indonesian-GSD|0.8637|
|Italian-ISDT|0.9546|
|Japanese-GSD|0.9565|
|Korean-Kaist|0.9296|
|Latin-ITTB|0.9597|
|Latvian-LVTB|0.8871|
|Lithuanian-HSE|0.6671|
|Maltese-MUDT|0.9082|
|Marathi-UFAL|0.7178|
|Norwegian-Bokmaal|0.9471|
|Old_Church_Slavonic-PROIEL|0.9298|
|Old_French-SRCMF|0.9318|
|Persian-Seraji|0.9510|
|Polish-LFG|0.9419|
|Portuguese-GSD|0.9427|
|Romanian-Nonstandard|0.9455|
|Russian-SynTagRus|0.9559|
|Serbian-SET|0.9180|
|Slovak-SNK|0.8412|
|Spanish-AnCora|0.9546|
|Swedish-LinES|0.9152|
|Swedish_Sign_Language-SSLC|0.5263|
|Tamil-TTB|0.7649|
|Telugu-MTG|0.8755|
|Turkish-IMST|0.8859|
|Ukrainian-IU|0.9104|
|Urdu-UDTB|0.9182|
|Uyghur-UDT|0.8310|
|Vietnamese-VTB|0.8727|
