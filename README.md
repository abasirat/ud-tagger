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

||single task|
|corpus|UPOS|DEPREL|
|------|-----|-----|
|Afrikaans-AfriBooms|0.9364|0.8340|
|Ancient_Greek-Perseus|0.8496|0.7460|
|Arabic-PADT|0.9378|0.8387|
|Basque-BDT|0.8868|0.7932|
|Belarusian-HSE|0.8109|0.6457|
|Bulgarian-BTB|0.9577|0.8698|
|Catalan-AnCora|0.9520|0.8996|
|Coptic-Scriptorium|0.9112|0.8026|
|Croatian-SET|0.9296|0.8097|
|Czech-CLTT|0.9589|0.8618|
|Danish-DDT|0.9234|0.8342|
|Dutch-LassySmall|0.9134|0.8082|
|English-LinES|0.9296|0.8349|
|Estonian-EDT|0.8966|0.8121|
|Finnish-TDT|0.8637|0.7759|
|French-FTB|0.3092|0.2521|
|Gothic-PROIEL|0.9265|0.7978|
|Greek-GDT|0.9201|0.8565|
|Hebrew-HTB|0.9426|0.8574|
|Hindi_English-HIENCS|0.3127|0.2399|
|Hungarian-Szeged|0.8062|0.7388|
|Indonesian-GSD|0.8637|0.8199|
|Italian-ISDT|0.9546|0.9038|
|Japanese-GSD|0.9565|0.9562|
|Korean-Kaist|0.9296|0.8640|
|Latin-ITTB|0.9597|0.8800|
|Latvian-LVTB|0.8871|0.7858|
|Lithuanian-HSE|0.6671|0.5322|
|Maltese-MUDT|0.9082|0.7998|
|Marathi-UFAL|0.7178|0.6655|
|Norwegian-Bokmaal|0.9471|0.8884|
|Old_Church_Slavonic-PROIEL|0.9298|0.8236|
|Old_French-SRCMF|0.9318|0.8477|
|Persian-Seraji|0.9510|0.8631|
|Polish-LFG|0.9419|0.8896|
|Portuguese-GSD|0.9427|0.9267|
|Romanian-Nonstandard|0.9455|0.8610|
|Russian-SynTagRus|0.9559|0.9018|
|Serbian-SET|0.9180|0.8304|
|Slovak-SNK|0.8412|0.7807|
|Spanish-AnCora|0.9546|0.8934|
|Swedish-LinES|0.9152|0.8205|
|Swedish_Sign_Language-SSLC|0.5263|0.4444|
|Tamil-TTB|0.7649|0.6912|
|Telugu-MTG|0.8755|0.8124|
|Turkish-IMST|0.8859|0.6872|
|Ukrainian-IU|0.9104|0.7835|
|Urdu-UDTB|0.9182|0.8261|
|Uyghur-UDT|0.8310|0.6430|
|Vietnamese-VTB|0.8727|0.6943|
