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

|corpus|I-UPOS|I-DEPREL|J-UPOS|J-DEPREL|D-UPOS|D-DEPREL|
|------|-----|-----|-----|-----|-----|-----|
|Afrikaans-AfriBooms|93.64|83.40|93.87|84.60|0.23|1.20|
|Ancient_Greek-Perseus|84.96|74.60|86.60|74.76|1.64|0.16|
|Arabic-PADT|93.78|83.87|94.12|84.03|0.34|0.16|
|Basque-BDT|88.68|79.32|90.01|79.59|1.33|0.27|
|Belarusian-HSE|81.09|64.57|80.28|65.84|-0.81|1.27|
|Bulgarian-BTB|95.77|86.98|95.66|87.88|-0.11|0.90|
|Catalan-AnCora|95.20|89.96|95.46|90.32|0.26|0.36|
|Coptic-Scriptorium|91.12|80.26|91.50|81.17|0.38|0.91|
|Croatian-SET|92.96|80.97|93.15|81.68|0.19|0.71|
|Czech-CLTT|95.89|86.18|95.63|86.26|-0.26|0.08|
|Danish-DDT|92.34|83.42|92.62|83.48|0.28|0.06|
|Dutch-LassySmall|91.34|80.82|92.16|81.88|0.82|1.06|
|English-LinES|92.96|83.49|93.49|84.62|0.53|1.13|
|Estonian-EDT|89.66|81.21|90.66|81.94|1.00|0.73|
|Finnish-TDT|86.37|77.59|87.61|77.82|1.24|0.23|
|French-FTB|30.92|25.21|0.00|0.00|-30.92|-25.21|
|Gothic-PROIEL|92.65|79.78|93.08|80.46|0.43|0.68|
|Greek-GDT|92.01|85.65|92.35|86.53|0.34|0.88|
|Hebrew-HTB|94.26|85.74|94.64|86.90|0.38|1.16|
|Hindi_English-HIENCS|31.27|23.99|32.01|22.89|0.74|-1.10|
|Hungarian-Szeged|80.62|73.88|83.10|74.65|2.48|0.77|
|Indonesian-GSD|86.37|81.99|88.17|82.02|1.80|0.03|
|Italian-ISDT|95.46|90.38|95.87|90.98|0.41|0.60|
|Japanese-GSD|95.65|95.62|96.55|96.28|0.90|0.66|
|Korean-Kaist|92.96|86.40|93.09|87.15|0.13|0.75|
|Latin-ITTB|95.97|88.00|96.29|88.34|0.32|0.34|
|Latvian-LVTB|88.71|78.58|89.44|79.34|0.73|0.76|
|Lithuanian-HSE|66.71|53.22|58.75|48.97|-7.96|-4.25|
|Maltese-MUDT|90.82|79.98|90.43|79.77|-0.39|-0.21|
|Marathi-UFAL|71.78|66.55|72.88|65.99|1.10|-0.56|
|Norwegian-Bokmaal|94.71|88.84|95.37|89.18|0.66|0.34|
|Old_Church_Slavonic-PROIEL|92.98|82.36|93.29|82.69|0.31|0.33|
|Old_French-SRCMF|93.18|84.77|93.41|85.46|0.23|0.69|
|Persian-Seraji|95.10|86.31|95.76|86.42|0.66|0.11|
|Polish-LFG|94.19|88.96|94.35|89.30|0.16|0.34|
|Portuguese-GSD|94.27|92.67|94.49|93.16|0.22|0.49|
|Romanian-Nonstandard|94.55|86.10|95.24|86.30|0.69|0.20|
|Russian-SynTagRus|95.59|90.18|95.93|90.36|0.34|0.18|
|Serbian-SET|91.80|83.04|93.13|82.99|1.33|-0.05|
|Slovak-SNK|84.12|78.07|85.23|77.12|1.11|-0.95|
|Spanish-AnCora|95.46|89.34|95.69|89.75|0.23|0.41|
|Swedish-LinES|91.52|82.05|91.63|82.83|0.11|0.78|
|Swedish_Sign_Language-SSLC|52.63|44.44|42.59|42.64|-10.04|-1.80|
|Tamil-TTB|76.49|69.12|76.36|65.05|-0.13|-4.07|
|Telugu-MTG|87.55|81.24|88.43|82.46|0.88|1.22|
|Turkish-IMST|88.59|68.72|87.06|68.05|-1.53|-0.67|
|Ukrainian-IU|91.04|78.35|90.98|78.77|-0.06|0.42|
|Urdu-UDTB|91.82|82.61|92.03|83.46|0.21|0.85|
|Uyghur-UDT|83.10|64.30|88.02|64.97|4.92|0.67|
|Vietnamese-VTB|87.27|69.43|87.11|70.42|-0.16|0.99|

