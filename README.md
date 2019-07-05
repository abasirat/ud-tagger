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
|Afrikaans-AfriBooms|0.9364|0.8340|0.9387|0.8460|0.0023|0.0120|
|Ancient_Greek-Perseus|0.8496|0.7460|0.8660|0.7476|0.0164|0.0016|
|Arabic-PADT|0.9378|0.8387|0.9412|0.8403|0.0034|0.0016|
|Basque-BDT|0.8868|0.7932|0.9001|0.7959|0.0133|0.0027|
|Belarusian-HSE|0.8109|0.6457|0.8028|0.6584|-0.0081|0.0127|
|Bulgarian-BTB|0.9577|0.8698|0.9566|0.8788|-0.0011|0.0090|
|Catalan-AnCora|0.9520|0.8996|0.9546|0.9032|0.0026|0.0036|
|Coptic-Scriptorium|0.9112|0.8026|0.9150|0.8117|0.0038|0.0091|
|Croatian-SET|0.9296|0.8097|0.9315|0.8168|0.0019|0.0071|
|Czech-CLTT|0.9589|0.8618|0.9563|0.8626|-0.0026|0.0008|
|Danish-DDT|0.9234|0.8342|0.9262|0.8348|0.0028|0.0006|
|Dutch-LassySmall|0.9134|0.8082|0.9216|0.8188|0.0082|0.0106|
|English-LinES|0.9296|0.8349|0.9349|0.8462|0.0053|0.0113|
|Estonian-EDT|0.8966|0.8121|0.9066|0.8194|0.0100|0.0073|
|Finnish-TDT|0.8637|0.7759|0.8761|0.7782|0.0124|0.0023|
|French-FTB|0.3092|0.2521|||-0.3092|-0.2521|
|Gothic-PROIEL|0.9265|0.7978|0.9308|0.8046|0.0043|0.0068|
|Greek-GDT|0.9201|0.8565|0.9235|0.8653|0.0034|0.0088|
|Hebrew-HTB|0.9426|0.8574|0.9464|0.8690|0.0038|0.0116|
|Hindi_English-HIENCS|0.3127|0.2399|0.3201|0.2289|0.0074|-0.0110|
|Hungarian-Szeged|0.8062|0.7388|0.8310|0.7465|0.0248|0.0077|
|Indonesian-GSD|0.8637|0.8199|0.8817|0.8202|0.0180|0.0003|
|Italian-ISDT|0.9546|0.9038|0.9587|0.9098|0.0041|0.0060|
|Japanese-GSD|0.9565|0.9562|0.9655|0.9628|0.0090|0.0066|
|Korean-Kaist|0.9296|0.8640|0.9309|0.8715|0.0013|0.0075|
|Latin-ITTB|0.9597|0.8800|0.9629|0.8834|0.0032|0.0034|
|Latvian-LVTB|0.8871|0.7858|0.8944|0.7934|0.0073|0.0076|
|Lithuanian-HSE|0.6671|0.5322|0.5875|0.4897|-0.0796|-0.0425|
|Maltese-MUDT|0.9082|0.7998|0.9043|0.7977|-0.0039|-0.0021|
|Marathi-UFAL|0.7178|0.6655|0.7288|0.6599|0.0110|-0.0056|
|Norwegian-Bokmaal|0.9471|0.8884|0.9537|0.8918|0.0066|0.0034|
|Old_Church_Slavonic-PROIEL|0.9298|0.8236|0.9329|0.8269|0.0031|0.0033|
|Old_French-SRCMF|0.9318|0.8477|0.9341|0.8546|0.0023|0.0069|
|Persian-Seraji|0.9510|0.8631|0.9576|0.8642|0.0066|0.0011|
|Polish-LFG|0.9419|0.8896|0.9435|0.8930|0.0016|0.0034|
|Portuguese-GSD|0.9427|0.9267|0.9449|0.9316|0.0022|0.0049|
|Romanian-Nonstandard|0.9455|0.8610|0.9524|0.8630|0.0069|0.0020|
|Russian-SynTagRus|0.9559|0.9018|0.9593|0.9036|0.0034|0.0018|
|Serbian-SET|0.9180|0.8304|0.9313|0.8299|0.0133|-0.0005|
|Slovak-SNK|0.8412|0.7807|0.8523|0.7712|0.0111|-0.0095|
|Spanish-AnCora|0.9546|0.8934|0.9569|0.8975|0.0023|0.0041|
|Swedish-LinES|0.9152|0.8205|0.9163|0.8283|0.0011|0.0078|
|Swedish_Sign_Language-SSLC|0.5263|0.4444|0.4259|0.4264|-0.1004|-0.0180|
|Tamil-TTB|0.7649|0.6912|0.7636|0.6505|-0.0013|-0.0407|
|Telugu-MTG|0.8755|0.8124|0.8843|0.8246|0.0088|0.0122|
|Turkish-IMST|0.8859|0.6872|0.8706|0.6805|-0.0153|-0.0067|
|Ukrainian-IU|0.9104|0.7835|0.9098|0.7877|-0.0006|0.0042|
|Urdu-UDTB|0.9182|0.8261|0.9203|0.8346|0.0021|0.0085|
|Uyghur-UDT|0.8310|0.6430|0.8802|0.6497|0.0492|0.0067|
|Vietnamese-VTB|0.8727|0.6943|0.8711|0.7042|-0.0016|0.0099|

