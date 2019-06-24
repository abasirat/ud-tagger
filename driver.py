from corpus import CoNLLUCorpus
from word_vector import WordVector
import numpy as np
from data import UDDataGenerator, Label
from tagger import Tagger

emb_file = 'data/en.vectors.txt'
train_corpus_file = "ud-treebanks-v2.3/UD_English-EWT/en_ewt-ud-train.conllu"
validation_corpus_file = "ud-treebanks-v2.3/UD_English-EWT/en_ewt-ud-dev.conllu"
test_corpus_file = "ud-treebanks-v2.3/UD_English-EWT/en_ewt-ud-test.conllu"

max_vocab_size=40000

print("loading corpora ...")
training_corpus = CoNLLUCorpus(train_corpus_file)
validation_corpus = CoNLLUCorpus(validation_corpus_file)
test_corpus = CoNLLUCorpus(test_corpus_file)

print("loading embddings ...")
emb = WordVector(emb_file)

print("constructing word list ...", end=' ')
words = sorted(training_corpus.vocabs, key=lambda key: training_corpus.vocabs[key], reverse=True)
words = words[0:np.min( (max_vocab_size, len(words)) )]
if emb is not None and len(words) < max_vocab_size:
  for w in sorted(emb.dictionary, key=lambda key: emb.dictionary[key], reverse=False):
    if len(words) >= max_vocab_size: break
    if w not in words: words.append(w)
print("{0} unique words counted".format(len(words)))

forms = Label('FORM', words, discrete=True)
print("constructing embedding matrix ...")
if emb is not None:
  emb_mat = np.zeros((len(forms), emb.dim))
  for w in forms: emb_mat[forms[w]] = emb[w]
  emb_mat[forms['-OOV-']] = emb[emb.unk].reshape(1,emb.dim)
  emb_mat[forms['-PAD-']] = np.zeros((1,emb.dim))

print("constructing character list ...")
chars = []
for w in words: chars.extend(w)
chars = Label('FORM', list(set(chars)))

upos_tags = training_corpus.tagset['UPOS'] | validation_corpus.tagset['UPOS']
upos = Label('UPOS', list(upos_tags), discrete=True)

batch_size=10
tags = [upos]
training_generator = UDDataGenerator(training_corpus, forms, chars, tags)
validation_generator = UDDataGenerator(validation_corpus, forms, chars, tags)
test_generator = UDDataGenerator(test_corpus, forms, chars, tags)

tagger = Tagger(training_generator, validation_generator, test_generator, word_emb_mat=emb_mat, num_epochs=10)
tagger.fit()
##tagger.save_model()
score = tagger.evaluate(test_generator)
print("Accuracy on the test data: {0:1.4f}".format(score[1]))
