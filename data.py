import numpy as np
from keras.utils import to_categorical, Sequence
from keras.preprocessing.sequence import pad_sequences

from corpus import UDSentence

class Label:
  def __init__(self, name, domain, discrete=True, dictionary=None):
    '''
    if discete == True: domain is a list of labels
    else: domain is a list of tuples whose elements are the range of each dimension 
    '''
    self.name = name
    self.domain = domain
    self.discrete = discrete

    self.pad_value = 0
    self.oov_value = 1

    self.dim = len(self.domain)

    self._iter_idx = 0

    self.dictionary = dictionary
    self.volume = None
    if self.discrete:
      if self.dictionary is None:
        self.dictionary = {t: i+2 for i, t in enumerate(self.domain)}
        self.dictionary['-PAD-'] = self.pad_value
        self.dictionary['-OOV-'] = self.oov_value
      self.rdictionary = {v:k for k,v in self.dictionary.items()}
    else :
      self.volume = 1
      for d in self.domain:
        assert(isinstance(d,tuple))
        assert(len(d) == 2)
        self.volume *= np.abs(d[1]-d[0])
    return

  def __getitem__(self, index):
    try:
      return self.dictionary[index]
    except KeyError:
      return self.dictionary['-OOV-']

  def __len__(self):
    if self.discrete: return len(self.dictionary)
    return self.dim

  def __iter__(self):
    if not self.discrete: raise Exception("continuous label is not iterable")
    self._iter_idx = 0
    return self
  
  def __next__(self):
    assert(self.discrete)
    if self._iter_idx >= self.__len__() : raise StopIteration
    item = self.rdictionary[self._iter_idx]
    self._iter_idx += 1
    return item

class UDDataGenerator(Sequence):

  def __init__(self, corpus, forms, chars, labels, batch_size=10,  shuffle=True):
    self.corpus = corpus
    self.forms = forms
    self.chars = chars
    self.labels= labels
    
    self.shuffle = shuffle
    self.batch_size = batch_size
    self.indexes = np.arange(len(corpus))
    self.num_batches= int(np.ceil(len(corpus)/batch_size))
    self.on_epoch_end()
    return

  def __len__(self):
    return self.num_batches

  def _char_list_padding(self, char_list):
    post = pad_sequences(char_list, padding='post', truncating='post', value=self.chars['-PAD-'], maxlen=3)
    pre  = pad_sequences(char_list, padding='pre', truncating='pre', value=self.chars['-PAD-'], maxlen=3)
    return np.concatenate((post[:, 0:3], pre[:, -3:]),axis=1)

  def _pad_sentence_tensor(self,x, value=0):
    max_snt_len = max([len(snt_mat) for snt_mat in x])
    return np.array([np.pad(snt_mat, ((0,max_snt_len-len(snt_mat)),(0,0)), 'constant', constant_values=value) for snt_mat in x])

  def __getitem__(self, index):
    x = [[], []]
    y = [[] for i in self.labels]
    start = (index-1)*self.batch_size
    end   = np.min((index*self.batch_size,len(self.corpus)))
    for i in self.indexes[range(start, end)]:
      sentence = self.corpus[i]
      forms = [position[self.forms.name] for position in sentence]
      x[0].append([self.forms[i] for i in forms])
      
      chars = self._char_list_padding([[self.chars[i] for i in w] for w in forms])
      x[1].append(chars) 

      for i,label in enumerate(self.labels):
        y[i].append([self.labels[i][position[label.name]] for position in sentence])

    x[0] = pad_sequences(x[0], padding='post', value=self.forms['-PAD-'])
    x[1] = self._pad_sentence_tensor(x[1], self.chars['-PAD-'])

    for i,label in enumerate(self.labels):
      if label.discrete:
        y[i] = pad_sequences(y[i], padding='post', value=label['-PAD-'])
        y[i] = to_categorical(y[i], num_classes=len(label))
      else:
        y[i] = self._pad_sentence_tensor(y[i], 0.0)
    return x, y

  def on_epoch_end(self):
    if self.shuffle: np.random.shuffle(self.indexes)
    return

