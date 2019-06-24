from scipy.spatial.distance import cosine
import numpy as np
from operator import itemgetter
import utils
import os

import pickle
import pdb

class WordVector :
  def __init__(self) :
    return

  def __init__(self, path, unk="<unk_vocab>", beg='<s>', end='</s>') :
    self.path = path 
    self.dim = -1
    self.unk = unk
    self.beg = beg
    self.end = end
    self.delim = " "
    self.size = os.stat(path).st_size 
    self.dic_pickle_path = self.path + '.dic.pkl'
    self.mat_pickle_path = self.path + '.mat.pkl'
    self.dictionary = {}
    self.batch_sz = 100000
    self.mat = np.zeros(shape=(0,0))

    try :
      with open(self.dic_pickle_path, 'rb') as dic_handle, open(self.mat_pickle_path, 'rb') as mat_handle:
        print(("Loading from the existing pickle file {0}".format(path + '.{dic,mat}.pkl')))
        self.dictionary = pickle.load(dic_handle)
        self.mat = pickle.load(mat_handle)
        self.dim = self.mat.shape[1]
    except FileNotFoundError: 
      self.load() 

    if not (self.unk in list(self.dictionary.keys())) : 
      print(("Warning: could not find the unknown token {0}. A zero vector will be used instead".format(self.unk)))
      self.unk = self.update_item(self.unk, np.array([0]*self.dim)) 
    if not (self.beg in list(self.dictionary.keys())) : 
      print(("Warning: could not find the sentence beginning token {0}. A zero vector will be used instead".format(self.beg)))
      self.unk = self.update_item(self.beg, np.array([0]*self.dim)) 
    if not (self.end in list(self.dictionary.keys())) : 
      print(("Warning: could not find the sentence ending token {0}. A zero vector will be used instead".format(self.end)))
      self.unk = self.update_item(self.end, np.array([0]*self.dim)) 

    self.vocabs = list(self.dictionary.keys())

    if not os.path.isfile(self.dic_pickle_path):
      with open(self.dic_pickle_path, 'wb') as handle:
        pickle.dump(self.dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if not os.path.isfile(self.mat_pickle_path):
      with open(self.mat_pickle_path, 'wb') as handle:
        pickle.dump(self.mat, handle, protocol=pickle.HIGHEST_PROTOCOL)

  def __getitem__(self, w): 
    if isinstance(w,int) :
      return self.mat[w]

    try :
      return self.mat[self.dictionary[w]]
    except KeyError :
      return self.mat[self.dictionary[self.unk]]

  def __len__(self):
    return len(self.dictionary)

  def __setitem__(self, w, val): 
    assert(len(val) == self.dim)
    sel.mat[self.dictionary[w]] = np.array(val)


  def load(self) :
    with open(self.path,'r', encoding='utf-8') as fp :
      cnt = 0
      progress = 0


      line = fp.readline()
      tokens = line.split(self.delim) 
      if self.dim < 0:
        if len(tokens) == 2 :
          self.dim = int(tokens[1])
          line = fp.readline()
        else :
          self.dim = len(tokens)-1

      assert(self.dim > 0)
      
      self.mat.resize((0,self.dim), refcheck=False)

      while line :
        if not(cnt%self.batch_sz) :
          (r,c) = self.mat.shape
          self.mat.resize((r+self.batch_sz,c), refcheck=False)

        if not(cnt%1000):
          progress = fp.tell()*1.0/self.size
          utils.update_progress(progress,"Loading word vectors", 40)

        tokens = line.rstrip().split(self.delim)
        assert(len(tokens) == self.dim + 1)
        self.dictionary[tokens[0]] = cnt 
        self.mat[cnt] = np.array([float(x) for x in tokens[1:]])

        line = fp.readline()
        cnt += 1

      if cnt < self.mat.shape[0] : self.mat.resize((cnt,self.dim), refcheck=False)

      utils.update_progress(1,"Loading word vectors", 40)
    return  

  def cosine_dist(self, w1, w2) :
    return cosine(self.dictionary[w1],self.dictionary[w2])

  def update_item(self, word, vector) :
    assert(len(vector) == self.dim)
    try :
      idx = self.dictionary[word]
    except KeyError :
      (r,c) = self.mat.shape 
      self.dictionary[word] = r
      self.mat.resize((r+1,c))
      idx = r
    self.mat[idx] = vector

  def normalize(self):
    mean = np.mean(self.mat, axis=0).tolist()
    std = np.std(self.mat, axis=0).tolist()
    self.mat = (self.mat - mean)/std 

