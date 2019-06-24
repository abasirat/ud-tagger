import os
import re
import pdb

class Sentence:
  field_dict = dict(zip(['ID', 'FORM', 'LEMMA', 'UPOS', 'XPOS', 'FEATS', 'HEAD', 'DEPREL', 'DEPS', 'MISC'],range(10)))
  def __init__(self, sentence):

    self.positions, self.mwes, self.forms, self.sentence_str = Sentence.extract(sentence)
    

  def extract(sentence):
    if isinstance(sentence, str):
      tokens = sentence.split(CoNLLU.EOL)
      tokens = [x for x in tokens if x] # remove th empty tokens. The last token is always redundant
    elif isinstance(sentence,list): # list of strings
      tokens = sentence

    positions = [['0', '*root*', '-', '-', '-', '-', '-', '-', '-', '-']]
    mwe = []
    forms = []
    sentence_str = ''
    for token in tokens:
      if (token.startswith(CoNLLU.COMMENT_MARKER)): continue
      parts = token.split(CoNLLU.DELIM)
      assert(len(parts) == 10)
      if re.search('-',parts[0]): 
        mwe.extend(parts[0].split('-'))
        sentence_str += parts[1]
        if not re.search('SpaceAfter=No', parts[9]): sentence_str += ' '
      elif re.search('^[0-9]+$',parts[0]): 
        positions.append(parts)
        forms.append(parts[1])
        if not(parts[0] in mwe): 
          sentence_str += parts[1]
          if not re.search('SpaceAfter=No', parts[9]): sentence_str += ' '
      elif re.search('.',parts[0]):
        print("Warning: the position with ID {0} is ignored".format(parts[0])) 
      else:
        raise ValueError("error in ID format: {0}".format(parts[0]))

    
    return positions, mwe, forms, sentence_str


  def __del__(self):
    return

  def __getitem__(self,x):
    if isinstance(x,int):
      try:
        v = self.positions[x]
      except IndexError:
        raise(IndexError("Index out of range: maximum sentence length is {0}".format(len(self.words))))
    elif isinstance(x,tuple) or isinstance(x,list) :
      (i,j) = x
      v = self[i]
      if isinstance(j,str):
        j = Sentence.field_dict[j]

      assert(isinstance(j,int))
      try:
        v = v[j]
      except IndexError:
        raise(IndexError("Invalid y index. Use either feild name or integer in [0,10)"))

    return v

  def __len__(self):
    return len(self.words)

class CoNLLU:
  EOL='\n'
  DELIM='\t'
  COMMENT_MARKER='#'

  def __init__(self,path) :
    self.path = path

    try:
      self._fp = open(self.path, mode="r", encoding="utf-8")
    except FileNotFoundError:
      raise FileNotFoundError("cannot find file {0}".format(self.path))

    self._fp.seek(0, os.SEEK_END)
    self._fsize = self._fp.tell()
    self._fp.seek(0,os.SEEK_SET)

    self._sentence_offset, self.num_sentences = self.index_corpus() 
    return 
  
  def __del__(self):
    self._fp.close()
    return

  def _next_sentence(self): # read the next sentence from _fp
    sentence_str = ""
    line = self._fp.readline()
    while line:
      if (line.startswith(CoNLLU.EOL)): break 
      sentence_str += line
      line = self._fp.readline()
    return Sentence(sentence_str)
    


  def __getitem__(self,index):
    try:
      self._fp.seek(self._sentence_offset[index],0)
    except IndexError:
      raise(IndexError("index must be in [0,{0})".format(self.num_sentences)))

    return self._next_sentence()

  def __iter__(self):
    self._fp.seek(0,os.SEEK_SET)
    return self

  def __next__(self):
    if self._fp.tell() == self._fsize : raise StopIteration
    return self._next_sentence()

  def index_corpus(self):
    offsets = [] 
    num_sentences = 0

    self._fp.seek(0,os.SEEK_SET)
    offset = self._fp.tell()

    line = self._fp.readline()
    for snt in self:
      offsets.append(offset)
      offset = self._fp.tell()
      num_sentences += 1
      
    return offsets, num_sentences

