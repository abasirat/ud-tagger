import os
import re
import mmap
import utils 
from collections import defaultdict

import pdb

class Sentence:
  def __init__(self):
    return

class RawSentence:
  # TO BE IMPLEMENTED
  def __init__(self, sentence):
    if isinstance(sentence,str):
      self.sentence_str = sentence
      self.forms = sentence.split()
      self.positions = self.forms

  def __len__(self): return len(self.forms)
  def __getitem__(self,idx): 
    try:
      return self.positions[idx]
    except IndexError:
      raise(IndexError("Index out of range: maximum sentence length is {0}".format(len(self))))
    
###################################################
class UDWord:
  def __init__(self, parts, lower):
    assert(len(parts) == 10)
    self.dictionary = dict(zip(['ID', 'FORM', 'LEMMA', 'UPOS', 'XPOS', 'FEATS', 'HEAD', 'DEPREL', 'DEPS', 'MISC'],parts))

    if lower: 
      self.dictionary['FORM'] = self.dictionary['FORM'].lower()
      self.dictionary['LEMMA'] = self.dictionary['LEMMA'].lower()

    if self.dictionary['FEATS'] != '_':
      feats = self.dictionary['FEATS'].split('|')
      for feat in feats:
        try: 
          (f,v) = feat.split('=')
        except ValueError:
          raise ValueError("error in features columns {0} feature {1}".format(self.dictionary['FEATS'],feat))

        self.dictionary[f] = v
       
  def __getitem__(self,key):
    try: 
      return self.dictionary[key]
    except KeyError:
      return '_'

  def __setitem__(self, key, value):
    try:
      self.dictionary[key] = value
    except KeyError:
      raise KeyError("UDWord: invalid key: ", key)

  def is_mwe(self): return re.search('-',self.dictionary['ID'])


class UDSentence:
  def __init__(self, sentence, verbose = 0, lower=True):
    self.verbose = verbose
    self.__current_idx = 0
    self.lower = lower
    self.positions, self.mwes, self.forms, self.utags, self.sentence_str = self._extract(sentence, self.verbose)
    

  def _extract(self, sentence, verbose = 0):
    if isinstance(sentence, str):
      tokens = sentence.split(CoNLLUCorpus.EOL)
      tokens = [x for x in tokens if x] # remove th empty tokens. The last token is always redundant
    elif isinstance(sentence,list): # list of strings
      tokens = sentence

    positions = [UDWord(['0', '*root*', '_', '_', '_', '_', '0', '_', '_', '_'], self.lower)]
    mwe = []
    forms = []
    utags = []
    sentence_str = ''
    for token in tokens:
      if (token.startswith(CoNLLUCorpus.COMMENT_MARKER)): continue
      udword = UDWord(token.split(CoNLLUCorpus.DELIM), self.lower)
      if udword.is_mwe():
        mwe.extend(udword['FORM'].split('-'))
        sentence_str += udword['FORM']
        if not re.search('SpaceAfter=No', udword['MISC']): sentence_str += ' '
      elif re.search('^[0-9]+$',udword['ID']): 
        positions.append(udword)
        forms.append(udword['FORM'])
        utags.append(udword['UTAG'])
        if not(udword['ID'] in mwe): 
          sentence_str += udword['FORM']
          if not re.search('SpaceAfter=No', udword['MISC']): sentence_str += ' '
      elif re.search('.',udword['ID']): 
        if verbose > 0:
          print("Warning: the position with ID {0} is ignored".format(udword['ID'])) 
      else:
        raise ValueError("error in ID format: {0}".format(udword['ID']))

    # replace head id with head form
    for p in positions: p['HEAD'] = positions[int(p['HEAD'])]['FORM']

    return positions, mwe, forms, utags, sentence_str


  def __del__(self):
    return

  def __getitem__(self,x):
    assert(isinstance(x,int))
    return self.positions[x]

  def __len__(self):
    return len(self.positions)

  def __iter__(self):
    self.__current_idx = 0
    return self

  def __next__(self):
    if self.__current_idx >= self.__len__() : raise StopIteration
    current = self[self.__current_idx]
    self.__current_idx += 1
    return current

###################################################

class Corpus:
  def __init__(self, path, encoding='utf-8', verbose=0) :
    self.path = path 
    self.encoding = encoding
    self.num_sentences = 0
    try:
      fp = open(self.path,'r+', encoding=self.encoding)
    except FileNotFoundError :
      raise Exception("corpus not found in {0}".format(self.path))
    else :
      self.__fp = fp

    self.size = os.stat(path).st_size 

  def __del__(self): self.__fp.close()
  def __len__(self): return self.num_sentences
  def reset(self): self.__fp.seek(0,os.SEEK_SET)
  def progress(self): return self.__fp.tell()*1.0/self.size
  def tell(self): return self.__fp.tell()
  def seek(self,pos, whence=os.SEEK_SET): self.__fp.seek(pos, whence)
  def handler(self): return self.__fp
  def readline(self): return self.__fp.readline()

###################################################

# each line is a sentence with no annotation
class RawCorpus(Corpus):
  def __init__(self, path, block_size=1024**2) :
    self.vocabs = {}
    super().__init__(path)
    self.vocabs, self.num_tokens, self.num_sentences = self.__get_vocabs()
    return

  def __iter__(self):
    self.seek(0)
    return self

  def __next__(self):
    if self.tell() >= self.size : raise StopIteration
    return self._next_sentence()

  def _next_sentence(self):
    return RawSentence(self.readline().rstrip().lstrip())

  def __get_vocabs(self) :
    num_tokens = 0
    num_sentences = 0
    vocabs = dict
    with open(self.path,'r') as f :
      progress = 0
      line = f.readline()
      while line :
        num_sentences += 1
        if not(num_sentences%10000) :
          progress = f.tell()*1.0 / self.size
          utils.update_progress(progress,"Counting vocabs", 40)

        tokens = line.rstrip().split()
        num_tokens += len(tokens)
        list(map(lambda x: utils.inc_dict_value(vocabs,x) , tokens))

        line = f.readline()

      if progress < 1: utils.update_progress(1,"Counting vocabs", 40)
    return vocabs, num_tokens, num_sentences

###################################################

class CoNLLUCorpus(Corpus):
  EOL='\n'
  DELIM='\t'
  COMMENT_MARKER='#'

  def __init__(self,path) :
    super().__init__(path, encoding="utf-8")
    self.vocabs, self._sentence_offset, self.num_sentences, self.tagset = self.__index_corpus() 
    return 

  def _next_sentence(self):
    sentence_str = ""
    line = self.readline()
    while line:
      if (line.startswith(CoNLLUCorpus.EOL)): break 
      sentence_str += line
      line = self.readline()
    return UDSentence(sentence_str)
    
  def __getitem__(self,index):
    try:
      self.seek(self._sentence_offset[index])
    except IndexError:
      raise(IndexError("index must be in [0,{0})".format(self.num_sentences)))

    return self._next_sentence()

  def __iter__(self):
    self.seek(0)
    return self

  def __next__(self):
    if self.tell() >= self.size : raise StopIteration
    return self._next_sentence()

  def __index_corpus(self):
    offsets = [] 
    num_sentences = 0

    tagset = defaultdict(set)
    vocabs = {}

    self.seek(0)
    offset = self.tell()

    line = self.readline()
    for snt in self:
      offsets.append(offset)
      offset = self.tell()
      for udword in snt:
        for f in udword.dictionary:
          if f not in ['FORM', 'LEMMA', 'HEAD', 'FEATS']: tagset[f].add(udword[f])
          utils.inc_dict_value(vocabs, udword['FORM'])
      num_sentences += 1
      
    return vocabs, offsets, num_sentences, tagset

