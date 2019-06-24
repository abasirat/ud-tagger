import numpy as np
import utils
import pdb

class ContextGenerator :
  def __init__(self, corpus, context) :
    self.corpus = corpus
    self.context = context

    self.dim = self.context.dim
    self.length = corpus.num_tokens + 2*corpus.num_sentences

    self.beg = self.context.embeddings.beg
    self.end = self.context.embeddings.end
    self.vocab_dict = context.embeddings.dictionary

    self.buf = []
    self.buf_beg_idx = 0
    self.buf_end_idx = 0
    self.buf_size = 1024
    return 

  def shape(self):
    return (self.length, self.dim)

  def __len__(self) :
    return self.length

  def __getitem__(self,pos) :
    if isinstance(pos, int): 
      return self.__get_item__(pos)
    elif isinstance(pos,slice):
      start = pos.start or 0
      stop = pos.stop or self.corpus_index.shape[0]
      step = pos.step or 1
      return np.vstack([self.__get_item__(ii) for ii in range(start, stop, step)])
    elif isinstance(pos,list):
      return np.vstack([self.__get_item__(ii) for ii in pos])
    else :
      raise(TypeError, "Invalid argument type")

  def __get_item__(self,pos):
    if self.beg_buf_idx <= pos and pos <= buf_end_idx:
      return self.buf[pos-self.buf_beg_idx]
    elif pos > self.beg_buf_idx:
      self.buf = []

######################################
class Context :
  def __init__(self, context_builder, param_dict, embeddings) :
    self.params = param_dict

    self.context_builder = context_builder

    self.embeddings = embeddings
    self.embeddings.normalize()
    self.__unk = embeddings.unk

    self.context_unit_dict = embeddings.dictionary 

    self.__padding_vector = np.zeros((self.embeddings.dim,))

    self.context_builder_dict = {"concat":self.concatenate_context, "cat":self.concatenate_context, "linear":self.linear_context, "lin":self.linear_context} 
    
    self.dim = 0
    if self.context_builder_dict[self.context_builder] == self.concatenate_context:
      self.dim = 2*self.embeddings.dim * self.params["window_size"] 
    elif self.context_builder_dict[self.context_builder] == self.linear_context:
      self.dim = self.embeddings.dim
    assert(self.dim > 0)

    return

  def get_unit_id(self, context) :
    try :
      return self.context_unit_dict[context]
    except KeyError:
      return self.context_unit_dict[self.__unk]

  def get_context(self, context_ids) :
    context_vectors = np.vstack([ self.embeddings[int(i)] for i in context_ids])
    context_matrix = self.context_builder_dict[self.context_builder](context_vectors)
    return context_matrix

  def concatenate_context(self, context_vectors) :
    win_sz = self.params["window_size"]
    assert(win_sz > 0) 
    padding_matrix = np.vstack( [ self.__padding_vector for _ in range(0,win_sz) ] )
    padded_context_vectors = np.vstack([padding_matrix, context_vectors , padding_matrix])
    context_matrix = np.vstack( \
        [ np.concatenate( \
          padded_context_vectors [ np.concatenate([np.arange(i-win_sz,i) , np.arange(i+1,i+win_sz+1)]),:], axis=0) \
          for i in range(win_sz,padded_context_vectors.shape[0]-win_sz) ] ) 
    return context_matrix

  def linear_context(self, context_vectors, params) :
    win_sz = self.params["window_size"]
    assert(win_sz > 0) 
    padding_matrix = np.vstack( [ self.__padding_vector for _ in range(0,win_sz) ] )
    padded_context_vectors = np.vstack([padding_matrix, context_vectors , padding_matrix])
    context_matrix = np.vstack( \
        ( \
            map( lambda i:  np.concatenate( (\
                  np.mean( padded_context_vectors[i-win_sz:i , :], axis=0 ) , \
                  np.mean( padded_context_vectors [i+1:i+win_sz+1,:], axis=0 )), \
                  axis=0 ), \
               range(win_sz,padded_context_vectors.shape[0]-win_sz)) \
        ) \
    )
    return context_matrix

