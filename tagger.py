import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding, Bidirectional, TimeDistributed, Input, Activation, Concatenate
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy, mean_squared_error
from keras.models import load_model, Model
from data import UDDataGenerator

class Tagger:
  def __init__(self, training_generator, validation_generator, test_generator, word_emb_mat=None, freez_word_emb=True, num_epochs=5, drop_out_rate=0.1, model_path='model/tagger.h5'):
    self.training_generator = training_generator
    self.validation_generator = validation_generator
    self.test_generator = test_generator

    assert(self.training_generator.labels == self.validation_generator.labels)
    assert(self.training_generator.labels == self.test_generator.labels)

    self.num_outputs = len(self.training_generator.labels) 
    self.num_classes = [len(label) for label in self.training_generator.labels]
    self.outputs_name=[label.name for label in self.training_generator.labels]
    self.losses = [categorical_crossentropy if label.discrete else mean_squared_error for label in self.training_generator.labels]
    self.activation = ['softmax' if label.discrete else 'linear' for label in self.training_generator.labels]
    self.metrics = ['accuracy']#, 'mse']


    self.num_epochs = num_epochs
    self.drop_out_rate = drop_out_rate

    self.model_path = model_path

    batch_size = self.training_generator.batch_size
    num_words = len(self.training_generator.forms)
    num_chars = len(self.training_generator.chars)

    if word_emb_mat is not None: 
      word_emb_dim = word_emb_mat.shape[1]
    else :
      word_emb_dim = 100

    char_emb_dim=50

    words_input = Input(shape=(None,), dtype='int32')
    if word_emb_mat is not None: 
      w = Embedding(num_words, output_dim=word_emb_dim, weights=[word_emb_mat], trainable=not(freez_word_emb), mask_zero=True, name='embeding')(words_input)
    else:
      w = Embedding(num_words, output_dim=word_emb_dim, mask_zero=True, name='embeding')(words_input)
    w = Dropout(self.drop_out_rate)(w)
    w = Bidirectional(LSTM(500, return_sequences=True))(w)
    w = Dropout(self.drop_out_rate)(w)

    chars_input = Input(shape=(None,6,), dtype='int32')
    char_emb_layer = Embedding(num_chars, output_dim=char_emb_dim, mask_zero=True, name='char_embeding')
    c = TimeDistributed(char_emb_layer)(chars_input)
    c = Dropout(self.drop_out_rate)(c)
    c = TimeDistributed(Bidirectional(LSTM(10, return_sequences=False)))(c)
    c = Dropout(self.drop_out_rate)(c)
    c = Bidirectional(LSTM(50, return_sequences=True))(c)
    c = Dropout(self.drop_out_rate)(c)
    
    concatenated = (Concatenate()([w, c]))

    y = []
    for o in range(self.num_outputs):
      x = TimeDistributed(Dense(self.num_classes[o]))(concatenated)
      y.append(Activation(self.activation[o], name=self.outputs_name[o])(x))

    self.model = Model([words_input, chars_input], y)
    self.model.compile(loss=self.losses, optimizer=Adam(0.001),metrics=self.metrics)
    self.model.summary()
    return

  def fit(self):
    self.model.fit_generator(generator=self.training_generator,
                        validation_data=self.validation_generator,
                        epochs=self.num_epochs)
    return

  def save_model(self):
    self.model.save(self.model_path)

  def load_model(path):
    return load_model(path)

  def evaluate(self, test_generator):
    score = self.model.evaluate_generator(test_generator, len(test_generator))
    return score

  def tag(self, test_generator):
    return self.model.predict_generator(test_generator, len(test_generator))
