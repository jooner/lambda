# -*- coding: utf-8 -*-

"""
This is where we char-wise encode our input texts (contexts, questions, answers).
With every given input (word), during training, we will first use
GloVe vectorization to encode our words in 50, 100, or 300-dimensional vectors.

Since GloVe does not handle unseen tokens, here, following the practice seen in
Yoon Kim et al.'s Character-Aware Neural Language Models, we will learn character
embeddings (from contexts) to retain additional information.

Modified implementation of character-level embedding code adopted from:
https://github.com/mkroutikov/tf-lstm-char-cnn

"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import io
import os
import collections
import numpy as np

import prosquad

class Vocab:

  def __init__(self):
    self._token2index = {}
    self._index2token = []

  def feed(self, token):
    if token not in self._token2index:
      # allocate new index for this token
      index = len(self._token2index)
      self._token2index[token] = index
      self._index2token.append(token)

    return self._token2index[token]

  @property
  def size(self):
    return len(self._token2index)

  def token(self, index):
    return self._index2token[index]

  def __getitem__(self, token):
    index = self.get(token)
    if index is None:
      raise KeyError(token)
    return index

  def get(self, token, default=None):
    return self._token2index.get(token, default)

  def save(self, filename):
    with open(filename, 'wb') as f:
      pickle.dump((self._token2index, self._index2token), f, pickle.HIGHEST_PROTOCOL)

  @classmethod
  def load(cls, filename):
    with open(filename, 'rb') as f:
      token2index, index2token = pickle.load(f)
    return cls(token2index, index2token)


class DataReader:

  def __init__(self, word_tensor, char_tensor, batch_size, num_unroll_steps):
    # tensors are np objects
    # length is total number of words in contexts
    length = word_tensor.shape[0]
    assert char_tensor.shape[0] == length

    max_word_length = char_tensor.shape[1]

    # round down length to whole number of slices
    reduced_length = (length // (batch_size * num_unroll_steps)) * batch_size * num_unroll_steps
    word_tensor = word_tensor[:reduced_length]
    char_tensor = char_tensor[:reduced_length, :]

    # rotate so y is indexed to next word
    ydata = np.zeros_like(word_tensor)
    ydata[:-1] = word_tensor[1:].copy()
    ydata[-1] = word_tensor[0].copy()

    x_batches = char_tensor.reshape([batch_size, -1, num_unroll_steps, max_word_length])
    y_batches = ydata.reshape([batch_size, -1, num_unroll_steps])

    x_batches = np.transpose(x_batches, axes=(1, 0, 2, 3))
    y_batches = np.transpose(y_batches, axes=(1, 0, 2))

    self._x_batches = list(x_batches)
    self._y_batches = list(y_batches)
    assert len(self._x_batches) == len(self._y_batches)
    self.length = len(self._y_batches)

    self.batch_size = batch_size
    self.num_unroll_steps = num_unroll_steps

  def iter(self):
    for x, y in zip(self._x_batches, self._y_batches):
      yield x, y


def load_data(data_dir, data_type, max_word_length, eos='.'):

  char_vocab = Vocab()
  char_vocab.feed(' ')  # blank is at index 0 in char vocab
  char_vocab.feed('{')  # start is at index 1 in char vocab
  char_vocab.feed('}')  # end   is at index 2 in char vocab
  char_vocab.feed(eos)  # eos   is at index 3 in char vocab

  word_vocab = Vocab()
  word_vocab.feed(eos) # eos is at index 0 in word vocab

  actual_max_word_length = 0

  word_tokens = []
  char_tokens = []

  # output context corpus into a txt file
  corpus = prosquad.initiate_process(data_dir, data_type)
  contexts = corpus.contexts # list of list of strings
  # flatten into list of context strings
  contexts = [item for sublist in contexts for item in sublist]
  # abridge because it's too big for GPU
  # contexts = contexts[:(len(contexts) * 5 // 10)]
  # make into a single string
  contexts = (" ".join(contexts)).lower()
  # access corpus word by word
  contexts = contexts.replace(eos, '')
  for word in contexts.split():
    # space for 'start' and 'end' chars
    if len(word) > max_word_length - 2:
      word = word[:max_word_length-2]

    word_tokens.append(word_vocab.feed(word))

    char_array = [char_vocab.feed(c) for c in '{' + word + '}']
    char_tokens.append(char_array)

    actual_max_word_length = max(actual_max_word_length, len(char_array))

  assert actual_max_word_length <= max_word_length

  print()
  print('actual longest token length is:', actual_max_word_length)
  print('size of word vocabulary:', word_vocab.size)
  print('size of char vocabulary:', char_vocab.size)

  # now we know the sizes, create tensors
  assert len(char_tokens) == len(word_tokens)

  word_tensors = np.array(word_tokens, dtype=np.int32)
  char_tensors = np.zeros([len(char_tokens), actual_max_word_length], dtype=np.int32)

  for i, char_array in enumerate(char_tokens):
    char_tensors[i,:len(char_array)] = char_array # number of rows is words, column is char

  return word_vocab, char_vocab, word_tensors, char_tensors, actual_max_word_length
