# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function

import numpy as np

import io
import os

import numpy as np
from char_encode import encode_character


WORD_EMBEDDING = 50

def word_encode(glove_dir, glove_corpus, glove_vec_size):
  """
  Returns a dictionary that can tokenize words
  """
  glove_path = os.path.join(glove_dir, "glove.{}.{}d.txt".format(glove_corpus, glove_vec_size))
  sizes = {'6B': int(4e5), '42B': int(1.9e6), '840B': int(2.2e6), '2B': int(1.2e6)}
  total = sizes[glove_corpus]
  word2vec_dict = {}
  vec2word_dict = {}
  #all_words = ""
  with io.open(glove_path, 'r', encoding='utf-8') as fh:
    for line in fh:
      array = line.split(" ")
      word = array[0].lower() # lower case all inputs
      #all_words += " {0}".format(word)
      # cache as numpy array for Keras
      vector = np.asarray(list(map(float, array[1:])))
      word2vec_dict[word] = vector
      vec2word_dict[tuple(vector)] = word # cache numpy array s.t. it is hashable
  #token1hot = text.one_hot(all_words, total)
  print("Vectorized {0} words from GloVe.{1}.{2}d".format(len(word2vec_dict), glove_corpus, glove_vec_size))
  return word2vec_dict, vec2word_dict



def word_char_concat(txt, corpus=None):
  pwd = os.path.expanduser(".")
  glove_dir = os.path.join(pwd, "data/GloVe-1.2/glove.6B")

  # TODO: Make this variable to parsed args
  print("GloVe Encoding in Progress...")
  w2v, v2w = word_encode(glove_dir, "6B", WORD_EMBEDDING)
  fp = os.path.join("data", "test.txt")

  word_embed = []
  for w in txt:
    try:
      vec = w2v[w]
    except KeyError:
      vec = [0] * WORD_EMBEDDING
    word_embed.append(vec)


  print("Character Encoding in Progress...")
  char_embedded = encode_character(txt, corpus)

  assert len(word_embed) == len(char_embedded)

  # concatenate
  word_char_embed = []
  for i, e in enumerate(word_embed):
    word_char_embed.append(np.append(e, char_embedded[i]))

  return np.asarray(word_char_embed, dtype=np.float32)


def main():
  return concat(input_txt)

if __name__ == '__main__':
  main()
