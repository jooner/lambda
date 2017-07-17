# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function

import sys

import numpy as np
import tensorflow as tf

import cPickle as pickle

# sys.path.append('lambda')

from char_embed import DataReader, load_data
from charmodel import inference_graph
from prosquad import initiate_process

flags = tf.flags

flags.DEFINE_string('load_model', None, 'char-embedding model to use')
flags.DEFINE_string('squad_train_data', "lambda/data", 'SQuAD data directory')
flags.DEFINE_string('data_type', "train", 'trainig or validation')
flags.DEFINE_string('squad_obj_fpath', "SquadObject.pkl", 'corpus filepath')

FLAGS = flags.FLAGS

"""
 TODO: (single-tape location-blind network)
 1. run prosquad and get corpus
 2. X_context is the corpus contexts
 3. X_question is the corresponding question
 4. Get GloVe embedding
 5. Get character-level embedding
 6. For every word in context and question, embed through 4 & 5 then concat
 7. Train: Run input through DNC and get output and compare to Answer




"""

def main(_):
    print("Initiating Vanilla Run...")
    print("Pickling SQuAD Data Corpus...")
    with open(FLAGS.squad_obj_fpath, 'rb') as input:
        loaded_corpus = pickle.load(input)



if __name__ == '__main__':
    tf.app.run()
