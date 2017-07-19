# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function

import sys
import io

import numpy as np
import tensorflow as tf

import cPickle as pickle

# sys.path.append('lambda')

from wordchar_embed import word_char_concat
from convmodel import shape_check, TextCNN

flags = tf.flags

flags.DEFINE_string('squad_obj_fpath', "SquadObject.pkl", 'corpus filepath')
flags.DEFINE_string('test_file', "data/test.txt", 'test file')
flags.DEFINE_integer('batch_size', 64, 'number of parallel training sequences')
flags.DEFINE_integer('processed_size', 128, 'dimensions of text after convolution')
flags.DEFINE_integer('max_context_length', 650, ' number of words in given context')
flags.DEFINE_integer('embedding_size', 65, 'embedded size after word+char preprocessing')
flags.DEFINE_string('kernel_sizes', '[5,7,9]', 'CNN kernel widths')
flags.DEFINE_string('num_kernels', 150, 'number of filters per filter size')



FLAGS = flags.FLAGS

"""
TODO:



"""
def pad_text(wordchar_embedded_txt):
    shape = list(wordchar_embedded_txt.shape)
    shape_check(shape)
    text_length, embed_size = shape[0], shape[1]
    assert embed_size == FLAGS.embedding_size
    excess = FLAGS.max_context_length - text_length
    if excess < 0:
        raise ValueError("ERROR: Text length exceeds maximum context length!")
    padding = np.zeros((excess, embed_size), dtype=np.float32)
    return np.concatenate((wordchar_embedded_txt, padding), axis=0)


def read_testfile(tfile):
    with io.open(tfile, 'r', encoding='utf-8') as _tf:
        output = _tf.read().lower().strip().split()
    return output # list of unicode words

def apply_convolution(concat_input, output_size, batch_size=1):
    # apply layers of convolution and transform into given CNN output size
    convNet = TextCNN(FLAGS.max_context_length, FLAGS.processed_size,
                      FLAGS.embedding_size, eval(FLAGS.kernel_sizes),
                      FLAGS.num_kernels, batch_size)
    conv_output = convNet.run_convolution(concat_input)
    reduced_output = convNet.linear(conv_output)
    return reduced_output

def main(_):
    print("Loading Pickled SQuAD Data Corpus...")
    with open(FLAGS.squad_obj_fpath, 'rb') as input:
        loaded_corpus = pickle.load(input)

    print("Encoding Text with Word+Chararacter Embedding...")
    _input = read_testfile(FLAGS.test_file)
    wordchar_input = word_char_concat(_input, loaded_corpus)
    padded_input = pad_text(wordchar_input)
    print(padded_input[0], padded_input[-1], len(padded_input))


if __name__ == '__main__':
    tf.app.run()
