# -*- coding: utf-8 -*-

"""
This is our trained model for char-wise embedding is used for the input text.
The output should be concatenated to a word vector encoding yielded by GloVe.


Modified implementation of character-level embedding code adopted from:
https://github.com/mkroutikov/tf-lstm-char-cnn

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import io
import time

import numpy as np
import tensorflow as tf

from char_embed import load_data, DataReader
import charmodel as model


flags = tf.flags

flags.DEFINE_string('load_model', None, '(optional) Useful for re-starting training from a checkpoint')

flags.DEFINE_integer('num_samples', 3, 'how many words to generate')
flags.DEFINE_string('seed', 3435, 'random number generator seed')
flags.DEFINE_string('EOS', '.', '<EOS> symbol.')


flags.DEFINE_string('data_dir', 'data', 'data directory. Should contain train.txt/valid.txt/test.txt with input data')
flags.DEFINE_string('train_dir', 'cv', 'training directory (models and summaries are saved there periodically)')

flags.DEFINE_integer('rnn_size',  650, 'size of LSTM internal state')
flags.DEFINE_integer('rnn_layers', 2, 'number of layers in the LSTM')
flags.DEFINE_integer('highway_layers', 2, 'number of highway layers')
flags.DEFINE_integer('char_embed_size', 15, 'dimensionality of character embeddings')
flags.DEFINE_string('kernels', '[1,2,3,4,5,6,7]', 'CNN kernel widths')
flags.DEFINE_string('kernel_features', '[50,100,150,200,200,200,200]', 'number of features in the CNN kernel')
flags.DEFINE_float('temperature', 1.0, 'sampling temperature')

flags.DEFINE_float('dropout', 0.5, 'dropout. 0 = no dropout')
flags.DEFINE_integer('max_word_length', 65, 'maximum word length')

FLAGS = flags.FLAGS

def main(_input):
  """Encode given text"""
  if FLAGS.load_model is None:
    print("Specify checkpoint file to load model from!")
    return -1

  if not os.path.exists(FLAGS.load_model + '.meta'):
    print("Checkpoint file NOT found, ", FLAGS.load_model)
    return -1

  word_vocab, char_vocab, tr_word_tensors, tr_char_tensors, max_word_length = load_data(FLAGS.data_dir,
                                                                                        'train',
                                                                                        FLAGS.max_word_length,
                                                                                        FLAGS.EOS)
  print('initialized test dataset reader')
  with tf.Graph().as_default(), tf.Session() as session:
    # tensorflow seed must be inside graph
    tf.set_random_seed(FLAGS.seed)
    np.random.seed(seed=FLAGS.seed)

    # build training graph
    with tf.variable_scope("Model"):
      m = model.inference_graph(char_vocab_size=char_vocab.size,
                                word_vocab_size=word_vocab.size,
                                char_embed_size=FLAGS.char_embed_size,
                                batch_size=1,
                                num_highway_layers=FLAGS.highway_layers,
                                num_rnn_layers=FLAGS.rnn_layers,
                                rnn_size=FLAGS.rnn_size,
                                max_word_length=max_word_length,
                                kernels=eval(FLAGS.kernels),
                                kernel_features=eval(FLAGS.kernel_features),
                                num_unroll_steps=1, dropout=0.0)

    # we need global step only because we want to read it from the model
    # global_step = tf.Variable(0, dtype=tf.int32, name='global_step')
    saver = tf.train.Saver()
    saver.restore(session, FLAGS.load_model)
    # print("Loaded model from", FLAGS.load_model, 'saved at global step', global_step.eval())

    # run RNN
    rnn_state = session.run(m.initial_rnn_state)
    """
    logits = np.ones((word_vocab.size,))

    _, _, word_tensors, char_tensors, _ = load_data(FLAGS.data_dir, 'test.txt', FLAGS.max_word_length, FLAGS.EOS)
    for i in range(FLAGS.num_samples):
      logits = logits / FLAGS.temperature
      prob = np.exp(logits)
      prob /= np.sum(prob)
      prob = prob.ravel()
      idx = np.random.choice(range(len(prob)), p=prob)
      word = word_vocab.token(idx)
      if word == '.':
        print("<unk> {}".format(word), end=' ')
      else:
        print(word, end=' ')
      char_input = np.zeros((1,1,max_word_length))

      for i, c in enumerate('{' + word + '}'):
        char_input[0,0,i] = char_vocab[c]
      logits, rnn_state = session.run([m.logits, m.final_rnn_state],
                                      {m.input: char_input, m.initial_rnn_state: rnn_state})
      logits = np.array(logits)
    char_input
    """

    # testing
    k = []
    for w in _input:
      char_input = np.zeros((1,1,max_word_length))
      for i, c in enumerate('{' + w + '}'):
        char_input[0,0,i] = char_vocab[c]
      output = session.run([m.output_cnn], {m.input: char_input, m.initial_rnn_state: rnn_state})
      k.append(list(output[0]))
  return k
      

if __name__ == '__main__':
  tf.app.run()
