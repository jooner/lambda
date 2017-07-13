# -*- coding: utf-8 -*-

"""
Implementation of character-level embedding code adopted from:
https://github.com/mkroutikov/tf-lstm-char-cnn

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
import tensorflow as tf

from char_embed import load_data, DataReader
import charmodel as model

flags = tf.flags

flags.DEFINE_string('data_dir', 'data', 'data directory. Should contain train.txt/valid.txt/test.txt with input data')
flags.DEFINE_string('train_dir', 'cv', 'training directory (models and summaries are saved there periodically)')

flags.DEFINE_integer('rnn_size',  500, 'size of LSTM internal state')
flags.DEFINE_integer('rnn_layers', 2, 'number of layers in the LSTM')
flags.DEFINE_integer('highway_layers', 1, 'number of highway layers')

flags.DEFINE_integer('batch_size', 128, 'number of sequences to train on in parallel')
flags.DEFINE_integer('char_embed_size', 15, 'dimensionality of character embeddings')

flags.DEFINE_float('param_init', 0.05, 'initialize parameters at')
flags.DEFINE_string('seed', 1004, 'random number generator seed')
flags.DEFINE_integer('print_every', 5, 'how often to print current loss')

flags.DEFINE_string('kernels', '[1,2,3,4,5,6,7]', 'CNN kernel widths')
flags.DEFINE_string('kernel_features', '[50,100,150,200,200,200,200]', 'number of features in the CNN kernel')

flags.DEFINE_float('dropout', 0.5, 'dropout. 0 = no dropout')

flags.DEFINE_integer('num_unroll_steps', 30, 'number of timesteps to unroll for')
flags.DEFINE_float('max_grad_norm', 5.0, 'normalize gradients at')
flags.DEFINE_integer('max_epochs', 50, 'number of full passes through the training data')
flags.DEFINE_integer('max_word_length', 65, 'maximum word length')

flags.DEFINE_float('learning_rate', 0.001, 'starting learning rate')


#flags.DEFINE_float('learning_rate_decay', 0.5, 'learning rate decay')
#flags.DEFINE_float('decay_when', 1.0, 'decay if validation perplexity does not improve by more than this much')
flags.DEFINE_string('load_model', None, '(optional) Useful for re-starting training from a checkpoint')
flags.DEFINE_string('EOS', '.', '<EOS> symbol.')




FLAGS = flags.FLAGS

def main(_):
  """Train model from data"""

  # look for directory
  if not os.path.exists('data'):
    os.mkdir('data')
    print('Created training directory', 'data')

  word_vocab, char_vocab, tr_word_tensors, tr_char_tensors, max_word_length = load_data(FLAGS.data_dir,
                                                                                        'train',
                                                                                        FLAGS.max_word_length,
                                                                                        FLAGS.EOS)
  _, _, va_word_tensors, va_char_tensors, _ = load_data(FLAGS.data_dir, 'valid',
                                                        FLAGS.max_word_length, FLAGS.EOS)

  train_reader = DataReader(tr_word_tensors, tr_char_tensors, FLAGS.batch_size, FLAGS.num_unroll_steps)
  valid_reader = DataReader(va_word_tensors, va_char_tensors, FLAGS.batch_size, FLAGS.num_unroll_steps)  
  
  with tf.Graph().as_default(), tf.Session() as session:
    # tensorflow seed must be inside graph
    tf.set_random_seed(FLAGS.seed)
    np.random.seed(seed=FLAGS.seed)

    # build training graph
    initializer = tf.random_uniform_initializer(-FLAGS.param_init, FLAGS.param_init)
    with tf.variable_scope("Model", initializer=initializer):
      train_model = model.inference_graph(char_vocab_size=char_vocab.size,
                                          word_vocab_size=word_vocab.size,
                                          char_embed_size=FLAGS.char_embed_size,
                                          batch_size=FLAGS.batch_size,
                                          num_highway_layers=FLAGS.highway_layers,
                                          num_rnn_layers=FLAGS.rnn_layers,
                                          rnn_size=FLAGS.rnn_size,
                                          max_word_length=max_word_length,
                                          kernels=eval(FLAGS.kernels),
                                          kernel_features=eval(FLAGS.kernel_features),
                                          num_unroll_steps=FLAGS.num_unroll_steps,
                                          dropout=FLAGS.dropout)
      train_model.update(model.loss_graph(train_model.logits, FLAGS.batch_size, FLAGS.num_unroll_steps))

      # scaling loss by FLAGS.num_unroll_steps effectively scales gradients by the same factor.
      # we need it to reproduce how the original Torch code optimizes. Without this, our gradients will be
      # much smaller (i.e. 35 times smaller) and to get system to learn we'd have to scale learning rate and
      # max_grad_norm appropriately, thus, scaling gradients so that this trainer is exactly compatible with the original
      train_model.update(model.training_graph(train_model.loss * FLAGS.num_unroll_steps,
                         FLAGS.learning_rate, FLAGS.max_grad_norm))

    saver = tf.train.Saver(max_to_keep=50)

    # create an identical graph model for validation
    with tf.variable_scope("Model", reuse=True):
      valid_model = model.inference_graph(char_vocab_size=char_vocab.size,
                                          word_vocab_size=word_vocab.size,
                                          char_embed_size=FLAGS.char_embed_size,
                                          batch_size=FLAGS.batch_size,
                                          num_highway_layers=FLAGS.highway_layers,
                                          num_rnn_layers=FLAGS.rnn_layers,
                                          rnn_size=FLAGS.rnn_size,
                                          max_word_length=max_word_length,
                                          kernels=eval(FLAGS.kernels),
                                          kernel_features=eval(FLAGS.kernel_features),
                                          num_unroll_steps=FLAGS.num_unroll_steps,
                                          dropout=FLAGS.dropout)
      valid_model.update(model.loss_graph(valid_model.logits, FLAGS.batch_size, FLAGS.num_unroll_steps))

    if FLAGS.load_model:
      saver.restore(session, FLAGS.load_model)
      print('Loaded model from', FLAGS.load_model, 'saved at global step', train_model.global_step.eval())
    else:
      tf.global_variables_initializer().run()
      session.run(train_model.clear_char_embedding_padding)
      print('Created and initialized fresh model. Size:', model.model_size())

    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph=session.graph)

    # take learning rate from CLI, not from saved graph
    session.run(tf.assign(train_model.learning_rate, FLAGS.learning_rate),)

    # training starts here
    best_valid_loss = None
    #rnn_state = session.run(train_model.initial_rnn_state)

    for epoch in range(FLAGS.max_epochs):
      epoch_start_time = time.time()
      avg_train_loss = 0.0
      count = 0
      for x, y in train_reader.iter():
        count += 1
        start_time = time.time()
        rnn_state = session.run(train_model.initial_rnn_state)
        loss, _, rnn_state, gradient_norm, step, _, tr_stats = session.run([
            train_model.loss,
            train_model.train_op,
            train_model.final_rnn_state,
            train_model.global_norm,
            train_model.global_step,
            train_model.clear_char_embedding_padding,
            train_model.stats
        ], {
            train_model.input  : x,
            train_model.targets: y,
            train_model.initial_rnn_state: rnn_state
        })

        avg_train_loss += 0.05 * (loss - avg_train_loss)

        time_elapsed = time.time() - start_time
        
        if count % FLAGS.print_every == 0:
          print('%d: %d [%5d/%5d], train_loss/perplexity = %6.8f/%6.7f secs/batch = %.4fs, grad.norm=%6.8f' % (step,
                                                  epoch, count,
                                                  train_reader.length,
                                                  loss, np.exp(loss),
                                                  time_elapsed,
                                                  gradient_norm))
          print("TRAIN STATS: \n {} \n {}".format(tr_stats[0],tr_stats[1]))

      print('Epoch training time:', time.time()-epoch_start_time)

      # epoch over, evaluate
      avg_valid_loss = 0.0
      count = 0
      # rnn_state = session.run(valid_model.initial_rnn_state)
      for x, y in valid_reader.iter():
        count += 1
        start_time = time.time()
        rnn_state = session.run(valid_model.initial_rnn_state)
        loss, rnn_state, va_stats = session.run([
            valid_model.loss,
            valid_model.final_rnn_state,
            valid_model.stats
        ], {
            valid_model.input  : x,
            valid_model.targets: y,
            valid_model.initial_rnn_state: rnn_state,
        })

        if count % FLAGS.print_every == 0:
            print("\t> validation loss = %6.8f, perplexity = %6.8f" % (loss, np.exp(loss)))
            # stat sanity check
            print("VALID STATS: \n {} \n {}".format(va_stats[0],va_stats[1]))   
        avg_valid_loss += loss / valid_reader.length

      # evaluation over, report
      print("at the end of epoch:", epoch)
      print("train loss = %6.8f, perplexity = %6.8f" % (avg_train_loss, np.exp(avg_train_loss)))
      print("validation loss = %6.8f, perplexity = %6.8f" % (avg_valid_loss, np.exp(avg_valid_loss)))

      save_as = '%s/epoch%d_%.4f.model' % (FLAGS.train_dir, epoch, avg_valid_loss)
      saver.save(session, save_as)
      print('Saved model', save_as)

      # write out summary events
      summary = tf.Summary(value=[tf.Summary.Value(tag="train_loss", simple_value=avg_train_loss),
                                  tf.Summary.Value(tag="valid_loss", simple_value=avg_valid_loss)])
      summary_writer.add_summary(summary, step)

      """
      # decay learning rate if needed
      if best_valid_loss is not None and np.exp(avg_valid_loss) > np.exp(best_valid_loss) - FLAGS.decay_when:
        print('validation perplexity did not improve enough, decay learning rate')
        current_learning_rate = session.run(train_model.learning_rate)
        print('learning rate was:', current_learning_rate)
        current_learning_rate *= FLAGS.learning_rate_decay
        if current_learning_rate < 1.e-5:
          print('learning rate too small - stopping now...')
          break

        session.run(train_model.learning_rate.assign(current_learning_rate))
        print('new learning rate is:', current_learning_rate)
      else:
        best_valid_loss = avg_valid_loss
      """


if __name__ == '__main__':
  tf.app.run()
