# -*- coding: utf-8 -*-

"""
Preprocess SQuAD dataset for use.

Original Dataset provided in
https://rajpurkar.github.io/SQuAD-explorer/

"""

from __future__ import print_function

import os
import io
import json

import argparse

PACKET_SIZE = 3 # number of contexts to combine
# TODO: add other kinds of sugars
SUGAR = "By looking at paragraph "


class Corpus(object):
  """
  Corpus Object with basic functionalities.

  Object contains the following:

  title - simple key(int)-value(str) dict{0: 'Super_Bowl_50', 1: 'Warsaw', ...}
  contexts - lol of str [[superbowl1, superbowl2,...],[warsaw1,...],...]
  questions - {'Super_Bowl_50':{0:[q1, q2,...],1:[q1,...]},Warsaw:{...}}
  answers - {'Super_Bowl_50':{0:[[a1,a2,a3],[a1,a2,a3],...],1:[[a1,...],...],Warsaw:{...}}

  """
  def __init__(self):
    self.title = {}
    self.contexts = []
    self.questions = {}
    self.answers = {}

  def getTitle(self, index):
    return unicode(self.title[index]).encode('utf-8')

  def getNumContexts(self, index):
    return len(self.contexts[index])

  def getContext(self, topic_idx, para_idx):
    return self.contexts[topic_idx][para_idx]

  def getLength(self, qual):
    if qual == 'title':
      return len(self.title)
    elif qual == 'contexts':
      return len(self.contexts)
    elif qual == 'questions':
      return len(self.questions)
    elif qual == 'answers':
      return len(self.answers)
    else:
      return ValueError("Corpus object does not have {}".format(qual))


def combine(corpus):
  """Combines contexts, questions, and answers in batches"""
  combcontexts = []
  combquestions = {}
  combanswers = {}
  num_topics = corpus.getLength('contexts')
  for i in xrange(num_topics):
    # print("Combining Contexts/Q&As for {}...".format(corpus.getTitle(i)))
    combcontexts.append([])
    title = corpus.getTitle(i)
    # create a copy to replace corpus.questions
    combquestions[title] = {}
    combanswers[title] = {}
    # identify how many batches to make
    num_packets = corpus.getNumContexts(i) / PACKET_SIZE
    num_leftover = corpus.getNumContexts(i) % PACKET_SIZE
    for idx in xrange(num_packets):
      c_comb , q_comb, a_comb= "", [], []
      for x in xrange(PACKET_SIZE):
        c_comb += corpus.getContext(i, 3*idx+x)
        q_comb.extend(corpus.questions[title][3*idx+x])
        a_comb.append(corpus.answers[title][3*idx+x])
      combcontexts[i].append(c_comb)
      combquestions[title][idx] = q_comb
      combanswers[title][idx] = a_comb
      # utility for adding leftovers in questions
      last_idx = idx

    if num_leftover != 0:
      for l in xrange(num_leftover):
        # add leftover contexts to the last combined context
        combcontexts[i][-1] += corpus.contexts[i][l+1-PACKET_SIZE]
        # leftover questions
        qleftover = corpus.questions[title][3*last_idx+PACKET_SIZE]
        combquestions[title][last_idx].extend(qleftover)
        # leftover answers
        aleftover = corpus.answers[title][3*last_idx+PACKET_SIZE]
        combanswers[title][last_idx].append(aleftover)

  # update corpus object with combined c,q,a
  corpus.contexts = combcontexts
  corpus.questions = combquestions
  corpus.answers = combanswers

  return corpus


def add_question_sugar(corpus):
  assert corpus.getLength('questions') == corpus.getLength('answers')
  num_topics = corpus.getLength('questions')
  for idx in xrange(num_topics):
    qs = corpus.questions[corpus.getTitle(idx)]
    num_leftover = len(qs) % 3
    num_passages = len(qs) - num_leftover
    for i in xrange(num_passages):
      sugared_qs = []
      # add the textual sugar to the beginning of each question
      for q in qs[i]:
        sugared = u''.join((SUGAR, str(i%PACKET_SIZE+1), ", ", q)).encode('utf-8').strip()
        sugared_qs.append(sugared)
      # update corpus object with sugared questions
      corpus.questions[corpus.getTitle(idx)][i] = sugared_qs
    # leftover passages should have appropriate numberings
    for l in xrange(num_leftover):
      sugared_qs = []
      for q in qs[num_passages+l]:
        sugared = u''.join((SUGAR, str(PACKET_SIZE+l+1), ", ", q)).encode('utf-8').strip()
        sugared_qs.append(sugared)
      corpus.questions[corpus.getTitle(idx)][num_passages+l] = sugared_qs
  return corpus


def process(dataset, version):
  """
  SQuAD Dataset
    |-version -> '1.1'
    |-data -> [0]: superbowl, [1]: warsaw, ..., [47]: force
      |-title -> 'Super_Bowl_50'
      |-paragraphs -> [0]: a passage & corresponding Q&As
        |-context -> 'unicode encoded string for context'
        |-qas -> [0]: 1 question and 3 answers
          |-question -> 'what was ...?'
          |-answers -> 3 answers, each formatted as a dict
            |-text -> string version of answer
            |-answer_start -> location to find answer
  """
  assert dataset['version'] == version
  # instantiate corpus class
  corpus = Corpus()
  # iterate through dataset and structure into corpus object
  for idx, topic in enumerate(dataset['data']):
    corpus.title[idx] = topic['title']
    # each topic has a list of contexts
    corpus.contexts.append([])
    title = corpus.getTitle(idx)
    # each topic has a dict: context(key) & list of q(value)
    corpus.questions[title] = {}
    corpus.answers[title] = {}
    for _idx, context_qas in enumerate(topic['paragraphs']):
      context, qas = context_qas['context'], context_qas['qas']
      corpus.contexts[idx].append(context)
      # questions and answers are stored as a list for each passage
      qus = corpus.questions[title][_idx] = []
      ans = corpus.answers[title][_idx] = []
      for _, qa in enumerate(qas):
        qus.append(qa['question'])
        # each is stored as {text, loc}
        ans.append(qa['answers'])

  # add reference to paragraph location for each question
  add_question_sugar(corpus)
  # update corpus with combined contexts
  combine(corpus)

  return corpus


def initiate_process(data_dir, data_type):
  """
  parser = argparse.ArgumentParser()
  parser.add_argument('-d', '--data_type', default='dev')
  parser.add_argument('-v', '--version', default='1.1')
  args = parser.parse_args()
  fpath = os.path.join(pwd, args.data_type + '-v' + args.version +'.json')
  """
  if data_type != 'test.txt':
    if data_type == 'valid':
      data_type = 'dev'

    pwd = os.getcwd()
    fpath = os.path.join(pwd, data_dir)
    fpath = os.path.join(fpath, data_type + '-v1.1.json')

    with io.open(fpath, 'r', encoding='utf-8') as f:
      dataset = json.load(f)
    # corpus object with all preprocessing configured
    corpus = process(dataset, '1.1')

  # temporary testing
  else:
    fpath = os.path.join(pwd, data_dir)
    fpath = os.path.join(fpath, data_type)
    corpus = Corpus()
    with io.open(fpath, 'r', encoding='utf-8') as f:
      corpus.contexts = [[line] for line in f]

  return corpus
