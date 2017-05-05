from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

__author__ = "bigfatnoob"

from utils.lib import O, Memoized
from network.mine import cite_graph
import cPickle as cPkl
from classify import predict
from sklearn.feature_extraction import text
from sklearn.model_selection import StratifiedKFold
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import tensorflow as tf
import math
import random


RANDOM_STATE = 1
GRAPH_CSV = "data/citemap_v8.csv"
ALL = 'all'
STOP_WORDS = text.ENGLISH_STOP_WORDS.union(['software', 'engineering', 'paper', 'study', 'based',
                                            'results', 'approach', 'case', 'workshop', 'international', 'research',
                                            'conference', 'introduction', 'editors', 'article', 'issue', 'month',
                                            'copyright', 'special', 'used', 'using', 'use', 'studies', 'review',
                                            'editorial', 'report', 'book', 'ieee', 'published', 'science', 'column',
                                            'author', 'proposed', 'icse', 'article', 'year', 'articles', 'page', '2000',
                                            '2004', 'papers', 'computer', 'held', 'editor'])
TOKEN_PATTERN = r"(?u)\b\w\w\w+\b"
VOCAB_SIZE = 30000
BATCH_SIZE = 128
EMBEDDING_SIZE = 64
NEGATIVE_SAMPLE_SIZE = 5

def retrieve_graph(graph_file=GRAPH_CSV, from_cache=True):
  cached = 'cache/graph.pkl'
  if os.path.isfile(cached) and from_cache:
    with open(cached) as f:
      graph = cPkl.load(f)
  else:
    graph = cite_graph(graph_file)
    with open(cached, 'wb') as f:
      cPkl.dump(graph, f, cPkl.HIGHEST_PROTOCOL)
  return graph



@Memoized
def retrieve_vocabulary(min_tfidf_score=0.1, from_cache=True):
  cached = 'cache/vocabulary.pkl'
  if os.path.isfile(cached) and from_cache:
    with open(cached) as f:
      vocabulary = cPkl.load(f)[:VOCAB_SIZE - 1].tolist()
  else:
    graph = retrieve_graph()
    vectorizer = TfidfVectorizer(analyzer=analyzer())
    papers, groups = predict.get_papers_and_groups(graph, is_independent=True)
    documents = [paper.raw for paper in papers]
    tfidf_matrix = vectorizer.fit_transform(documents).toarray()
    tfidf_matrix[tfidf_matrix < min_tfidf_score] = 0
    tfidf_means = np.mean(tfidf_matrix, axis=0)
    sorted_indices = np.argsort(tfidf_means)[::-1][:]
    vocabulary = np.array(vectorizer.get_feature_names())[sorted_indices]
    with open(cached, "wb") as f:
      cPkl.dump(vocabulary, f, cPkl.HIGHEST_PROTOCOL)
    vocabulary = vocabulary[:VOCAB_SIZE - 1].tolist()
  vocab_map, reverse_vocab_map = {}, {}
  for i, v in enumerate(vocabulary):
    vocab_map[v] = i
    reverse_vocab_map[i] = v
  vocab_map["UNK"] = VOCAB_SIZE - 1
  reverse_vocab_map[VOCAB_SIZE - 1] = "UNK"
  return vocab_map, reverse_vocab_map

VOCABULARY, REVERSE_VOCABULARY = retrieve_vocabulary()
VOCABULARY_WORDS = VOCABULARY.keys()
VOCABULARY_INDICES = REVERSE_VOCABULARY.keys()

def split(dependent, independent, n_folds):
  skf = StratifiedKFold(n_splits=n_folds, random_state=RANDOM_STATE)
  for train_indices, test_indices in skf.split(dependent, independent):
    train_x = dependent[train_indices]
    train_y = independent[train_indices]
    test_x = dependent[test_indices]
    test_y = independent[test_indices]
    yield train_x, train_y, test_x, test_y


def pre_processor(lowercase=True):
  if lowercase:
    return lambda doc: doc.lower()
  else:
    return lambda doc: doc


def tokenizer(token_pattern=TOKEN_PATTERN):
  t_p = re.compile(token_pattern)
  return lambda doc: t_p.findall(doc)


def stop_words_cleaner(tokens, stop_words=None):
  if stop_words is not None:
    tokens = [token for token in tokens if token not in stop_words]
  return tokens


def analyzer():
  pre_process = pre_processor()
  tokenize = tokenizer()
  return lambda doc: stop_words_cleaner(tokenize(pre_process(doc)), STOP_WORDS)


class Doc(O):
  def __init__(self, _id, tokens, label):
    O.__init__(self, id=_id, tokens=tokens, label=label)


class Link(O):
  SEPARATOR = "$|$"

  def __init__(self, source, target, weight=None):
    O.__init__(self, source=source, target=target)
    self.weight = 1 if weight is None else weight
    # self.id = Link.make_key(source, target)

  def increment(self):
    self.weight += 1

  @staticmethod
  def make_key(source, target):
    return "%s%s%s" % (source, Link.SEPARATOR, target)


def citation_map(graph):
  cite_edges = graph.cite_edges
  cite_map = {}
  for edge in cite_edges.values():
    references = cite_map.get(edge.source, [])
    cite_map[edge.source] = references + [edge.target]
  return cite_map


def add_tokens(tokens, node_map):
  for token in tokens:
    if token not in node_map:
      node_map[token] = len(node_map)


def add_edge(source, target, edge_map):
  # key = Link.make_key(source, target)
  # if key in edge_map:
  #   edge_map[key].increment()
  # else:
  #   edge_map[key] = Link(source, target)
  # edge_map.update({(source, target): 1})
  edge_map[source][target] += 1


def make_self_edges(tokens, edge_map):
  for i in range(len(tokens) - 1):
    # t_i = tokens[i]
    t_i = VOCABULARY[tokens[i]]
    for j in range(i + 1, len(tokens)):
      # t_j = tokens[j]
      t_j = VOCABULARY[tokens[j]]
      add_edge(t_i, t_j, edge_map)
      add_edge(t_j, t_i, edge_map)


def make_edges(source_tokens, target_tokens, edge_map):
  for s in source_tokens:
    t_i = VOCABULARY[s]
    for t in target_tokens:
      t_j = VOCABULARY[t]
      add_edge(t_i, t_j, edge_map)


def build_graph(index, train_x, train_y, cite_map, use_references=True, from_cache=True):
  if use_references:
    cached = "cache/graphs/%d_ref.pkl" % index
  else:
    cached = "cache/graphs/%d.pkl" % index
  if os.path.isfile(cached) and from_cache:
    with open(cached) as f:
      return cPkl.load(f)
  analyze = analyzer()
  doc_map = {}
  for x, y in zip(train_x, train_y):
    tokens = set(analyze(x.raw)).intersection(VOCABULARY_WORDS)
    # add_tokens(tokens, nodes)
    doc = Doc(x.id, tokens, y)
    doc_map[x.id] = doc
  edges = np.zeros((VOCAB_SIZE, VOCAB_SIZE), dtype=np.int16)
  for i, x in enumerate(train_x):
    if i % 1000 == 0:
      print(i)
    tokens = list(doc_map[x.id].tokens)
    make_self_edges(tokens, edges)
    if use_references:
      references = cite_map.get(x.id, [])
      for reference in references:
        if reference not in doc_map:  # belongs to test set
          continue
        make_edges(tokens, list(doc_map[reference].tokens), edges)
  word_network = O(doc_map=doc_map, edges=edges)
  with open(cached, "wb") as f:
    cPkl.dump(word_network, f, cPkl.HIGHEST_PROTOCOL)
  return word_network


data_index = 0

def generate_edge_batch(edges):
  """
  :param edges: np.ndarray(shape=[VOCAB_SIZE, VOCAB_SIZE])
  :return:
  """
  global data_index
  batch = np.ndarray(shape=[BATCH_SIZE, 3], dtype=np.int32)
  length = edges.shape[0]
  total_edges = length * length
  for index in range(BATCH_SIZE):
    row = data_index // length
    col = data_index % length
    weight = edges[row][col]
    batch[index][0] = row
    batch[index][1] = col
    batch[index][2] = weight
    data_index = (data_index + 1) % total_edges
  return batch


def make_negative_samples_map(edges, index, use_references, deg=0.75):
  if use_references:
    file_name = "cache/graphs/neg_samples/%d_ref.pkl" % index
  else:
    file_name = "cache/graphs/neg_samples/%d.pkl" % index
  if os.path.isfile(file_name):
    with open(file_name) as f:
      return cPkl.load(f)
  num_neg_samples = 1e8
  degrees = (edges != 0).sum(1)
  norm = sum([math.pow(node_degree, deg) for node_degree in degrees])
  num_nodes = edges.shape[0]
  p = 0
  i = 0
  neg_samples = np.zeros(int(num_neg_samples), dtype=np.int32)
  for j in range(num_nodes):
    if j % 10000 == 0:
      print(j)
    if i >= num_neg_samples:
      break
    p += math.pow(degrees[j], deg) / norm
    while i < num_neg_samples and i / num_neg_samples < p:
      neg_samples[i] = j
      i += 1
  print("Done :", index)
  with open(file_name, "wb") as f:
    cPkl.dump(neg_samples, f, cPkl.HIGHEST_PROTOCOL)
  return neg_samples


def line(total_batches, initial_learn_rate=0.025):
  graph = tf.Graph()
  with graph.as_default(), tf.device("/cpu:0"):


    # Input data
    train_dataset = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 3])
    counter = tf.placeholder(tf.int32, shape=[1])

    # Variables
    all_projections = tf.Variable(tf.random_uniform([VOCAB_SIZE, EMBEDDING_SIZE], -1.0, 1.0))
    all_contexts = tf.Variable(tf.random_uniform([VOCAB_SIZE, EMBEDDING_SIZE], -1.0, 1.0))
    # all_projections = tf.nn.l2_normalize(all_projections, dim=1)
    # all_contexts = tf.nn.l2_normalize(all_contexts, dim=1)

    source_projections = tf.nn.embedding_lookup(all_projections, train_dataset[:, 0])
    source_projections = tf.nn.l2_normalize(source_projections, dim=1)
    # source_contexts = tf.nn.embedding_lookup(all_contexts, train_dataset[:, 1])
    target_contexts = tf.nn.embedding_lookup(all_projections, train_dataset[:, 1])
    # source_projections = tf.Print(source_projections, [source_projections], message="Source Projections")
    # target_contexts = tf.Print(target_contexts, [target_contexts], message="Target Contexts")

    weights = tf.to_float(tf.convert_to_tensor(train_dataset[:, 2], dtype=tf.int32))
    den = tf.reduce_sum(tf.exp(tf.matmul(source_projections, all_contexts, transpose_b=True)), axis=1)
    num = tf.exp(tf.reduce_sum(tf.multiply(source_projections, target_contexts), axis=1))
    # num = tf.Print(num, [num], message="Num")
    # den = tf.Print(den, [den], message="Den")

    learn_rate = initial_learn_rate * (1 - counter[0] / total_batches)
    divs = tf.div(num, den)
    # divs = tf.Print(divs, [divs], message="Divs")
    logs = tf.log(tf.sigmoid(divs))
    # logs = tf.Print(logs, [logs], message="Logs")
    objective = -1 * tf.reduce_sum(tf.multiply(weights, logs))
    # loss = 0.01 * (tf.nn.l2_loss(all_projections) + tf.nn.l2_loss(all_contexts))
    # objective = objective + loss
    optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(objective)
  return graph, train_dataset, counter, optimizer, all_projections, all_contexts, objective


def line_negative(total_batches, sample_size=NEGATIVE_SAMPLE_SIZE, initial_learn_rate=0.025):
  graph = tf.Graph()
  with graph.as_default(), tf.device("/cpu:0"):


    # Input data
    train_dataset = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 3])
    negative_samples = tf.placeholder(tf.int32, shape=[BATCH_SIZE * sample_size])
    positive_samples = tf.placeholder(tf.int32, shape=[BATCH_SIZE * sample_size])
    counter = tf.placeholder(tf.int32, shape=[1])

    # Variables
    all_projections = tf.Variable(tf.random_uniform([VOCAB_SIZE, EMBEDDING_SIZE], -1.0, 1.0))
    all_contexts = tf.Variable(tf.random_uniform([VOCAB_SIZE, EMBEDDING_SIZE], -1.0, 1.0))

    source_projections = tf.nn.embedding_lookup(all_projections, train_dataset[:, 0])
    target_contexts = tf.nn.embedding_lookup(all_contexts, train_dataset[:, 1])
    # target_projections = tf.nn.embedding_lookup(all_projections, train_dataset[:, 1])
    neg_samples_lookup = tf.nn.embedding_lookup(all_projections, negative_samples)
    pos_samples_lookup = tf.nn.embedding_lookup(all_projections, positive_samples)

    # source_projections = tf.Print(source_projections, [source_projections], message="Source Projections")
    # target_contexts = tf.Print(target_contexts, [target_contexts], message="Target Contexts")

    negative_scores = tf.log(tf.sigmoid(-1 * tf.reduce_sum(tf.multiply(neg_samples_lookup, pos_samples_lookup), axis=1)))
    negative_scores = tf.reduce_sum(tf.reshape(negative_scores, shape=[BATCH_SIZE, sample_size]), axis=1)

    weights = tf.to_float(tf.convert_to_tensor(train_dataset[:, 2], dtype=tf.int32))
    positive_scores = tf.log(tf.sigmoid(tf.reduce_sum(tf.multiply(source_projections, target_contexts), axis=1)))
    total_scores = positive_scores + negative_scores

    learn_rate = initial_learn_rate * (1 - counter[0] / total_batches)
    objective = -1 * tf.reduce_sum(total_scores)
    # objective = -1 * tf.reduce_sum(tf.multiply(weights, total_scores))
    # loss = 0.01 * (tf.nn.l2_loss(all_projections) + tf.nn.l2_loss(all_contexts))
    # objective = objective + loss
    optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(objective)
  return graph, train_dataset, negative_samples, positive_samples, counter, optimizer, all_projections, all_contexts, objective


def train_words(word_network, log_factor=1):
  total_edges = word_network.edges.shape[0] * word_network.edges.shape[1]
  total_batches = 4 * total_edges // BATCH_SIZE
  graph, train_dataset, counter, optimizer, all_projections, all_contexts, objective = line(total_batches)
  global data_index
  data_index = 0
  projections, contexts = None, None
  # session = tf.InteractiveSession(graph=graph)
  with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    for gen in range(total_batches):
      batch = generate_edge_batch(word_network.edges)
      feed_dict = {train_dataset: batch, counter: [gen]}
      _, projections, contexts, obj = session.run([optimizer, all_projections, all_contexts, objective],
                                                        feed_dict=feed_dict)
      if gen % log_factor == 0:
        print("Batch : %d / %d, Loss: %f" % (gen, total_batches, obj))
  # session.close()
  return projections, contexts


def train__words_negative(word_network, negative_samples_list, log_factor=1):
  total_edges = word_network.edges.shape[0] * word_network.edges.shape[1]
  total_batches = 4 * total_edges // BATCH_SIZE
  graph, train_dataset, negative_samples, positive_samples, counter, optimizer, all_projections, all_contexts, objective = \
      line_negative(total_batches)
  global data_index
  data_index = 0
  projections, contexts = None, None
  # session = tf.InteractiveSession(graph=graph)
  with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    for gen in range(total_batches):
      batch = generate_edge_batch(word_network.edges)
      sources = batch[:, 0]
      neg_samples = np.ndarray(shape=[BATCH_SIZE * NEGATIVE_SAMPLE_SIZE], dtype=np.int32)
      pos_samples = np.ndarray(shape=[BATCH_SIZE * NEGATIVE_SAMPLE_SIZE], dtype=np.int32)
      for index, source in enumerate(sources):
        k = 0
        while k < NEGATIVE_SAMPLE_SIZE:
          negative_sample = None
          while negative_sample is None or negative_sample == source:
            negative_sample = negative_samples_list[random.randint(0, negative_samples_list.shape[0])]
            neg_samples[index * NEGATIVE_SAMPLE_SIZE + k] = negative_sample
            pos_samples[index * NEGATIVE_SAMPLE_SIZE + k] = source
          k += 1
      feed_dict = {train_dataset: batch, negative_samples: neg_samples, positive_samples: pos_samples, counter: [gen]}
      _, projections, contexts, obj = session.run([optimizer, all_projections, all_contexts, objective],
                                                        feed_dict=feed_dict)
      if gen % log_factor == 0:
        print("Batch : %d / %d, Loss: %f" % (gen, total_batches, obj))
  # session.close()
  return projections, contexts


def runner(use_references, use_neg_samples):
  graph = retrieve_graph()
  cite_map = citation_map(graph)
  papers, groups = predict.get_papers_and_groups(graph, is_independent=True)
  for index, (train_x, train_y, test_x, test_y) in enumerate(split(papers, groups, 5)):
    word_network = build_graph(index, train_x, train_y, cite_map, use_references)
    if use_neg_samples:
      negative_samples = make_negative_samples_map(word_network.edges, index, use_references)
      projections, contexts = train__words_negative(word_network, negative_samples)
    else:
      projections, contexts = train_words(word_network)
    exit()

    # dump = {"projections": projections, "contexts": contexts}
    # if use_references:
    #   file_name = "cache/graphs/results/%d_ref.pkl" % index
    # else:
    #   file_name = "cache/graphs/results/%d.pkl" % index
    # with open(file_name, 'wb') as f:
    #   cPkl.dump(dump, f, cPkl.HIGHEST_PROTOCOL)


if __name__ == "__main__":
  # runner(False)
  # with open("cache/graphs/results/0_ref.pkl") as f:
  #   dump = cPkl.load(f)
  #   # print(dump["projections"])
  #   print(np.argwhere(np.isnan(dump["projections"])))
  runner(False, True)
