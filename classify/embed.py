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
BATCH_SIZE = 1
EMBEDDING_SIZE = 128


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


def line(total_batches, initial_learn_rate=0.025):
  graph = tf.Graph()
  with graph.as_default(), tf.device("/cpu:0"):


    # Input data
    train_dataset = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 3])
    counter = tf.placeholder(tf.int32, shape=[1])

    # Variables
    all_projections = tf.Variable(tf.random_uniform([VOCAB_SIZE, EMBEDDING_SIZE], -1.0, 1.0))
    all_contexts = tf.Variable(tf.random_uniform([VOCAB_SIZE, EMBEDDING_SIZE], -1.0, 1.0))
    all_projections = tf.nn.l2_normalize(all_projections, dim=1)
    all_contexts = tf.nn.l2_normalize(all_contexts, dim=1)

    source_projections = tf.nn.embedding_lookup(all_projections, train_dataset[:, 0])
    source_contexts = tf.nn.embedding_lookup(all_contexts, train_dataset[:, 1])
    target_contexts = tf.nn.embedding_lookup(all_projections, train_dataset[:, 1])

    weights = tf.to_float(tf.convert_to_tensor(train_dataset[:, 2], dtype=tf.int32))
    den = tf.reduce_sum(tf.exp(tf.matmul(source_projections, all_contexts, transpose_b=True)), axis=1)
    num = tf.exp(tf.reduce_sum(tf.multiply(source_projections, target_contexts), axis=1))

    learn_rate = initial_learn_rate * (1 - counter[0] / total_batches)
    divs = tf.div(num, den)
    logs = tf.log(divs)
    objective = -1 * tf.reduce_sum(tf.multiply(weights, logs))
    loss = 0.01 * (tf.nn.l2_loss(all_projections) + tf.nn.l2_loss(all_contexts))
    objective = objective + loss
    optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(objective)

  return graph, train_dataset, counter, optimizer, all_projections, all_contexts


def train_words(word_network, log_factor=10000):
  total_edges = np.prod(word_network.edges.shape)
  total_batches = total_edges // BATCH_SIZE
  graph, train_dataset, counter, optimizer, all_projections, all_contexts = line(total_batches)
  global data_index
  data_index = 0
  projections, contexts = None, None
  session = tf.InteractiveSession(graph=graph)
  # with tf.InteractiveSession(graph=graph) as session:
  tf.global_variables_initializer().run()
  for gen in range(total_batches)[:10]:
    batch = generate_edge_batch(word_network.edges)
    if gen % log_factor == 0:
      print("Batch : %d" % gen)
    feed_dict = {train_dataset: batch, counter: [gen]}
    _, projections, contexts = session.run([optimizer, all_projections, all_contexts], feed_dict=feed_dict)
  session.close()
  return projections, contexts


def runner(use_references):
  graph = retrieve_graph()
  cite_map = citation_map(graph)
  papers, groups = predict.get_papers_and_groups(graph, is_independent=True)
  for index, (train_x, train_y, test_x, test_y) in enumerate(split(papers, groups, 5)):
    word_network = build_graph(index, train_x, train_y, cite_map, use_references)
    projections, contexts = train_words(word_network)
    dump = {"projections": projections, "contexts": contexts}
    if use_references:
      file_name = "cache/graphs/results/%d_ref.pkl" % index
    else:
      file_name = "cache/graphs/results/%d.pkl" % index
    with open(file_name, 'wb') as f:
      cPkl.dump(dump, f, cPkl.HIGHEST_PROTOCOL)
    exit()


if __name__ == "__main__":
  # runner(False)
  # with open("cache/graphs/results/0_ref.pkl") as f:
  #   dump = cPkl.load(f)
  #   # print(dump["projections"])
  #   print(np.argwhere(np.isnan(dump["projections"])))
  runner(True)
