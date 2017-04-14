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
from collections import Counter

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

  def __init__(self, source, target):
    O.__init__(self, source=source, target=target)
    self.weight = 1
    self.id = Link.make_key(source, target)

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
  edge_map.update({(source, target): 1})


def make_self_edges(tokens, edge_map):
  for i in range(len(tokens) - 1):
    t_i = tokens[i]
    for j in range(i + 1, len(tokens)):
      add_edge(t_i, tokens[j], edge_map)
      add_edge(tokens[j], t_i, edge_map)


def make_edges(source_tokens, target_tokens, edge_map):
  for s in source_tokens:
    for t in target_tokens:
      add_edge(s, t, edge_map)


def build_graph(index, train_x, train_y, cite_map, use_references=True, from_cache=True):
  vocabulary = retrieve_vocabulary()
  if use_references:
    cached = "cache/graphs/%d_ref.pkl" % index
  else:
    cached = "cache/graphs/%d.pkl" % index
  if os.path.isfile(cached) and from_cache:
    with open(cached) as f:
      return cPkl.load(f)
  analyze = analyzer()
  doc_map = {}
  nodes = set(["UNK"])
  edges = Counter()
  for x, y in zip(train_x, train_y):
    tokens = set(analyze(x.raw)).intersection(vocabulary)
    nodes = nodes.union(tokens)
    doc = Doc(x.id, tokens, y)
    doc_map[x.id] = doc
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
  word_network = O(doc_map=doc_map, nodes=nodes, edges=edges)
  with open(cached, "wb") as f:
    cPkl.dump(word_network, f, cPkl.HIGHEST_PROTOCOL)
  return word_network


@Memoized
def retrieve_vocabulary(min_tfidf_score=0.1, from_cache=True):
  cached = 'cache/vocabulary.pkl'
  if os.path.isfile(cached) and from_cache:
    with open(cached) as f:
      vocabulary = set(cPkl.load(f)[:VOCAB_SIZE].tolist())
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
    vocabulary = set(vocabulary[:VOCAB_SIZE].tolist())
  return vocabulary


def runner(use_references):
  graph = retrieve_graph()
  cite_map = citation_map(graph)
  papers, groups = predict.get_papers_and_groups(graph, is_independent=True)
  for index, (train_x, train_y, test_x, test_y) in enumerate(split(papers, groups, 5)):
    word_network = build_graph(index, train_x, train_y, cite_map, use_references)
    print(len(word_network.doc_map))
    print(len(word_network.nodes))
    print(len(word_network.edges))


if __name__ == "__main__":
  runner(True)
