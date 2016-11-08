from __future__ import print_function, division
import os, sys
sys.path.append(os.path.abspath("."))
from utils.lib import O
import numpy as np
import lda
from network.graph import Graph
from sklearn.feature_extraction.text import CountVectorizer
import warnings
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction import text

__author__ = "panzer"

# Considering only tokens of size 3 or more
TOKEN_PATTERN = r"(?u)\b\w\w\w+\b"
ITERATIONS = 1000
ALPHA = None
BETA = None
STOP_WORDS = text.ENGLISH_STOP_WORDS.union(['software', 'engineering'])


class StemTokenizer(object):
  def __init__(self):
    self.stemmer = PorterStemmer()

  def __call__(self, doc):
    return [self.stemmer.stem(t) for t in word_tokenize(doc)]


class StemmedCountVectorizer(CountVectorizer):
  def __init__(self, stemmer, **params):
    super(StemmedCountVectorizer, self).__init__(**params)
    self.stemmer = stemmer

  def build_analyzer(self):
    analyzer = super(StemmedCountVectorizer, self).build_analyzer()
    return lambda doc: (self.stemmer.stem(w) for w in analyzer(doc))


class Document(O):
  def __init__(self, raw=None):
    O.__init__(self)
    self.raw = raw
    self.vector = None
    self.topics_count = None
    self.topics_score = None


class Miner(O):
  def __init__(self, graph):
    O.__init__(self, graph=graph)
    self.vectorizer = None
    self.doc_2_vec = None
    self.documents = None

  def get_documents(self):
    if self.documents: return self.documents
    paper_nodes = self.graph.paper_nodes
    documents = {}
    for paper_id, paper in paper_nodes.items():
      if paper.abstract is not None and paper.abstract != 'None':
        raw = paper.abstract
      else:
        raw = paper.title
      documents[paper_id] = Document(raw)
    self.documents = documents
    return documents

  def vectorize(self, **params):
    if self.vectorizer is None:
      # self.vectorizer = StemmedCountVectorizer(PorterStemmer(), **params)
      self.vectorizer = CountVectorizer(**params)
    if self.doc_2_vec is None:
      docs = [document.raw for _, document in self.get_documents().items()]
      self.doc_2_vec = self.vectorizer.fit_transform(docs).toarray()
      for vector, (paper_id, document) in zip(self.doc_2_vec, self.documents.items()):
        document.vector = vector
        self.documents[paper_id] = document

  def lda(self, n_topics, n_iter=1000, random_state=1, alpha=ALPHA, beta=BETA):
    self.vectorize(stop_words=STOP_WORDS, token_pattern=TOKEN_PATTERN)
    alpha = alpha if alpha else 50/n_topics
    beta = beta if beta else 0.01
    model = lda.LDA(n_topics=n_topics, alpha=alpha, eta=beta, n_iter=n_iter, random_state=random_state)
    model.fit(self.doc_2_vec)
    topics = model.ndz_
    for topic, (paper_id, document) in zip(topics, self.documents.items()):
      document.topics_count = topic
      sum_t = sum(topic)
      sum_t = sum_t if sum_t else 0.00001
      document.topics_score = [t/sum_t for t in topic]
      self.documents[paper_id] = document
    return model, self.vectorizer.get_feature_names()


def cite_graph(file_name):
  return Graph.from_file(file_name)


def write_to_file(file_name, vals):
  with open(file_name, "w") as f:
    for topic_count, ll in vals:
      f.write("%d,%f\n"%(topic_count, ll))


def _lda_authors():
  graph = cite_graph()
  miner = Miner(graph)
  lda_model, vocab = miner.lda(22, n_iter=100, alpha=0.847433736937, beta=0.763774618977)
  authors = graph.get_papers_by_authors()
  for author_id, papers in authors.items():
    docs = []
    for paper_id, _, __ in papers:
      print(miner.documents[paper_id])
      exit()


def _run():
  graph = cite_graph()
  miner = Miner(graph)
  topics = range(5, 105, 5)
  topic_scores = []
  for topic_count in topics:
    print("TOPICS : ", topic_count)
    lda_model, vocab = miner.lda(topic_count, n_iter=ITERATIONS)
    topic_scores.append((topic_count, lda_model.loglikelihood()))
  write_to_file("scores_50.csv", topic_scores)
  # topic_word = lda_model.topic_word_
  # n_top_words = 8
  # for i, topic_dist in enumerate(topic_word):
  #   topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
  #   print('Topic {}: {}'.format(i, ' '.join(topic_words)))


if __name__ == "__main__":
  _lda_authors()

