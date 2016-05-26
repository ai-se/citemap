from __future__ import print_function, division
import os, sys
sys.path.append(os.path.abspath("."))
from utils.lib import O
import numpy as np
import lda
from network.graph import Graph
from sklearn.feature_extraction.text import CountVectorizer
import warnings

__author__ = "panzer"

# Considering only tokens of size 3 or more
TOKEN_PATTERN = r"(?u)\b\w\w\w+\b"
ITERATIONS = 1000

class Miner(O):
  def __init__(self, graph):
    O.__init__(self, graph=graph)
    self.vectorizer = None
    self.doc_2_vec = None

  def get_documents(self):
    paper_nodes = self.graph.paper_nodes
    documents = {}
    for node_id, paper in paper_nodes.items():
      documents[node_id] = paper.abstract if paper.abstract else paper.title
    return documents

  def vectorize(self, **params):
    if self.vectorizer is None:
      self.vectorizer = CountVectorizer(**params)
    if self.doc_2_vec is None:
      docs = self.get_documents().values()
      self.doc_2_vec = self.vectorizer.fit_transform(docs).toarray()

  def lda(self, n_topics, n_iter=1000, random_state=1):
    self.vectorize(stop_words='english', token_pattern=TOKEN_PATTERN)
    model = lda.LDA(n_topics=n_topics, alpha=50/n_topics, n_iter=n_iter, random_state=random_state)
    model.fit(self.doc_2_vec)
    return model, self.vectorizer.get_feature_names()


def cite_graph():
  return Graph.from_file("citemap.csv")

def write_to_file(file_name, vals):
  with open(file_name, "w") as f:
    for topic_count, ll in vals:
      f.write("%d,%f\n"%(topic_count, ll))




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
  _run()

