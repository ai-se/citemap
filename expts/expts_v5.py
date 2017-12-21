from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

__author__ = "bigfatnoob"

from utils.lib import O, Memoized, file_exists
import numpy as np
from network.mine import cite_graph, Miner
import matplotlib.pyplot as plt
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
import cPickle as cPkl
from sklearn.externals import joblib
import lda
from utils import perplexity
import logging
logging.getLogger('lda').setLevel(logging.ERROR)

GRAPH_CSV = "data/citemap_v10.csv"

# For 11 TOPICS
ALPHA = 0.22359
BETA = 0.53915
ITERATIONS = 100

THE = O()
THE.permitted = "all"  # conference/journal/all
THE.version = "v5"
THE.use_numeric = True
THE.random_state = 0

STOP_WORDS = text.ENGLISH_STOP_WORDS.union(['software', 'engineering', 'paper', 'study', 'based',
                                            'results', 'approach', 'case', 'workshop', 'international', 'research',
                                            'conference', 'introduction', 'editors', 'article', 'issue', 'month',
                                            'copyright', 'special', 'used', 'using', 'use', 'studies', 'review',
                                            'editorial', 'report', 'book', 'ieee', 'published', 'science', 'column',
                                            'author', 'proposed', 'icse', 'article', 'year', 'articles', 'page', '2000',
                                            '2004', 'papers', 'computer', 'held', 'editor'])
TOKEN_PATTERN = r"(?u)\b\w\w\w+\b"


def mkdir(directory):
  """
  Implements the "mkdir" linux function
  :param directory:
  :return:
  """
  if directory and not os.path.exists(directory):
    os.makedirs(directory)


def get_n_topics():
  return 11


def get_topics():
  if THE.use_numeric:
    return ["Topic-%2d" % i for i in range(get_n_topics())]
  # if THE.permitted == "all":
  #   return TOPICS_ALL


@Memoized
def retrieve_graph_lda_data():
  """
  Fetch stored metadata
  :return:
  """
  graph_file = 'cache/%s/%s/graph.pkl' % (THE.version, THE.permitted)
  vectorizer_file = 'cache/%s/%s/vectorizer.pkl' % (THE.version, THE.permitted)
  doc_2_vec_file = 'cache/%s/%s/doc_2_vec.pkl' % (THE.version, THE.permitted)
  documents_file = 'cache/%s/%s/documents.pkl' % (THE.version, THE.permitted)
  lda_model_file = 'cache/%s/%s/lda_model.pkl' % (THE.version, THE.permitted)
  vocabulary_file = 'cache/%s/%s/vocabulary.pkl' % (THE.version, THE.permitted)
  if os.path.isfile(graph_file) and os.path.isfile(vectorizer_file) \
          and os.path.isfile(doc_2_vec_file) and os.path.isfile(documents_file) \
          and os.path.isfile(lda_model_file) and os.path.isfile(vocabulary_file):
    with open(graph_file) as f:
      graph = cPkl.load(f)
    miner = Miner(graph, permitted=THE.permitted)
    with open(vectorizer_file) as f:
      miner.vectorizer = cPkl.load(f)
    with open(doc_2_vec_file) as f:
      miner.doc_2_vec = joblib.load(f)
    with open(documents_file) as f:
      miner.documents = cPkl.load(f)
    with open(lda_model_file) as f:
      lda_model = cPkl.load(f)
    with open(vocabulary_file) as f:
      vocab = cPkl.load(f)
  else:
    miner, graph, lda_model, vocab = store_graph_lda_data()
  return miner, graph, lda_model, vocab


def load_graph():
  graph_file = 'cache/%s/%s/graph.pkl' % (THE.version, THE.permitted)
  if os.path.isfile(graph_file):
    with open(graph_file) as f:
      graph = cPkl.load(f)
    return graph
  else:
    return None



def store_graph_lda_data():
  miner, graph, lda_model, vocab = get_graph_lda_data()
  folder_name = 'cache/%s/%s' % (THE.version, THE.permitted)
  mkdir(folder_name)
  with open('cache/%s/%s/graph.pkl' % (THE.version, THE.permitted), 'wb') as f:
    cPkl.dump(graph, f, cPkl.HIGHEST_PROTOCOL)
  with open('cache/%s/%s/vectorizer.pkl' % (THE.version, THE.permitted), 'wb') as f:
    cPkl.dump(miner.vectorizer, f, cPkl.HIGHEST_PROTOCOL)
  with open('cache/%s/%s/doc_2_vec.pkl' % (THE.version, THE.permitted), 'wb') as f:
    joblib.dump(miner.doc_2_vec, f)
  with open('cache/%s/%s/documents.pkl' % (THE.version, THE.permitted), 'wb') as f:
    cPkl.dump(miner.documents, f, cPkl.HIGHEST_PROTOCOL)
  with open('cache/%s/%s/lda_model.pkl' % (THE.version, THE.permitted), 'wb') as f:
    cPkl.dump(lda_model, f, cPkl.HIGHEST_PROTOCOL)
  with open('cache/%s/%s/vocabulary.pkl' % (THE.version, THE.permitted), 'wb') as f:
    cPkl.dump(vocab, f, cPkl.HIGHEST_PROTOCOL)
  return miner, graph, lda_model, vocab


def retrieve_graph():
  graph_file = 'cache/%s/%s/graph.pkl' % (THE.version, THE.permitted)
  if os.path.isfile(graph_file):
    with open(graph_file) as f:
      graph = cPkl.load(f)
  else:
    graph = cite_graph(GRAPH_CSV)
    with open(graph_file, 'wb') as f:
      cPkl.dump(graph, f, cPkl.HIGHEST_PROTOCOL)
  return graph


@Memoized
def get_graph_lda_data():
  graph = cite_graph(GRAPH_CSV)
  miner = Miner(graph, THE.permitted)
  lda_model, vocab = miner.lda(get_n_topics(), n_iter=ITERATIONS, alpha=ALPHA, beta=BETA, stop_words=STOP_WORDS)
  return miner, graph, lda_model, vocab


def report(lda_model, vocab, fig_name="topic_dist", n_top_words=10, plot_terms=50):
  fig = plt.figure()
  x_axis = range(1, plot_terms + 1)
  legends = []
  for index, topic_dist in enumerate(lda_model.topic_word_):
    sorted_dist = np.sort(topic_dist)
    # scores = sorted_dist[:-(n_top_words + 1):-1]
    plot_scores = sorted_dist[:-(plot_terms + 1):-1]
    plot_scores = np.log(plot_scores)
    plt.plot(x_axis, plot_scores)
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
    legends.append("Topic %d" % index)
    # legends.append(TOPICS_ALL[index])
    print('%2d : %16s : %s' % (index, get_topics()[index].upper(), ', '.join(topic_words)))
  plt.legend(legends, loc='upper right')
  plt.title(fig_name)
  plt.xlabel("Term Index")
  plt.ylabel("Log. Word Score")
  mkdir("figs/%s/%s" % (THE.version, THE.permitted))
  plt.savefig("figs/%s/%s/%s.png" % (THE.version, THE.permitted, fig_name))
  fig.clf()


def reporter():
  print("TOPIC REPORTS for %s" % THE.permitted)
  miner, graph, lda_model, vocab = retrieve_graph_lda_data()
  report(lda_model, vocab, n_top_words=12)


def get_documents(graph):
  docs = []
  for paper_id, paper in graph.paper_nodes.items():
    if paper.abstract is not None and paper.abstract != 'None':
      raw = paper.abstract
    else:
      raw = paper.title
    docs.append(raw)
  return docs


def split_perplexity(splits):
  result_file = "cache/%s/%s/perplexity.pkl" % (THE.version, THE.permitted)

  def get_cached_result():
    if os.path.isfile(result_file):
      with open(result_file) as f:
        cached_result = cPkl.load(f)
    else:
      cached_result = {}
    return cached_result

  def save_cached_result(cached_result):
    with open(result_file, "wb") as f:
      cPkl.dump(cached_result, f, cPkl.HIGHEST_PROTOCOL)

  topics = range(2, 51, 1)
  graph = retrieve_graph()
  raw_docs = np.array([doc for doc in get_documents(graph)])
  k_folds = KFold(n_splits=splits, random_state=THE.random_state, shuffle=True)
  results = get_cached_result()
  split = 0
  for train_index, test_index in k_folds.split(raw_docs):
    print("## SPLIT %d" % split)
    train_docs = raw_docs[train_index]
    test_docs = raw_docs[test_index]
    vectorizer = CountVectorizer(stop_words=STOP_WORDS, token_pattern=TOKEN_PATTERN)
    train_doc_vec = vectorizer.fit_transform(train_docs).toarray()
    test_doc_vec = vectorizer.transform(test_docs).toarray()
    for topic in topics:
      print("#### N_Topics = %d" % topic)
      if topic in results and split in results[topic]:
        print("Split %d exists for n_topics = %d" % (split, topic))
        continue
      lda_model = lda.LDA(n_topics=topic, alpha=0.1, eta=0.01, n_iter=100)
      lda_model.fit(train_doc_vec)
      log_perplexity = perplexity.log_perplexity(lda_model, test_doc_vec)
      # log_perplexity = np.random.random()
      topic_results = results.get(topic, {})
      topic_results[split] = log_perplexity
      results[topic] = topic_results
      save_cached_result(results)
      # print(log_perplexity)
    split += 1
  print(get_cached_result())


def _main():
  reporter()


if __name__ == "__main__":
  # reporter()
  split_perplexity(10)
