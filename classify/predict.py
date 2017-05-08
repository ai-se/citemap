from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True
from network.mine import Miner, cite_graph
from sklearn.feature_extraction import text
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
import sklearn.metrics as sk_metrics
import numpy as np
import lda
import matplotlib.pyplot as plt
import pandas as pd
from utils.lib import O
import cPickle as cPkl
import re

GRAPH_CSV = "data/citemap_v8.csv"
CLASSIFY_CSV = "classify/data.csv"
ACCEPTED = 0
REJECTED = 1
RANDOM_STATE = 1
"""
Short Name Start Group
15  RE     A1
32  REJ    A2

18  ESEM   B1
25  ESE    B2
24  S/W    B3
30  SMR    B4
35  SQJ    B5

 5  MSR    C1
 2  ICSM   C2
 3  WCRE   C3
 8  ICPC   C4

10  SCAM   D1
 4  CSMR   D2
12  SANER  D3
 1  ICSE   D4
 9  FSE    D5
11  ASE    D6

16  ISSTA  E1
17  ICST   E2
36  STVR   E3

 6  GPCE   F1
 7  FASE   F2

29  ISSE   G1
27  IJSEKE G2
31  NOTES  G3
33  TOSEM  G4
23  TSE    G5
34  ASEJ   G6

13  SSBSE  H1
20  JSS    H2
26  SPE    H3
"""
GROUP_CONFERENCE_MAP = {
    15: 'A', 32: 'A',
    18: 'B', 25: 'B', 24: 'B', 30: 'B', 35: 'B',
    5: 'C', 2: 'C', 3: 'C', 8: 'C',
    10: 'D', 4: 'D', 12: 'D', 1: 'D', 9: 'D', 11: 'D',
    16: 'E', 17: 'E', 36: 'E',
    6: 'F', 7: 'F',
    29: 'G', 27: 'G', 31: 'G', 33: 'G', 23: 'G', 34: 'G',
    13: 'H', 20: 'H', 26: 'H'
}


# For 11 TOPICS
N_TOPICS = 11
ALPHA = 0.22359
BETA = 0.53915
ITERATIONS = 100
TOPICS = ["Design", "Testing", "Modelling", "Mobile", "Energy", "Defects",
          "SourceCode", "WebApps", "Configuration", "Developer", "Mining"]
TOPIC_THRESHOLD = 3
DELIMITER = '$|$'
STOP_WORDS = text.ENGLISH_STOP_WORDS.union(['software', 'engineering', 'paper', 'study', 'based',
                                            'results', 'approach', 'case', 'workshop', 'international', 'research',
                                            'conference', 'introduction', 'editors', 'article', 'issue', 'month',
                                            'copyright', 'special', 'used', 'using', 'use', 'studies', 'review',
                                            'editorial', 'report', 'book', 'ieee', 'published', 'science', 'column',
                                            'author', 'proposed', 'icse', 'article', 'year', 'articles', 'page', '2000',
                                            '2004', 'papers', 'computer', 'held', 'editor'])
TOKEN_PATTERN = r"(?u)\b\w\w\w+\b"
K_BEST_RATE = 0.2
IS_INDEPENDENT_VENUE = True


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


class Metrics(O):
  EPS = 0.00000001

  def __init__(self, predicted=None, actual=None):
    O.__init__(self)
    if predicted is not None and actual is not None:
      self.accuracy = sk_metrics.accuracy_score(actual, predicted)
      self.precision = sk_metrics.precision_score(actual, predicted, average='weighted')
      self.recall = sk_metrics.recall_score(actual, predicted, average='weighted')
      self.f_score = sk_metrics.f1_score(actual, predicted, average='weighted')

  @staticmethod
  def avg_score(metrics_arr):
    accuracies, precisions, recalls, f_scores = [], [], [], []
    for metrics in metrics_arr:
      accuracies.append(metrics.accuracy)
      precisions.append(metrics.precision)
      recalls.append(metrics.recall)
      f_scores.append(metrics.f_score)
    score = Metrics()
    score.accuracy = O(median=Metrics.median(accuracies), iqr=Metrics.iqr(accuracies))
    score.precision = O(median=Metrics.median(precisions), iqr=Metrics.iqr(precisions))
    score.recall = O(median=Metrics.median(recalls), iqr=Metrics.iqr(recalls))
    score.f_score = O(median=Metrics.median(f_scores), iqr=Metrics.iqr(f_scores))
    return score

  @staticmethod
  def iqr(x):
    return round(np.subtract(*np.percentile(x, [75, 25])), 2)

  @staticmethod
  def median(x):
    return round(np.median(x), 2)

  def __repr__(self):
    return "Accuracy:%0.2f+-%0.2f; F Score:%0.2f+-%0.2f; Precision:%0.2f+-%0.2f; Recall:%0.2f+-%0.2f"\
           % (self.accuracy.median, self.accuracy.iqr,
              self.f_score.median, self.f_score.iqr,
              self.precision.median, self.precision.iqr,
              self.recall.median, self.recall.iqr)


def get_papers_and_groups(graph, is_independent=False):
  """
  :param graph: Graph object
  :param is_independent: boolean - If true returns conference ID else returns group ID
  :return:
  """
  paper_nodes = graph.paper_nodes
  papers = []
  groups = []
  for paper_id, paper in paper_nodes.items():
    if paper.abstract is not None and paper.abstract != 'None':
      paper.raw = paper.abstract
    else:
      paper.raw = paper.title
    papers.append(paper)
    if is_independent:
      groups.append(paper.venue)
    else:
      paper.group = GROUP_CONFERENCE_MAP[int(paper.venue)]
      groups.append(paper.group)
  return np.array(papers), np.array(groups)


def split(dependent, independent, n_folds):
  skf = StratifiedKFold(n_splits=n_folds, random_state=RANDOM_STATE)
  for train_indices, test_indices in skf.split(dependent, independent):
    train_x = dependent[train_indices]
    train_y = independent[train_indices]
    test_x = dependent[test_indices]
    test_y = independent[test_indices]
    yield train_x, train_y, test_x, test_y


def pre_process_ldade(*args):
  x_train = args[0]
  y_train = args[1]
  x_test = args[2]
  y_test = args[3]
  x_train = [x.topics_score for x in x_train]
  x_test = [x.topics_score for x in x_test]
  return x_train, y_train, x_test, y_test


def pre_process_tfidf(*args):
  x_train = args[0]
  y_train = args[1]
  x_test = args[2]
  y_test = args[3]
  x_train = [x.vectorized for x in x_train]
  x_test = [x.vectorized for x in x_test]
  return x_train, y_train, x_test, y_test


def pre_process_pruned_tfidf(*args):
  x_train = args[0]
  y_train = args[1]
  x_test = args[2]
  y_test = args[3]
  x_train = [x.vectorized for x in x_train]
  x_test = [x.vectorized for x in x_test]
  k = int(round(K_BEST_RATE * len(x_train[0]), ))
  k_best = SelectKBest(chi2, k=k)
  k_best.fit(x_train, y_train)
  train_transformed = k_best.transform(x_train)
  test_transformed = k_best.transform(x_test)
  return train_transformed, y_train, test_transformed, y_test


def pre_process_64(*args):
  x_train = args[0]
  y_train = args[1]
  x_test = args[2]
  y_test = args[3]
  x_train = [x.tsne_64 for x in x_train]
  x_test = [x.tsne_64 for x in x_test]
  return x_train, y_train, x_test, y_test


def pre_process_64_ref(*args):
  x_train = args[0]
  y_train = args[1]
  x_test = args[2]
  y_test = args[3]
  x_train = [x.tsne_64_ref for x in x_train]
  x_test = [x.tsne_64_ref for x in x_test]
  return x_train, y_train, x_test, y_test


def pre_process_128(*args):
  x_train = args[0]
  y_train = args[1]
  x_test = args[2]
  y_test = args[3]
  x_train = [x.tsne_128 for x in x_train]
  x_test = [x.tsne_128 for x in x_test]
  return x_train, y_train, x_test, y_test


def pre_process_128_ref(*args):
  x_train = args[0]
  y_train = args[1]
  x_test = args[2]
  y_test = args[3]
  x_train = [x.tsne_128_ref for x in x_train]
  x_test = [x.tsne_128_ref for x in x_test]
  return x_train, y_train, x_test, y_test


def logistic_regression(pre_process, x_train, y_train, x_test, y_test):
  x_train, y_train, x_test, y_test = pre_process(x_train, y_train, x_test, y_test)
  clf = LogisticRegression().fit(x_train, y_train)
  predicted = clf.predict(x_test)
  return predicted, Metrics(predicted, y_test)


def decision_tree(pre_process, x_train, y_train, x_test, y_test):
  x_train, y_train, x_test, y_test = pre_process(x_train, y_train, x_test, y_test)
  clf = DecisionTreeClassifier(random_state=RANDOM_STATE).fit(x_train, y_train)
  predicted = clf.predict(x_test)
  return predicted, Metrics(predicted, y_test)


def svm(pre_process, x_train, y_train, x_test, y_test):
  x_train, y_train, x_test, y_test = pre_process(x_train, y_train, x_test, y_test)
  clf = LinearSVC(random_state=RANDOM_STATE).fit(x_train, y_train)
  predicted = clf.predict(x_test)
  return predicted, Metrics(predicted, y_test)


def random_forest(pre_process, x_train, y_train, x_test, y_test):
  x_train, y_train, x_test, y_test = pre_process(x_train, y_train, x_test, y_test)
  clf = RandomForestClassifier(random_state=RANDOM_STATE).fit(x_train, y_train)
  predicted = clf.predict(x_test)
  return predicted, Metrics(predicted, y_test)


def naive_bayes(pre_process, x_train, y_train, x_test, y_test):
  x_train, y_train, x_test, y_test = pre_process(x_train, y_train, x_test, y_test)
  clf = MultinomialNB().fit(x_train, y_train)
  predicted = clf.predict(x_test)
  return predicted, Metrics(predicted, y_test)


def fetch_pkl(file_name):
  with open(file_name) as f:
    return cPkl.load(f)


def process_embeddings(index, train_x, test_x):
  vocab = fetch_pkl("cache/vocabulary/%d.pkl" % index)
  vocabulary, reverse_vocabulary = vocab['forward'], vocab['reverse']
  analyze = analyzer()

  def operate(doc, t_64, t_64_ref, t_128, t_128_ref):
    doc.tokens = analyze(doc.raw)
    doc.tsne_64 = np.zeros(64, np.float64)
    doc.tsne_64_ref = np.zeros(64, np.float64)
    doc.tsne_128 = np.zeros(128, np.float64)
    doc.tsne_128_ref = np.zeros(128, np.float64)
    count = 0
    for token in doc.tokens:
      if token not in vocabulary:
        continue
      word_index = vocabulary[token]
      count += 1
      doc.tsne_64 += t_64[word_index]
      doc.tsne_64_ref += t_64_ref[word_index]
      doc.tsne_128 += t_128[word_index]
      doc.tsne_128_ref += t_128_ref[word_index]
    if count > 0:
      doc.tsne_64 /= count
      doc.tsne_64_ref /= count
      doc.tsne_128 /= count
      doc.tsne_128_ref /= count

  tsne_64 = fetch_pkl("cache/tsne/64_components_%d.pkl" % index)
  tsne_64_ref = fetch_pkl("cache/tsne/64_components_%d_ref.pkl" % index)
  tsne_128 = fetch_pkl("cache/tsne/128_components_%d.pkl" % index)
  tsne_128_ref = fetch_pkl("cache/tsne/128_components_%d_ref.pkl" % index)
  for d in train_x:
    operate(d, tsne_64, tsne_64_ref, tsne_128, tsne_128_ref)
  for d in test_x:
    operate(d, tsne_64, tsne_64_ref, tsne_128, tsne_128_ref)


def predict_venues(estimators, is_independent=IS_INDEPENDENT_VENUE,
                   n_folds=5, n_topics=N_TOPICS, alpha=ALPHA, beta=BETA,
                   n_iter=100, min_tfidf_score=0.1, tfidf_top=100, random_state=RANDOM_STATE):
  def make_key(pred, pre_proc):
    return "%s - %s" % (pred.__name__, pre_proc.__name__)

  graph = cite_graph(GRAPH_CSV)
  miner = Miner(graph)
  papers, groups = get_papers_and_groups(graph, is_independent=is_independent)
  metrics_map = {make_key(predictor, preprocessor): [] for predictor, preprocessor in estimators}
  for index, (train_x, train_y, test_x, test_y) in enumerate(split(papers, groups, n_folds=n_folds)):
    print("#### Iteration %d" % (index + 1))
    # TSNE
    process_embeddings(index, train_x, test_x)
    # Count Vectorizer
    vectorizer = CountVectorizer(stop_words=STOP_WORDS, token_pattern=TOKEN_PATTERN)
    train_docs = [paper.raw for paper in train_x]
    test_docs = [paper.raw for paper in test_x]
    train_vectorized = vectorizer.fit_transform(train_docs).toarray()
    test_vectorized = vectorizer.transform(test_docs).toarray()
    # TFIDF
    tfidf_transformer = TfidfTransformer()
    tfidf_matrix = tfidf_transformer.fit_transform(train_vectorized).toarray()
    tfidf_matrix[tfidf_matrix < min_tfidf_score] = 0
    tfidf_means = np.mean(tfidf_matrix, axis=0)
    tfidf_top_indices = np.argsort(tfidf_means)[::-1][:tfidf_top]
    # LDA-DE
    alpha = alpha if alpha else 50 / N_TOPICS
    beta = beta if beta else 0.01
    lda_model = lda.LDA(n_topics=n_topics, alpha=alpha, eta=beta, n_iter=n_iter, random_state=random_state)
    train_transformed = lda_model.fit_transform(train_vectorized)
    # Putting it together
    for i, (vectorized, topics) in enumerate(zip(train_vectorized, train_transformed)):
      train_x[i].vectorized = vectorized[tfidf_top_indices]
      train_x[i].topics_count = topics
      sum_t = sum(topics)
      sum_t = sum_t if sum_t else 0.00001
      train_x[i].topics_score = [np.float(t / sum_t) for t in topics]
    test_transformed = lda_model.transform(test_vectorized)
    for i, (vectorized, topics) in enumerate(zip(test_vectorized, test_transformed)):
      test_x[i].vectorized = vectorized[tfidf_top_indices]
      test_x[i].topics_count = topics
      sum_t = sum(topics)
      sum_t = sum_t if sum_t else 0.00001
      test_x[i].topics_score = [t / sum_t for t in topics]
    for predictor, preprocessor in estimators:
      key = make_key(predictor, preprocessor)
      print(key)
      predicted, metrics = predictor(preprocessor, train_x, train_y, test_x, test_y)
      metrics_map[key].append(metrics)
  for predictor, preprocessor in estimators:
    key = make_key(predictor, preprocessor)
    print("### " + key)
    measures = Metrics.avg_score(metrics_map[key])
    print(measures)


if __name__ == "__main__":
  pre_processors = [pre_process_tfidf, pre_process_pruned_tfidf, pre_process_ldade,
                    pre_process_64, pre_process_64_ref, pre_process_128, pre_process_128_ref]
  # predictors = [random_forest]
  predictors = [decision_tree, logistic_regression, svm, random_forest, naive_bayes]
  predict_estimators = [(pred, pp) for pred in predictors for pp in pre_processors]
  print("## Independent\n")
  predict_venues(predict_estimators, is_independent=True)
  print("\n\n## Clustered\n")
  predict_venues(predict_estimators, is_independent=False)
