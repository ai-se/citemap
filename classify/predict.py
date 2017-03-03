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

GRAPH_CSV = "data/citemap_v4.csv"
CLASSIFY_CSV = "classify/data.csv"
ACCEPTED = 0
REJECTED = 1
RANDOM_STATE = 1
"""
Short Name Start Group
15 ISSTA A1
16 ICST  A2
5  MSR   B1
8  ICPC  B2
2  ICSM  B3
3  WCRE  B4
10 SCAM  C1
4  CSMR  C2
12 SANER C3
14 RE    D1
13 SSBSE E1
1  ICSE  E2
9  FSE   E3
11 ASE   E4
6  GPCE  F1
7  FASE  F2
"""
GROUP_CONFERENCE_MAP = {
  15: 'A', 16: 'A',
  5: 'B', 8: 'B', 2: 'B', 3: 'B',
  10: 'C', 4: 'C', 12: 'C',
  14: 'D',
  13: 'E', 1: 'E', 9: 'E', 11: 'E',
  6: 'F', 7: 'F'
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
STOP_WORDS = text.ENGLISH_STOP_WORDS.union(['software', 'engineering'])
TOKEN_PATTERN = r"(?u)\b\w\w\w+\b"
K_BEST_RATE = 0.2
IS_INDEPENDENT_CONFERENCE = True


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


def get_papers_and_groups(miner, is_independent=False):
  """
  :param miner: Miner object
  :param is_independent: boolean - If true returns conference ID else returns group ID
  :return:
  """
  paper_nodes = miner.graph.paper_nodes
  papers = []
  groups = []
  for paper_id, paper in paper_nodes.items():
    if paper.abstract is not None and paper.abstract != 'None':
      paper.raw = paper.abstract
    else:
      paper.raw = paper.title
    paper.group = GROUP_CONFERENCE_MAP[int(paper.conference)]
    papers.append(paper)
    if is_independent:
      groups.append(paper.conference)
    else:
      groups.append(paper.group)
  return np.array(papers), np.array(groups)


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
  k = K_BEST_RATE * len(x_train[0])
  k_best = SelectKBest(chi2, k=k).fit(x_train, y_train)
  return k_best.transform(x_train), y_train, k_best.transform(x_test), y_test


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


def predict_conference(estimators, n_folds=10, n_topics=N_TOPICS, alpha=None, beta=None, n_iter=100,
                       min_tfidf_score=0.1, tfidf_top=100,
                       random_state=RANDOM_STATE):

  def make_key(pred, pre_proc):
    return "%s - %s" % (pred.__name__, pre_proc.__name__)

  graph = cite_graph(GRAPH_CSV)
  miner = Miner(graph)
  papers, groups = get_papers_and_groups(miner, is_independent=IS_INDEPENDENT_CONFERENCE)
  skf = StratifiedKFold(n_splits=n_folds)
  metrics_map = {make_key(predictor, preprocessor): [] for predictor, preprocessor in estimators}
  index = 0
  for train_indices, test_indices in skf.split(papers, groups):
    index += 1
    print("#### Iteration %d" % index)

    # Count Vectorizer
    train_papers = papers[train_indices]
    test_papers = papers[test_indices]
    y_train, y_test = groups[train_indices], groups[test_indices]
    vectorizer = CountVectorizer(stop_words=STOP_WORDS, token_pattern=TOKEN_PATTERN)
    train_docs = [paper.raw for paper in train_papers]
    test_docs = [paper.raw for paper in test_papers]
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
      train_papers[i].vectorized = vectorized[tfidf_top_indices]
      train_papers[i].topics_count = topics
      sum_t = sum(topics)
      sum_t = sum_t if sum_t else 0.00001
      train_papers[i].topics_score = [np.float(t / sum_t) for t in topics]
    test_transformed = lda_model.transform(test_vectorized)
    for i, (vectorized, topics) in enumerate(zip(test_vectorized, test_transformed)):
      test_papers[i].vectorized = vectorized[tfidf_top_indices]
      test_papers[i].topics_count = topics
      sum_t = sum(topics)
      sum_t = sum_t if sum_t else 0.00001
      test_papers[i].topics_score = [t / sum_t for t in topics]
    x_train, x_test = train_papers, test_papers
    for predictor, preprocessor in estimators:
      key = make_key(predictor, preprocessor)
      print(key)
      predicted, metrics = predictor(preprocessor, x_train, y_train, x_test, y_test)
      metrics_map[key].append(metrics)
  for predictor, preprocessor in estimators:
    key = make_key(predictor, preprocessor)
    print ("### " + key)
    measures = Metrics.avg_score(metrics_map[key])
    print (measures)


predict_conference([(decision_tree, pre_process_tfidf), (decision_tree, pre_process_pruned_tfidf), (decision_tree, pre_process_ldade),
                    (logistic_regression, pre_process_tfidf), (logistic_regression, pre_process_pruned_tfidf), (logistic_regression, pre_process_ldade),
                    (svm, pre_process_tfidf), (svm, pre_process_pruned_tfidf), (svm, pre_process_ldade),
                    (random_forest, pre_process_tfidf), (random_forest, pre_process_pruned_tfidf), (random_forest, pre_process_ldade),
                    (naive_bayes, pre_process_tfidf), (naive_bayes, pre_process_pruned_tfidf), (decision_tree, pre_process_ldade)])


