from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True
from network.mine import Miner, cite_graph
from classify.hack import Submission
from sklearn.feature_extraction import text
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import lda
import matplotlib.pyplot as plt
import pandas as pd
from utils.lib import O

GRAPH_CSV = "data/citemap_v4.csv"
CLASSIFY_CSV = "classify/data.csv"
ACCEPTED = 0
REJECTED = 1
RANDOM_STATE = 0

# For 11 TOPICS
N_TOPICS = 11
ALPHA = 0.22359
BETA = 0.53915
ITERATIONS = 100
TOPICS = ["Design", "Testing", "Modelling", "Mobile", "Energy", "Defects",
          "SourceCode", "WebApps", "Configuration", "Developer", "Mining"]
TOPIC_THRESHOLD = 3
DELIMITER = '|'
STOP_WORDS = text.ENGLISH_STOP_WORDS.union(['software', 'engineering'])
TOKEN_PATTERN = r"(?u)\b\w\w\w+\b"

PRE_REJECT = 'pre-reject'

class Metrics(O):
  EPS = 0.00000001

  def __init__(self, predicted, actual, positive, negative, raw_decisions):
    O.__init__(self)
    self.tp, self.fp, self.fn, self.tn = 0, 0, 0, 0
    self.pre_reject, self.pre_reject_missed = 0, 0
    for i, (p, a) in enumerate(zip(predicted, actual)):
      if p == positive and a == positive:
        self.tp += 1
      elif p == positive and a == negative:
        self.fp += 1
      elif p == negative and a == positive:
        self.fn += 1
      else:
        self.tn += 1
      if raw_decisions[i] == PRE_REJECT and p == positive:
        self.pre_reject_missed += 1
      elif raw_decisions[i] == PRE_REJECT:
        self.pre_reject += 1
    self.accuracy = (self.tp + self.tn) / len(predicted)
    self.precision = self.tp / (self.tp + self.fp + Metrics.EPS)
    self.recall = self.tp / (self.tp + self.fn + Metrics.EPS)
    self.specificity = self.tn / (self.tn + self.fp + Metrics.EPS)
    self.f_score = 2 * self.precision * self.recall / (self.precision + self.recall + Metrics.EPS)

  @staticmethod
  def avg_score(metrics_arr):
    accuracies, precisions, recalls, f_scores, specificities = [], [], [], [], []
    pre_reject_misseds = []
    for metrics in metrics_arr:
      accuracies.append(metrics.accuracy)
      precisions.append(metrics.precision)
      recalls.append(metrics.recall)
      f_scores.append(metrics.f_score)
      specificities.append(metrics.specificity)
      pre_reject_misseds.append(metrics.pre_reject_missed / (metrics.pre_reject+metrics.EPS))
    score = O()
    score.accuracy = O(median=Metrics.median(accuracies), iqr=Metrics.iqr(accuracies))
    score.precision = O(median=Metrics.median(precisions), iqr=Metrics.iqr(precisions))
    score.recall = O(median=Metrics.median(recalls), iqr=Metrics.iqr(recalls))
    score.f_score = O(median=Metrics.median(f_scores), iqr=Metrics.iqr(f_scores))
    score.specificity = O(median=Metrics.median(specificities), iqr=Metrics.iqr(specificities))
    score.pre_reject_missed = O(median=Metrics.median(pre_reject_misseds), iqr=Metrics.iqr(pre_reject_misseds))
    return score

  @staticmethod
  def iqr(x):
    return round(np.subtract(*np.percentile(x, [75, 25])),2)

  @staticmethod
  def median(x):
    return round(np.median(x), 2)



def get_graph_lda_data(iterations=ITERATIONS):
  graph = cite_graph(GRAPH_CSV)
  miner = Miner(graph)
  lda_model, vocab = miner.lda(N_TOPICS, n_iter=iterations, alpha=ALPHA, beta=BETA)
  return miner, graph, lda_model, vocab


def read_papers():
  submissions = []
  with open(CLASSIFY_CSV, 'rb') as f:
    f.readline().strip().lower().split(DELIMITER)
    for line in f.readlines():
      submission = Submission()
      line = line.decode('utf-8', 'ignore').encode("utf-8")
      columns = line.strip().split(DELIMITER)
      submission.conference = columns[0]
      submission.year = columns[1]
      submission.title = columns[2]
      submission.authors = columns[3].split(",")
      submission.keywords = columns[4].split(",")
      submission.abstract = columns[5]
      submission.category = columns[6] if columns[6] != 'None' else None
      submission.decision = columns[7]
      submission.raw_decision = columns[8]
      submissions.append(submission)
  return submissions


def vectorize(papers, iterations=ITERATIONS):
  miner, graph, lda_model, vocab = get_graph_lda_data(iterations=iterations)
  # vectorizer = text.CountVectorizer(stop_words=STOP_WORDS, token_pattern=TOKEN_PATTERN)
  docs = [paper.abstract
          if paper.abstract is not None and paper.abstract != 'None' else paper.title for paper in papers]
  doc_2_vec = miner.vectorizer.transform(docs)
  doc_2_vec_array = doc_2_vec.toarray()
  transformed = lda_model.transform(doc_2_vec_array)
  report(lda_model, vocab)
  for paper, t, d_2_v in zip(papers, transformed, doc_2_vec_array):
    paper.transformed = t
    paper.doc_2_vec = d_2_v
  return O(miner=miner, graph=graph, lda_model=lda_model, vocab=vocab, doc_2_vec=doc_2_vec)


def format_conf_acceptance(papers):
  formatted = {}
  print(len(papers))
  for paper in papers:
    key = "%s-%s" % (paper.conference, paper.year)
    if key not in formatted:
      formatted[key] = {
          "accept": [],
          "reject": []
      }
    formatted[key][paper.decision].append(paper)
  return formatted


def make_heatmap(arr, row_labels, column_labels, title, figname):
  plt.figure(figsize=(4, 3))
  df = pd.DataFrame(arr, columns=column_labels, index=row_labels)
  cax = plt.matshow(df, interpolation='nearest', cmap='hot_r')
  plt.colorbar(cax)
  plt.xticks(np.arange(len(list(df.columns))), list(df.columns), rotation="vertical")
  plt.yticks(np.arange(len(list(df.index))), list(df.index))
  plt.title("Topics to Conference Distribution for %s" % title, y=1.2)
  plt.savefig(figname, bbox_inches='tight')
  plt.clf()


def report(lda_model, vocab, n_top_words=10):
  for index, topic_dist in enumerate(lda_model.topic_word_):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
    print('Topic {}: {}'.format(index, ', '.join(topic_words)))


def acceptance_pattern():
  papers = read_papers()
  vectorize(papers)
  formatted = format_conf_acceptance(papers)
  print(formatted.keys())
  for conference_id, paper_dict in formatted.items():
    acceptance_heatmaps = {}
    for key, ps in paper_dict.items():
      topics = np.array([0] * N_TOPICS)
      for paper in ps:
        topics = np.add(topics, paper.transformed)
      topics = topics / float(len(ps))
      acceptance_heatmaps[key] = topics
    column_labels = TOPICS
    row_labels = sorted(acceptance_heatmaps.keys())
    heatmap_arr = []
    for key in sorted(acceptance_heatmaps.keys()):
      heatmap_arr.append(acceptance_heatmaps[key])
    make_heatmap(heatmap_arr, row_labels, column_labels, conference_id, "classify/figs/%s.png" % conference_id)


def stratified_kfold(accepted, rejected, folds=10):
  labels = [ACCEPTED] * len(accepted) + [REJECTED] * len(rejected)
  all_papers = np.array(accepted + rejected)
  skf = StratifiedKFold(n_splits=folds)
  for train_indices, test_indices in skf.split(all_papers, labels):
    train = all_papers[train_indices]
    test = all_papers[test_indices]
    x_train, y_train = [], []
    for paper in train:
      # x_train.append(paper.transformed)
      x_train.append(paper)
      y_train.append(ACCEPTED if paper.decision == 'accept' else REJECTED)
    x_test, y_test = [], []
    for paper in test:
      # x_test.append(paper.transformed)
      x_test.append(paper)
      y_test.append(ACCEPTED if paper.decision == 'accept' else REJECTED)
    yield x_train, y_train, x_test, y_test


def decision_tree(*args):
  x_train = args[0]
  y_train = args[1]
  x_test = args[2]
  y_test = args[3]
  y_test_raw = [x.raw_decision for x in x_test]
  x_train = [x.transformed for x in x_train]
  x_test = [x.transformed for x in x_test]
  clf = DecisionTreeClassifier(random_state=RANDOM_STATE).fit(x_train, y_train)
  predicted = clf.predict(x_test)
  return predicted, Metrics(predicted, y_test, ACCEPTED, REJECTED, y_test_raw)


def linear_regression(*args):
  x_train = args[0]
  y_train = args[1]
  x_test = args[2]
  y_test = args[3]
  y_test_raw = [x.raw_decision for x in x_test]
  x_train = [x.transformed for x in x_train]
  x_test = [x.transformed for x in x_test]
  clf = LogisticRegression().fit(x_train, y_train)
  predicted = clf.predict(x_test)
  return predicted, Metrics(predicted, y_test, ACCEPTED, REJECTED, y_test_raw)


def tfidf_preprocessor(*args):
  x_train = args[0]
  x_test = args[1]
  x_train = [x.doc_2_vec for x in x_train]
  x_test = [x.doc_2_vec for x in x_test]
  tfidf_model = TfidfTransformer().fit(x_train)
  x_train_tfidf = tfidf_model.transform(x_train)
  x_test_tfidf = tfidf_model.transform(x_test)
  return x_train_tfidf, x_test_tfidf


def tfidf_decision_tree(*args):
  x_train = args[0]
  y_train = args[1]
  x_test = args[2]
  y_test = args[3]
  y_test_raw = [x.raw_decision for x in x_test]
  x_train, x_test = tfidf_preprocessor(x_train, x_test)
  clf = DecisionTreeClassifier(random_state=RANDOM_STATE).fit(x_train, y_train)
  predicted = clf.predict(x_test)
  return predicted, Metrics(predicted, y_test, ACCEPTED, REJECTED, y_test_raw)


def tfidf_linear_regression(*args):
  x_train = args[0]
  y_train = args[1]
  x_test = args[2]
  y_test = args[3]
  y_test_raw = [x.raw_decision for x in x_test]
  x_train, x_test = tfidf_preprocessor(x_train, x_test)
  clf = LogisticRegression().fit(x_train, y_train)
  predicted = clf.predict(x_test)
  return predicted, Metrics(predicted, y_test, ACCEPTED, REJECTED, y_test_raw)


def classify(classifiers):
  papers = read_papers()
  vectorize(papers, iterations=100)
  formatted = format_conf_acceptance(papers)
  accepteds, rejecteds = [], []
  for conference_id, paper_dict in formatted.items():
    print("## %s" % conference_id)
    accepted = paper_dict['accept']
    rejected = paper_dict['reject']
    accepteds += accepted
    rejecteds += rejected
    metrics_map = {classifier.__name__: [] for classifier in classifiers}
    for x_train, y_train, x_test, y_test in stratified_kfold(accepted, rejected):
      for classifier in classifiers:
        predicted, metrics = classifier(x_train, y_train, x_test, y_test)
        metrics_map[classifier.__name__].append(metrics)
    for classifier in classifiers:
      print(classifier.__name__)
      measures = Metrics.avg_score(metrics_map[classifier.__name__])
      print(measures)
  metrics_map = {classifier.__name__: [] for classifier in classifiers}
  print('## ALL')
  for x_train, y_train, x_test, y_test in stratified_kfold(accepteds, rejecteds):
    for classifier in classifiers:
      predicted, metrics = classifier(x_train, y_train, x_test, y_test)
      metrics_map[classifier.__name__].append(metrics)
  for classifier in classifiers:
    print(classifier.__name__)
    measures = Metrics.avg_score(metrics_map[classifier.__name__])
    print(measures)


if __name__ == "__main__":
  classify([decision_tree, tfidf_decision_tree])
# acceptance_pattern()
