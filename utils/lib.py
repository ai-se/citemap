from __future__ import print_function, division
import sys, os
sys.path.append(os.path.abspath("."))
import numpy as np

__author__ = "panzer"


class O:
  def __init__(self, **d): self.has().update(**d)

  def has(self): return self.__dict__

  def update(self, **d) : self.has().update(d); return self

  def __repr__(self)   :
    show = [':%s %s' % (k, self.has()[k])
            for k in sorted(self.has().keys())
            if k[0] is not "_"]
    txt = ' '.join(show)
    if len(txt) > 60:
      show = map(lambda x: '\t' + x + '\n', show)
    return '{' + ' '.join(show) + '}'

  def __getitem__(self, item):
    return self.has().get(item)

  def __setitem__(self, key, value):
    self.has()[key] = value


def say(*lst):
  print(*lst, end="")
  sys.stdout.flush()


class Paper(O):
  def __init__(self, **kwargs):
    """
    :param kwargs:
     title - str
     authors - [str]
     year - int
     venue - str
     idx - str
     abstract - str
     refs = [str]
    """
    O.__init__(self, **kwargs)


class PC(O):
  def __init__(self):
    """

    """
    O.__init__(self)
    self.author_id = None
    self.conference_id = None
    self.year = None
    self.role = None

  def set_short_role(self, role):
    if role == "General Chair":
      self.role = "GC"
    elif role == "Program Chair":
      self.role = "PC"
    elif role == "PC member main track":
      self.role = "PCM"
    else:
      raise RuntimeError("Invalid role  %s"%role)


class Conference(O):
  def __init__(self):
    """
    Initialize Conference
    """
    O.__init__(self)
    self.id = None
    self.acronym = None
    self.name = None
    self.impact = None


class Node(O):
  id = 1

  def __init__(self, **params):
    O.__init__(self, **params)
    #self.id = Node.id
    Node.id += 1


class Edge(O):
  id = 1

  def __init__(self, **params):
    O.__init__(self, **params)
    #self.id =  Edge.id
    Edge.id += 1


class Metrics(O):
  EPS = 0.00000001

  def __init__(self, predicted, actual, positive, negative):
    O.__init__(self)
    self.tp, self.fp, self.fn, self.tn = 0, 0, 0, 0
    for p, a in zip(predicted, actual):
      if p == positive and a == positive:
        self.tp += 1
      elif p == positive and a == negative:
        self.fp += 1
      elif p == negative and a == positive:
        self.fn += 1
      else:
        self.tn += 1
    self.accuracy = (self.tp + self.tn) / len(predicted)
    self.precision = self.tp / (self.tp + self.fp + Metrics.EPS)
    self.recall = self.tp / (self.tp + self.fn + Metrics.EPS)
    self.specificity = self.tn / (self.tn + self.fp + Metrics.EPS)
    self.f_score = 2 * self.precision * self.recall / (self.precision + self.recall + Metrics.EPS)

  @staticmethod
  def avg_score(metrics_arr):
    accuracies, precisions, recalls, f_scores, specificities = [], [], [], [], []
    for metrics in metrics_arr:
      accuracies.append(metrics.accuracy)
      precisions.append(metrics.precision)
      recalls.append(metrics.recall)
      f_scores.append(metrics.f_score)
      specificities.append(metrics.specificity)
    score = O()
    score.accuracy = O(median=Metrics.median(accuracies), iqr=Metrics.iqr(accuracies))
    score.precision = O(median=Metrics.median(precisions), iqr=Metrics.iqr(precisions))
    score.recall = O(median=Metrics.median(recalls), iqr=Metrics.iqr(recalls))
    score.f_score = O(median=Metrics.median(f_scores), iqr=Metrics.iqr(f_scores))
    score.specificity = O(median=Metrics.median(specificities), iqr=Metrics.iqr(specificities))
    return score

  @staticmethod
  def iqr(x):
    return round(np.subtract(*np.percentile(x, [75, 25])),2)

  @staticmethod
  def median(x):
    return round(np.median(x), 2)
