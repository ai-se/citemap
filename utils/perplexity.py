from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True
from scipy.special import gammaln
from sklearn.utils.extmath import logsumexp
from sklearn.decomposition._online_lda import _dirichlet_expectation_2d
import numpy as np
import lda

__author__ = "bigfatnoob"


def log_likelihood(prior, dist, dirichlet_dist, size):
  score = np.sum((prior - dist) * dirichlet_dist)
  score += np.sum(gammaln(dist) - gammaln(prior))
  score += np.sum(gammaln(prior * size) - gammaln(np.sum(dist, 1)))
  return score


def approx_bound(lda_model, documents, doc_topic_dist):
  n_samples, n_topics = doc_topic_dist.shape
  n_features = lda_model.components_.shape[1]
  score = 0
  dirichlet_doc_topic = _dirichlet_expectation_2d(doc_topic_dist)
  dirichlet_component_ = _dirichlet_expectation_2d(lda_model.components_)
  doc_topic_prior = lda_model.alpha
  topic_word_prior = lda_model.eta
  for idx_d in xrange(0, n_samples):
    ids = np.nonzero(documents[idx_d, :])[0]
    cnts = documents[idx_d, ids]
    norm_phi = logsumexp(dirichlet_doc_topic[idx_d, :, np.newaxis] + dirichlet_component_[:, ids])
    score += np.dot(cnts, norm_phi)
  # score += log_likelihood(doc_topic_prior, doc_topic_dist, dirichlet_doc_topic, lda_model.n_topics)
  # score += log_likelihood(topic_word_prior, lda_model.components_, dirichlet_component_, n_features)
  return score


def transform(lda_model, X, max_iter=20, tol=1e-16):
  doc_topic = np.empty((X.shape[0], lda_model.n_topics))
  WS, DS = lda.utils.matrix_to_lists(X)
  for d in np.unique(DS):
    doc_topic[d] = transform_single(lda_model, WS[DS == d], max_iter, tol)
  return doc_topic


def transform_single(lda_model, doc, max_iter, tol):
  PZS = np.zeros((len(doc), lda_model.n_topics))
  for iteration in range(max_iter + 1):  # +1 is for initialization
    PZS_new = lda_model.components_[:, doc].T
    PZS_new *= (PZS.sum(axis=0) - PZS + lda_model.alpha)
    PZS_new /= PZS_new.sum(axis=1)[:, np.newaxis]  # vector to single column matrix
    delta_naive = np.abs(PZS_new - PZS).sum()
    PZS = PZS_new
    if delta_naive < tol:
      break
  # print(PZS)
  theta_doc = PZS.sum(axis=0) / PZS.sum()
  assert len(theta_doc) == lda_model.n_topics
  assert theta_doc.shape == (lda_model.n_topics,)
  return theta_doc


def log_perplexity(lda_model, documents):
  # x = graph.vectorizer.transform([document.get_raw() for document in documents]).toarray()
  doc_topic_dist = transform(lda_model, documents)
  # print(doc_topic_dist)
  bound = approx_bound(lda_model, documents, doc_topic_dist)
  word_count = sum(map(sum, documents))
  return bound / word_count
