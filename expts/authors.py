from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True
from utils.lib import O, Memoized, file_exists
import numpy as np
from collections import OrderedDict
from network.graph import Graph
import pandas as pd
from sklearn import preprocessing
import cPickle as cPkl
import db.mysqldb as mysql
import matplotlib.pyplot as plt
import seaborn as sns


__author__ = "bigfatnoob"


GRAPH_CSV = "data/citemap_v10.csv"

THE = O()
THE.permitted = 'all'
THE.version = 'v4'

@Memoized
def cite_graph(file_name):
  return Graph.from_file(file_name)


def is_not_none(s):
  return s and s != 'None'


def all_authors(graph, min_year=None):
  """
  :param graph:
  :param min_year:
  :return: [(str(author_id), paper_count, cite_count)]
  """
  authors = graph.get_papers_by_authors(THE.permitted)
  author_cites = []
  for author_id, papers in authors.items():
    cite_count = 0
    paper_count = 0
    for paper_id, year, __ in papers:
      if min_year is not None and int(year) < min_year: continue
      cited = graph.paper_nodes[paper_id].cited_count
      paper_count += 1
      if is_not_none(cited):
        cite_count += int(cited)
    author_cites.append((author_id, paper_count, cite_count))
  # tops = sorted(author_cites, key=lambda x: x[1], reverse=True)[:int(top_percent * len(author_cites))]
  return author_cites


def most_cited_authors(graph, top_percent=None, min_year=None):
  """
  [(str(author_id), paper_count, cite_count)]
  :param graph:
  :param top_percent:
  :param min_year:
  :return:
  """
  authors = all_authors(graph, min_year)
  if top_percent is None:
    top_percent = 1
  return [a for a in sorted(authors, key=lambda x: x[2], reverse=True)[:int(top_percent * len(authors))]]


def link_matrix(graph, valid_authors, min_year=None):
  links_file = "figs/%s/%s/authors/links.pkl" % (THE.version, THE.permitted)
  collaborators_file = "figs/%s/%s/authors/collaborators.pkl" % (THE.version, THE.permitted)
  if file_exists(links_file) and file_exists(collaborators_file):
    with open(collaborators_file) as f:
      collaborators = cPkl.load(f)
    with open(links_file) as f:
      data = cPkl.load(f)
      if data.columns.values.tolist() == valid_authors:
        return data, collaborators
  papers = graph.get_paper_nodes(permitted=THE.permitted)
  valid_authors_set = set(valid_authors)
  matrix = pd.DataFrame(index=valid_authors, columns=valid_authors).fillna(0)
  collaborators = {a: set() for a in valid_authors}
  for paper_id, paper in papers.items():
    if min_year is not None and int(paper.year) < min_year:
      continue
    author_ids = paper.author_ids.strip().split(",")
    for i in range(len(author_ids) - 1):
      if author_ids[i] not in valid_authors_set:
        continue
      for j in range(i + 1, len(author_ids)):
        if author_ids[j] not in valid_authors_set:
          continue
        # Setting as 1 rather than forced increment to
        # avoid over collaboration
        matrix.ix[author_ids[i], author_ids[j]] += 1
        matrix.ix[author_ids[j], author_ids[i]] += 1
        i_collaborators = collaborators[author_ids[i]]
        i_collaborators.add(author_ids[j])
        collaborators[author_ids[i]] = i_collaborators
        j_collaborators = collaborators[author_ids[j]]
        j_collaborators.add(author_ids[i])
        collaborators[author_ids[j]] = j_collaborators
  with open(links_file, "wb") as f:
    cPkl.dump(matrix, f)
  with open(collaborators_file, "wb") as f:
    cPkl.dump(collaborators, f)
  return matrix, collaborators


def max_normalize(d):
  max_val = max(d.values())
  return {d_i: d[d_i] / max_val for d_i in d.keys()}


def page_rank(authors, links, author_collaborators, damp=0.5, iterations=100000,
              file_name="page_rank", prefix=None):
  init = 1 / len(authors)
  pr = {a: init for a in authors}
  for i in xrange(iterations):
    if i % 100 == 0:
      print("Iteration : %d" % i)
    recomputed_pr = {}
    for a in pr.keys():
      collaborators = author_collaborators[a]
      link_score = 0
      for collaborator in collaborators:
        # link_score += pr[collaborator] / sum(links.ix[:, collaborator] > 0)
        link_score += pr[collaborator] / len(author_collaborators[collaborator])
      link_score *= damp
      self_score = (1 - damp) * init
      recomputed_pr[a] = self_score + link_score
    pr = max_normalize(recomputed_pr)
  if prefix is None:
    f_name = "figs/%s/%s/authors/%s.pkl" % (THE.version, THE.permitted, file_name)
  else:
    f_name = "figs/%s/%s/authors/%s/%s.pkl" % (THE.version, THE.permitted, prefix, file_name)
  with open(f_name, "wb") as f:
    cPkl.dump(pr, f, cPkl.HIGHEST_PROTOCOL)
  return pr


def df_column_normalize(df):
  x = df.values  # returns a numpy array
  min_max_scaler = preprocessing.MinMaxScaler()
  x_scaled = min_max_scaler.fit_transform(x)
  return pd.DataFrame(x_scaled, index=df.index.values, columns=df.columns.values)


def weighted_page_rank(authors, links, weights, author_collaborators, damp=0.5, iterations=100000,
                       file_name="cite_page_rank", prefix=None):
  init = 1 / len(authors)
  pr = {a: init for a in authors}
  links = df_column_normalize(links).sum(axis=1)
  links = {a: links[a] for a in pr.keys()}
  total_weights = sum(weights.values())
  weights = {a_id: w / total_weights for a_id, w in weights.items()}
  for i in xrange(iterations):
    if i % 100 == 0:
      print("Iteration : %d" % i)
    recomputed_pr = {}
    for a in pr.keys():
      collaborators = author_collaborators[a]
      link_score = 0
      for collaborator in collaborators:
        # link_score += pr[collaborator] / sum(links.ix[:, collaborator] > 0)
        link_score += pr[collaborator] / links[collaborator]
      link_score *= damp
      self_score = (1 - damp) * weights[a]
      recomputed_pr[a] = self_score + link_score
    pr = max_normalize(recomputed_pr)
  if prefix is None:
    f_name = "figs/%s/%s/authors/%s.pkl" % (THE.version, THE.permitted, file_name)
  else:
    f_name = "figs/%s/%s/authors/%s/%s.pkl" % (THE.version, THE.permitted, prefix, file_name)
  with open(f_name, "wb") as f:
    cPkl.dump(pr, f, cPkl.HIGHEST_PROTOCOL)
  return pr


def run_page_rank(min_year, damp, top=0.01, iterations=100000, use_prefix=False):
  graph = cite_graph(GRAPH_CSV)
  authors = [a[0] for a in most_cited_authors(graph, top, min_year)]
  links, author_collaborators = link_matrix(graph, authors, min_year)
  if use_prefix:
    page_rank(authors, links, author_collaborators, damp, iterations=iterations,
              file_name="page_rank_%0.2f" % damp, prefix="naive")
  else:
    page_rank(authors, links, author_collaborators, damp, iterations=iterations,
              file_name="page_rank_%0.2f" % damp)


def run_weighted_page_rank(min_year, damp, top=0.01, iterations=100000, weight_param="cite", use_prefix=False):
  graph = cite_graph(GRAPH_CSV)
  authors, weights = [], OrderedDict()
  for a in most_cited_authors(graph, top, min_year):
    authors.append(a[0])
    if weight_param == "cite":
      weights[a[0]] = a[2]
    elif weight_param == "publ":
      weights[a[0]] = a[1]
    else:
      print ("Invalid weighing scheme : %s " % weight_param)
      exit()
  links, author_collaborators = link_matrix(graph, authors, min_year)
  if use_prefix:
    weighted_page_rank(authors, links, weights, author_collaborators, damp,
                       iterations=iterations, file_name="page_rank_%0.2f" % damp, prefix=weight_param)
  else:
    weighted_page_rank(authors, links, weights, author_collaborators, damp,
                       iterations=iterations, file_name="%s_page_rank_%0.2f" % (weight_param, damp))


def plot_damp_top_authors(folder, damps, top, min_year, plot_author_count=20):
  graph = cite_graph(GRAPH_CSV)
  top_authors = most_cited_authors(graph, top, min_year)[:plot_author_count]
  author_nodes = mysql.get_authors()
  x_labels = [author_nodes[a[0]].name for a in top_authors]
  x_axis = range(1, plot_author_count + 1)
  top_author_ids = np.array([a[0] for a in top_authors])
  folder_path = "figs/%s/%s/authors/%s" % (THE.version, THE.permitted, folder)
  palette = np.array(sns.color_palette("hls", plot_author_count))
  legends = []
  # for i, f_name in enumerate(os.listdir(folder_path)):
  y_axes = []
  means = np.array([0.0] * plot_author_count)
  for i, _ in enumerate(damps):
    # file_name = "%s/%s" % (folder_path, name)
    file_name = "%s/page_rank_%0.2f.pkl" % (folder_path, damps[i])
    with open(file_name) as f:
      pr_scores = cPkl.load(f)
      y_axis = np.array([pr_scores[a] for a in top_author_ids])
      y_axes.append(y_axis)
      means += y_axis
  indices = np.argsort(means)[::-1]
  top_author_ids = top_author_ids[indices]
  # sns.set_style("whitegrid", {'axes.grid': False})
  sns.set_style("white")
  for i, y_axis in enumerate(y_axes):
    plt.plot(x_axis, y_axis[indices], c=palette[i])
    legends.append("%0.2f" % damps[i])
  plt.legend(legends, loc='upper right', ncol=2)
  fig_name = "figs/%s/%s/authors/damp_%s.png" % (THE.version, THE.permitted, folder)
  plt.ylabel("Page Rank Score")
  plt.xlabel("Author ID")
  plt.xticks(x_axis, top_author_ids, rotation='vertical')
  plt.title("Page Rank Score for top %d cited author with varying damping factors" % plot_author_count)
  plt.savefig(fig_name)
  plt.clf()


def _damp_plotter(damps=None):
  if damps is None:
    damps = np.arange(0.05, 1.00, 0.05)
  plot_damp_top_authors("naive", damps, top=0.01, min_year=1992)
  plot_damp_top_authors("cite", damps, top=0.01, min_year=1992)
  plot_damp_top_authors("publ", damps, top=0.01, min_year=1992)


def _damp_scores():
  iters = 1000
  damps = np.arange(0.05, 1.00, 0.05)
  for damp in damps:
    run_page_rank(1992, damp, use_prefix=True, iterations=iters)
    run_weighted_page_rank(1992, damp, weight_param="cite", use_prefix=True, iterations=iters)
    run_weighted_page_rank(1992, damp, weight_param="publ", use_prefix=True, iterations=iters)
  _damp_plotter(damps)


if __name__ == "__main__":
  # _damp_scores()
  _damp_plotter()
