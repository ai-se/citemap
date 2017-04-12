from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True
from network.mine import Miner, cite_graph
from sklearn.feature_extraction import text
from utils.lib import O
from collections import OrderedDict
from classify.model import read_papers, make_heatmap, vectorize
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from classify.predict import Metrics

GRAPH_CSV = "data/citemap_v4.csv"
CLASSIFY_CSV = "classify/data.csv"
ACCEPTED = 0
REJECTED = 1
RANDOM_STATE = 1
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
DELIMITER = '|'
STOP_WORDS = text.ENGLISH_STOP_WORDS.union(['software', 'engineering'])
TOKEN_PATTERN = r"(?u)\b\w\w\w+\b"
K_BEST_RATE = 0.2
IS_INDEPENDENT_CONFERENCE = True
STUDIED_CONFERENCES = ['FSE', 'MSR']


def top_authors(graph):
  authors = graph.get_papers_by_authors()
  author_cites = []
  for author_id, papers in authors.items():
    cite_count = 0
    for paper_id, year, __ in papers:
      cited = graph.paper_nodes[paper_id].cited_count
      if cited:
        cite_count += cited
    author_cites.append((author_id, cite_count, graph.author_nodes[author_id].name))
  tops = sorted(author_cites, key=lambda x: x[1], reverse=True)
  author_dict = OrderedDict()
  for a_id, cites, name in tops:
    author_dict[name] = (a_id, cites, name)
  return author_dict


def format_conf_acceptance(papers):
  formatted = {}
  for paper in papers:
    # if paper.conference not in STUDIED_CONFERENCES: continue
    key = "%s-%s" % (paper.conference, paper.year)
    if key not in formatted:
      formatted[key] = []
    formatted[key].append(paper)
  return formatted


def desk_rejects():
  papers = read_papers()
  vectorize(papers)
  submissions = format_conf_acceptance(papers)
  for conf_id, papers in submissions.items():
    a_topics, a_count = np.array([0] * N_TOPICS), 0
    r_topics, r_count = np.array([0] * N_TOPICS), 0
    da_topics, da_count = np.array([0] * N_TOPICS), 0
    dr_topics, dr_count = np.array([0] * N_TOPICS), 0
    for paper in papers:
      if paper.raw_decision == 'pre-reject':
        dr_topics = np.add(dr_topics, paper.transformed)
        dr_count += 1
      elif paper.raw_decision == 'pre-accept':
        da_topics = np.add(da_topics, paper.transformed)
        da_count += 1
      elif paper.decision == 'reject':
        r_topics = np.add(r_topics, paper.transformed)
        r_count += 1
      elif paper.decision == 'accept':
        a_topics = np.add(a_topics, paper.transformed)
        a_count += 1

    if dr_count > 0: dr_topics = dr_topics / float(dr_count)
    if da_count > 0: da_topics = da_topics / float(da_count)
    if r_count > 0: r_topics = r_topics / float(r_count)
    if a_count > 0: a_topics = a_topics / float(a_count)
    col_labels = TOPICS
    row_labels = ["Accept - Desk Rejects"]
    heatmap_arr = np.array([[int(round(100 * (a - dr), 0)) for dr, a in zip(dr_topics, a_topics)]], np.int)
    cmap = mpl.colors.ListedColormap(['red', 'lightsalmon', 'white', 'palegreen', 'lime'])
    bounds = [-10, -8, -2, 2, 8, 10]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    cax = plt.matshow(heatmap_arr, interpolation='nearest', cmap=cmap, norm=norm)
    for (i, j), z in np.ndenumerate(heatmap_arr):
      plt.text(j, i, abs(z), ha='center', va='center', fontsize=11)
    ticks = [-20, -10, 0, 10, 20]
    plt.colorbar(cax, cmap=cmap, norm=norm, boundaries=bounds, ticks=ticks)
    plt.xticks(np.arange(len(list(col_labels))), list(col_labels), rotation="vertical")
    plt.yticks(np.arange(len(list(row_labels))), list(row_labels))
    plt.savefig("classify/figs/desks-%s.png" % conf_id, bbox_inches='tight')


def reputation(only_first=False):
  if only_first:
    print("## First Authors Only")
  papers = read_papers()
  submissions = format_conf_acceptance(papers)
  author_map = top_authors(cite_graph(GRAPH_CSV))
  accepteds, rejecteds = [], []
  for conf_id, papers in submissions.items():
    accepted, rejected = [], []
    for paper in papers:
      for i, author in enumerate(paper.authors):
        if only_first and i > 0: break
        cites = 0
        if author in author_map:
          cites = author_map[author][1]
        if paper.decision == 'accept':
          accepted.append(cites)
        else:
          rejected.append(cites)
    print("#### %s" % conf_id)
    print("**Accepted** => Med: %0.2f, IQR: %0.2f, Min: %d, Max: %d" %
          (Metrics.median(accepted), Metrics.iqr(accepted), min(accepted), max(accepted)))
    print("**Rejected** => Med: %0.2f, IQR: %0.2f, Min: %d, Max: %d" %
          (Metrics.median(rejected), Metrics.iqr(rejected), min(rejected), max(rejected)))
    accepteds += accepted
    rejecteds += rejected
  print("#### All")
  print("**Accepted** => Med: %0.2f, IQR: %0.2f, Min: %d, Max: %d" %
        (Metrics.median(accepteds), Metrics.iqr(accepteds), min(accepteds), max(accepteds)))
  print("**Rejected** => Med: %0.2f, IQR: %0.2f, Min: %d, Max: %d" %
        (Metrics.median(rejecteds), Metrics.iqr(rejecteds), min(rejecteds), max(rejecteds)))


if __name__ == "__main__":
  # desk_rejects()
  reputation(True)
