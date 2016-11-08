from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True
from utils.lib import O
from network.mine import cite_graph, Miner
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import matplotlib.patches as mpatches
import db.mysql as mysql
import pandas as pd
import pickle
from matplotlib.colors import ColorConverter
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram

GRAPH_CSV = "data/citemap_v4.csv"

# For 11 TOPICS
N_TOPICS = 11
ALPHA = 0.22359
BETA = 0.53915
ITERATIONS = 100
# TOPICS = ["TPC %d" % d for d in range(11)]
TOPICS = ["Design", "Testing", "Modelling", "Mobile", "Energy", "Defects",
          "SourceCode", "WebApps", "SPL", "Developer", "RE"]

dendo_11_settings = O(
    fig_size=(6, 8),
    col_axes=[0.25, 0.65, 0.50, 0.11],
    row_axes=[0.0, 0.215, 0.2, 0.375],
    plot_axes=[0.25, 0.1, 0.63, 0.6],
)


dendo_14_settings = O(
    fig_size=(8, 8),
    col_axes=[0.25, 0.65, 0.50, 0.11],
    row_axes=[0.0, 0.2, 0.21, 0.4],
    plot_axes=[0.25, 0.1, 0.63, 0.6],
)

dendo_16_settings = O(
    fig_size=(8, 8),
    col_axes=[0.25, 0.65, 0.50, 0.11],
    row_axes=[0.0, 0.225, 0.21, 0.35],
    plot_axes=[0.25, 0.1, 0.63, 0.6],
)


def make_dendo_heatmap(arr, row_labels, column_labels, figname, settings):
  df = pd.DataFrame(arr, columns=column_labels, index=row_labels)
  # Compute pairwise distances for columns
  col_clusters = linkage(pdist(df.T, metric='euclidean'), method='complete')
  # plot column dendrogram
  fig = plt.figure(figsize=settings.fig_size)
  axd2 = fig.add_axes(settings.col_axes)
  col_dendr = dendrogram(col_clusters, orientation='top',
                         color_threshold=np.inf)  # makes dendrogram black)
  axd2.set_xticks([])
  axd2.set_yticks([])
  # plot row dendrogram
  axd1 = fig.add_axes(settings.row_axes)
  row_clusters = linkage(pdist(df, metric='euclidean'), method='complete')
  row_dendr = dendrogram(row_clusters, orientation='left',
                         count_sort='ascending',
                         color_threshold=np.inf)  # makes dendrogram black
  axd1.set_xticks([])
  axd1.set_yticks([])
  # remove axes spines from dendrogram
  for i, j in zip(axd1.spines.values(), axd2.spines.values()):
    i.set_visible(False)
    j.set_visible(False)
  # reorder columns and rows with respect to the clustering
  df_rowclust = df.ix[row_dendr['leaves'][::-1]]
  df_rowclust.columns = [df_rowclust.columns[col_dendr['leaves']]]
  # plot heatmap
  axm = fig.add_axes(settings.plot_axes)
  cax = axm.matshow(df_rowclust, interpolation='nearest', cmap='hot_r')
  fig.colorbar(cax)
  axm.set_xticks(np.arange(len(list(df_rowclust.columns))))
  axm.set_xticklabels(list(df_rowclust.columns), rotation="vertical")
  axm.set_yticks(np.arange(len(list(df_rowclust.index))))
  axm.set_yticklabels(list(df_rowclust.index))
  plt.savefig(figname, bbox_inches='tight')
  plt.clf()


def make_heatmap(arr, row_labels, column_labels, figname):
  plt.figure(figsize=(4, 3))
  df = pd.DataFrame(arr, columns=column_labels, index=row_labels)
  cax = plt.matshow(df, interpolation='nearest', cmap='hot_r')
  plt.colorbar(cax)
  plt.xticks(np.arange(len(list(df.columns))), list(df.columns), rotation="vertical")
  plt.yticks(np.arange(len(list(df.index))), list(df.index))
  plt.title("Topics to Conference Distribution", y=1.2)
  plt.savefig(figname, bbox_inches='tight')
  plt.clf()


COLOR_CONVERTER = ColorConverter()


def get_color(index):
  colors_7 = ["lightgray", "red", "blue", "darkslategray",
              "yellow", "darkmagenta", "cyan", "saddlebrown",
              "orange", "lime", "hotpink"]
  return COLOR_CONVERTER.to_rgb(colors_7[index])


def get_spaced_colors(n):
  max_value = 16581375  # 255**3
  interval = int(max_value / n)
  colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]
  return [(int(index[:2], 16) / 255.0, int(index[2:4], 16) / 255.0, int(index[4:], 16) / 255.0) for index in colors]


def percent_sort(arr):
  sum_arr = sum(arr)
  sum_arr = sum_arr if sum_arr else 0.0001
  tmp = [(i, round(t * 100 / sum_arr, 2)) for i, t in enumerate(arr)]
  return sorted(tmp, key=lambda yp: yp[1], reverse=True)


def yearize(paps):
  paps = sorted(paps, key=lambda tup: tup[1], reverse=True)
  pap_dict = {}
  for pap in paps:
    year_paps = pap_dict.get(int(pap[1]), [])
    year_paps.append(pap[0])
    pap_dict[int(pap[1])] = year_paps
  return OrderedDict(sorted(pap_dict.items(), key=lambda t: t[0]))


def report(lda_model, vocab, n_top_words=10):
  for index, topic_dist in enumerate(lda_model.topic_word_):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
    print('Topic {}: {}'.format(index, ', '.join(topic_words)))


def save_object(obj, filename):
  with open(filename, 'wb') as f:
    pickle.dump(obj, f)


def load_object(filename):
  with open(filename, 'rb') as f:
    return pickle.load(f)


def get_graph_lda_data():
  graph = cite_graph(GRAPH_CSV)
  miner = Miner(graph)
  lda_model, vocab = miner.lda(N_TOPICS, n_iter=ITERATIONS, alpha=ALPHA, beta=BETA)
  return miner, graph, lda_model, vocab


def conference_diversity(fig_name, dend_settings, paper_range=None):
  miner, graph, lda_model, vocab = get_graph_lda_data()
  conferences = graph.get_papers_by_conference()
  conference_topics = {}
  conference_heatmaps = {}
  valid_conferences = []
  for conference_id, papers in conferences.items():
    topics = np.array([0] * lda_model.n_topics)
    for tup in yearize(papers).items():
      if paper_range and tup[0] not in paper_range: continue
      for paper_id in tup[1]:
        topics = np.add(topics, miner.documents[paper_id].topics_count)
    if sum(topics) > 0:
      conference_topics[conference_id] = percent_sort(topics)
      conference_heatmaps[conference_id] = topics
      valid_conferences.append(conference_id)
  # row_labels = [str(ind) + "-" + name for ind, name in zip(range(lda_model.n_topics), TOPICS)]
  row_labels = ["%2d" % ind for ind in range(lda_model.n_topics)]
  column_labels = [c.acronym for c in mysql.get_conferences() if c.id in valid_conferences]
  # Heatmap
  heatmap_arr = []
  for conference_id in sorted(conference_heatmaps.keys(), key=lambda x: int(x)):
    tot = sum(conference_heatmaps[conference_id])
    dist = [top / tot for top in conference_heatmaps[conference_id]]
    heatmap_arr.append(dist)
  report(lda_model, vocab)
  make_dendo_heatmap(np.transpose(heatmap_arr), row_labels, column_labels,
                     "figs/v2/diversity/%s_dend.png" % fig_name, dend_settings)
  make_heatmap(np.transpose(heatmap_arr), row_labels, column_labels, "figs/v2/diversity/%s.png" % fig_name)


def topic_evolution():
  miner, graph, lda_model, vocab = get_graph_lda_data()
  paper_nodes = graph.paper_nodes
  topics_map = {}
  n_topics = lda_model.n_topics
  for paper_id, paper in paper_nodes.items():
    if int(paper.year) < 1993: continue
    document = miner.documents[paper_id]
    year_topics = topics_map.get(paper.year, np.array([0] * n_topics))
    topics_map[paper.year] = np.add(year_topics, document.topics_count)
  yt_map = {}
  for year, t_count in topics_map.items():
    yt_map[year] = percent_sort(t_count)
  width = 0.8
  plts = []
  x_axis = np.arange(1, len(yt_map.keys()) + 1)
  y_offset = np.array([0] * len(yt_map.keys()))
  colors_dict = {}
  top_topic_count = 9
  for index in range(top_topic_count):
    bar_val, color = [], []
    for year in sorted(yt_map.keys(), key=lambda x: int(x)):
      topic = yt_map[year][index]
      colors_dict[topic[0]] = get_color(topic[0])
      color.append(colors_dict[topic[0]])
      bar_val.append(topic[1])
    plts.append(plt.bar(x_axis, bar_val, width, color=color, bottom=y_offset))
    y_offset = np.add(y_offset, bar_val)
  plt.ylabel("Topic %")
  plt.xlabel("Year")
  plt.xticks(x_axis + width / 2, [str(y)[2:] for y in sorted(yt_map.keys(), key=lambda x: int(x))], fontsize=9)
  plt.yticks(np.arange(0, 101, 10))
  plt.ylim([0, 101])
  # Legends
  patches = []
  for index, (topic, color) in enumerate(colors_dict.items()):
    patches.append(mpatches.Patch(color=color, label='Topic %s' % str(topic)))
  plt.legend(tuple(patches), tuple(TOPICS), loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=6, fontsize=10,
             handlelength=0.7)
  plt.savefig("figs/v2/topic_evolution/topic_evolution_7.png")
  plt.clf()
  report(lda_model, vocab)


def pc_topics_heatmap(fig_name, dendo_settings, paper_range=None):
  def index_by_year(tups):
    y_comm = {}
    for tup in tups:
      comm = y_comm.get(tup[1], set())
      comm.add(tup[0])
      y_comm[tup[1]] = comm
    return y_comm

  miner, graph, lda_model, vocab = get_graph_lda_data()
  p_conferences = graph.get_papers_by_conference()
  p_committees = graph.get_committee_by_conference()
  conference_topics = {}
  for conference in mysql.get_conferences():
    year_committees = index_by_year(p_committees[conference.id])
    year_papers = index_by_year(p_conferences[conference.id])
    year_scores = {}
    topics = np.array([0] * lda_model.n_topics)
    for year in sorted(year_committees.keys(), key=lambda y: int(y)):
      if (paper_range is not None) and (int(year) not in paper_range):
        continue
      papers = year_papers.get(year, None)
      if papers is None:
        year_scores[int(year)] = None
        continue
      committee = year_committees[year]
      for paper_id in papers:
        paper = graph.paper_nodes[paper_id]
        author_ids = set(paper.author_ids.strip().split(","))
        if author_ids.intersection(committee):
          continue
        topics = np.add(topics, miner.documents[paper_id].topics_count)
    conference_topics[conference.id] = topics
  heatmap_arr = []
  valid_conferences = []
  for conference_id in sorted(conference_topics.keys(), key=lambda x: int(x)):
    tot = sum(conference_topics[conference_id])
    if tot > 0:
      valid_conferences.append(conference_id)
    dist = [top / tot for top in conference_topics[conference_id]]
    heatmap_arr.append(dist)
  row_labels = ["%2d" % ind for ind in range(lda_model.n_topics)]
  column_labels = [c.acronym for c in mysql.get_conferences() if c.id in valid_conferences]
  make_dendo_heatmap(np.transpose(heatmap_arr), row_labels, column_labels, "figs/v2/pc/%s.png" % fig_name, dendo_settings)


if __name__ == "__main__":
  # conference_diversity("heatmap_93_00", dendo_11_settings, range(1993, 2001))
  # conference_diversity("heatmap_01_08", dendo_14_settings, range(2001, 2009))
  # conference_diversity("heatmap_09_16", dendo_16_settings, range(2009, 2017))
  # pc_topics_heatmap("pc_heatmap_09_16", dendo_16_settings, range(2009, 2017))
  topic_evolution()
