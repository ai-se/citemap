from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True
from utils.lib import O
from network.mine import cite_graph, Miner
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import OrderedDict, Counter
import matplotlib.patches as mpatches
import db.mysql as mysql
import pandas as pd
import pickle
from matplotlib.colors import ColorConverter
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram
from prettytable import PrettyTable

GRAPH_CSV = "data/citemap_v4.csv"

# For 11 TOPICS
N_TOPICS = 11
ALPHA = 0.22359
BETA = 0.53915
ITERATIONS = 100
# TOPICS = ["TPC %d" % d for d in range(11)]
TOPICS = ["Design", "Testing", "Modelling", "Mobile", "Energy", "Defects",
          "SourceCode", "WebApps", "Configuration", "Developer", "Mining"]
TOPIC_THRESHOLD = 3

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
  row_labels = [str(ind) + "-" + name for ind, name in zip(range(lda_model.n_topics), TOPICS)]
  # row_labels = ["%2d" % ind for ind in range(lda_model.n_topics)]
  column_labels = [c.acronym for c in mysql.get_conferences() if c.id in valid_conferences]
  # Heatmap
  heatmap_arr = []
  for conference_id in sorted(conference_heatmaps.keys(), key=lambda x: int(x)):
    tot = sum(conference_heatmaps[conference_id])
    dist = [top / tot for top in conference_heatmaps[conference_id]]
    heatmap_arr.append(dist)
  report(lda_model, vocab, 15)
  # make_dendo_heatmap(np.transpose(heatmap_arr), row_labels, column_labels,
  #                    "figs/v2/diversity/%s_dend.png" % fig_name, dend_settings)
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
        if len(author_ids.intersection(committee)) == 0:
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
  col_labels = [c.acronym for c in mysql.get_conferences() if c.id in valid_conferences]
  make_dendo_heatmap(np.transpose(heatmap_arr), row_labels, col_labels, "figs/v2/pc/%s.png" % fig_name, dendo_settings)


def pc_heatmap_delta(fig_name, title=None, paper_range=None):
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
  pc_conference_topics = {}
  for conference in mysql.get_conferences():
    year_committees = index_by_year(p_committees[conference.id])
    year_papers = index_by_year(p_conferences[conference.id])
    topics = np.array([0] * lda_model.n_topics)
    pc_topics = np.array([0] * lda_model.n_topics)
    for year in sorted(year_committees.keys(), key=lambda y: int(y)):
      if (paper_range is not None) and (int(year) not in paper_range):
        continue
      papers = year_papers.get(year, None)
      if papers is None:
        continue
      committee = year_committees[year]
      for paper_id in papers:
        paper = graph.paper_nodes[paper_id]
        author_ids = set(paper.author_ids.strip().split(","))
        paper_topics = miner.documents[paper_id].topics_count
        if len(author_ids.intersection(committee)) != 0:
          pc_topics = np.add(pc_topics, paper_topics)
        topics = np.add(topics, paper_topics)
    pc_conference_topics[conference.id] = pc_topics
    conference_topics[conference.id] = topics
  heatmap_arr = []
  valid_conferences = []
  for conference_id in sorted(conference_topics.keys(), key=lambda x: int(x)):
    tot = sum(conference_topics[conference_id])
    pc_tot = sum(pc_conference_topics[conference_id])
    if tot <= 0 or pc_tot <= 0:
      continue
    valid_conferences.append(conference_id)
    dist = [top / tot for top in conference_topics[conference_id]]
    pc_dist = [top / pc_tot for top in pc_conference_topics[conference_id]]
    # heatmap_arr.append([round(pc_d - d, 2) for d, pc_d in zip(dist, pc_dist)])
    heatmap_arr.append([int(round(100 * (pc_d - d), 0)) for d, pc_d in zip(dist, pc_dist)])

  # HeatMap
  row_labels = ["%2d" % ind for ind in range(lda_model.n_topics)]
  col_labels = [c.acronym for c in mysql.get_conferences() if c.id in valid_conferences]
  heatmap_arr = np.transpose(np.array(heatmap_arr, np.int))
  plt.figure(figsize=(4, 3))
  cmap = mpl.colors.ListedColormap(['red', 'lightsalmon', 'white', 'palegreen','lime'])
  bounds = [-20, -12, -5, 5, 12, 20]
  # bounds = [-0.2, -0.12, -0.05, 0.05, 0.12, 0.2]
  norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
  # df = pd.DataFrame(heatmap_arr, columns=col_labels, index=row_labels)
  cax = plt.matshow(heatmap_arr, interpolation='nearest', cmap=cmap, norm=norm)
  for (i, j), z in np.ndenumerate(heatmap_arr):
    plt.text(j, i, abs(z), ha='center', va='center', fontsize=11)
  # ticks = [-0.2, -0.1, 0, 0.1, 0.2]
  ticks = [-20, -10, 0, 10, 20]
  plt.colorbar(cax, cmap=cmap, norm=norm, boundaries=bounds, ticks=ticks)
  plt.xticks(np.arange(len(list(col_labels))), list(col_labels), rotation="vertical")
  plt.yticks(np.arange(len(list(row_labels))), list(row_labels))
  if title is None:
    title = "Topic Distribution Delta between papers by PC and all papers"
  plt.title(title, y=1.2)
  plt.savefig("figs/v2/pc/%s.png" % fig_name, bbox_inches='tight')
  plt.clf()


def get_top_papers(file_name, min_year=None):
  miner, graph, lda_model, vocab = get_graph_lda_data()
  top_papers = {index: [] for index in xrange(N_TOPICS)}
  for paper_id, paper in graph.paper_nodes.items():
    topics = miner.documents[paper_id].topics_count
    if min_year is not None and int(paper.year) < min_year: continue
    if max(topics) == 0:
      continue
    topic = topics.argmax()
    # cites = len(paper.cites.split(",")) if paper.cites else 0
    cites = paper.cited_counts
    top_papers[topic].append([(cites, paper.title, paper.authors, paper.year)])
  with open(file_name, 'wb') as f:
    for index in range(N_TOPICS):
      top_papers[index] = sorted(top_papers[index], reverse=True)[:4]
      f.write("*** %d ***\n" % index)
      for paper in top_papers[index]:
        paper = paper[0]
        f.write("%s %s - %s, %s\n" % (paper[0], paper[-1], paper[1], paper[2]))


def top_authors(graph, top_percent=0.01, min_year=None):
  authors = graph.get_papers_by_authors()
  author_cites = []
  if top_percent is None:
    top_percent = 1
  for author_id, papers in authors.items():
    cite_count = 0
    for paper_id, year, __ in papers:
      if min_year is not None and int(year) < min_year: continue
      cited = graph.paper_nodes[paper_id].cited_counts
      if cited:
        # cite_count += len(cited.split(","))
        cite_count += cited
    author_cites.append((author_id, cite_count))
  tops = sorted(author_cites, key=lambda x: x[1], reverse=True)[:int(top_percent*len(author_cites))]
  return set([t[0] for t in tops])


def print_top_authors(file_name, top_percent=None, min_year=None):
  graph = cite_graph(GRAPH_CSV)
  tops = top_authors(graph, top_percent=top_percent, min_year=min_year)
  author_papers = graph.get_papers_by_authors()
  top_tups = []
  for author_id, author in graph.author_nodes.items():
    if author_id in tops:
      papers = author_papers.get(author_id, None)
      if papers is None: continue
      total_cites = 0
      counts = 0
      for paper_tup in papers:
        if min_year is not None and int(paper_tup[1]) < min_year: continue
        paper_id = paper_tup[0]
        total_cites += graph.paper_nodes[paper_id].cited_counts
        counts += 1
      top_tups.append((author.name, counts, total_cites))
  top_tups = sorted(top_tups, key=lambda x: x[-1], reverse=True)
  with open(file_name, "wb") as f:
    for top_tup in top_tups:
      f.write(str(top_tup))
      f.write("\n")


def super_author(top_percents):
  miner, graph, lda_model, vocab = get_graph_lda_data()
  authors = graph.get_papers_by_authors()
  for top_percent in top_percents:
    author_topics = {}
    tops = top_authors(graph, top_percent)
    for author_id, papers in authors.items():
      if author_id not in tops:
        continue
      topics = [0] * lda_model.n_topics
      for paper_id, _, __ in papers:
        document = miner.documents[paper_id]
        for index, topic_count in enumerate(document.topics_count):
          if topic_count >= TOPIC_THRESHOLD:
            topics[index] = 1
      author_topics[author_id] = sum(topics)
    vals = sorted(author_topics.values(), reverse=True)
    counter = Counter()
    for val in vals:
      counter[val] += 1
    bar_x = []
    bar_y = []
    for key in sorted(counter.keys()):
      bar_x.append(key)
      bar_y.append(counter[key])
    print(bar_x, bar_y)


def plot_super_authors(fig_name):
  y_1 = [0, 1, 1, 1, 5, 4, 8, 12, 24, 31, 29, 47]
  y_10 = [0, 32, 72, 183, 180, 202, 199, 224, 187, 156, 114, 82]
  y_20 = [0, 85, 256, 480, 552, 492, 374, 357, 264, 189, 129, 84]
  y_all = [1358, 3921, 2336, 2631, 2272, 1527, 876, 586, 360, 219, 140, 88]
  ind = np.arange(1, len(y_1) + 1)  # the x locations for the groups
  fig = plt.figure(figsize=(8, 2))
  width = 0.2  # the width of the bars
  fig, ax = plt.subplots()
  rects_1 = ax.bar(ind, y_1, width, color='orange')
  rects_10 = ax.bar(ind + width, y_10, width, color='cyan')
  rects_20 = ax.bar(ind + 2 * width, y_20, width, color='green')
  rects_all = ax.bar(ind + 3 * width, y_all, width, color='hotpink')
  ax.legend((rects_1[0], rects_10[0], rects_20[0], rects_all[0]), ('1%', '10%', '20%', '100%'), loc='upper center',
            bbox_to_anchor=(0.5, 1.12), ncol=4, fontsize=16)
  plt.xticks(ind + 2 * width, ind, fontsize=16)
  plt.xlabel("Cumulative # of Topics", fontsize=16)
  plt.ylabel("Authors Count", fontsize=16)
  plt.yticks(fontsize=16)
  plt.yscale('log')
  plt.savefig(fig_name)
  plt.clf()


def pc_paper_count_table():
  def index_by_year(tups):
    y_comm = {}
    for tup in tups:
      comm = y_comm.get(tup[1], set())
      comm.add(tup[0])
      y_comm[tup[1]] = comm
    return y_comm

  graph = cite_graph(GRAPH_CSV)
  start = 2009
  p_conferences = graph.get_papers_by_conference()
  p_committees = graph.get_committee_by_conference()
  header = ["conf", "# Accepted", "# from PC", "Percentage"]
  table = PrettyTable(header)
  all_papers, all_pc_papers = 0, 0
  for conference in mysql.get_conferences():
    year_committees = index_by_year(p_committees[conference.id])
    year_papers = index_by_year(p_conferences[conference.id])
    year_scores = {}
    comm_papers = 0
    tot_papers = 0
    for year in sorted(year_committees.keys(), key=lambda y: int(y)):
      if int(year) < start: continue
      papers = year_papers.get(year, None)
      if papers is None:
        year_scores[int(year)] = None
        continue
      committee = year_committees[year]
      for paper_id in papers:
        paper = graph.paper_nodes[paper_id]
        author_ids = set(paper.author_ids.strip().split(","))
        if author_ids.intersection(committee):
          comm_papers += 1
        tot_papers += 1
    table.add_row([conference.acronym, tot_papers, comm_papers, int(round(100 * comm_papers / tot_papers, 0))])
    all_papers += tot_papers
    all_pc_papers += comm_papers
  table.add_row(["all", all_papers, all_pc_papers, int(round(100 * all_pc_papers / all_papers, 0))])
  print("```")
  print(table)
  print("```")


def paper_bar():
  graph = cite_graph(GRAPH_CSV)
  conferences = graph.get_papers_by_conference()
  start = 2001
  end = 2015
  year_count = {}
  for year in range(start, end + 1):
    year_count[year] = 0
  for conference_id, papers in conferences.items():
    for tup in papers:
      count = year_count.get(int(tup[1]), None)
      if count is None: continue
      year_count[int(tup[1])] += 1
  bar_x, bar_y = [], []
  for year, count in year_count.items():
    bar_x.append(year)
    bar_y.append(count)
  fig = plt.figure(figsize=(8, 3))
  plt.bar(bar_x, bar_y, color='blue', align='center')
  plt.xlim([start - 1, end + 1])
  plt.xticks(bar_x, rotation=45)
  plt.ylim(300, 1100)
  plt.xlabel('Year')
  plt.ylabel('# of Papers')
  plt.savefig("figs/v2/paper_count.png", bbox_inches='tight')
  plt.clf()


if __name__ == "__main__":
  # conference_diversity("heatmap_93_00", dendo_11_settings, range(1993, 2001))
  # conference_diversity("heatmap_01_08", dendo_14_settings, range(2001, 2009))
  # conference_diversity("heatmap_09_16", dendo_16_settings, range(2009, 2017))
  # conference_diversity("heatmap_all", dendo_16_settings)
  # pc_topics_heatmap("pc_heatmap_09_16", dendo_16_settings, range(2009, 2017))
  # topic_evolution()
  # pc_heatmap_delta("delta", "Topic Dist. Delta between papers by PC and all papers(2009-2016)", range(2009, 2016))
  # get_top_papers("figs/v2/stats/top_papers.txt")
  # get_top_papers("figs/v2/stats/top_papers_recent.txt", 2009)
  # print_top_authors("figs/v2/stats/top_authors.txt", 0.01)
  # print_top_authors("figs/v2/stats/top_authors_recent.txt", 0.01, 2009)
  plot_super_authors("figs/v2/super_author/all_authors.png")
  # pc_paper_count_table()
  # paper_bar()
