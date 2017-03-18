from __future__ import print_function, division
import os, sys
sys.path.append(os.path.abspath("."))
from network.mine import cite_graph, Miner
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import matplotlib.patches as mpatches
from collections import Counter, OrderedDict
import numpy as np
import db.mysql as mysql
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram
from prettytable import PrettyTable
sys.dont_write_bytecode=True
__author__ = "panzer"

#COLORS = sorted(clrs.cnames.keys())
TOPIC_THRESHOLD = 3
# GRAPH_CSV = "data/citemap.csv"

"""
# For 7 TOPICS
TOPICS = 7
ALPHA = 0.847433736937
BETA = 0.763774618977
"""

# For 11 TOPICS
TOPICS = 11
ALPHA = 0.22359
BETA = 0.53915


GRAPH_CSV = "data/citemap_v4.csv"

# COLORS = ["#808080", "#000000", "#FF0000", "#800000",
#           "#FFFF00", "#808000", "#00FF00", "008000",
#           "#00FFFF", "#008080", "#0000FF", "#000080",
#           "#000080", "#FF00FF", "#800080", "#C0C0C0"]

COLORS = ["grey", "black", "red", "maroon", "yellow",
          "olive", "lime", "green", "aqua", "teal",
          "blue", "navy", "fuchisa", "purple", "silver"]

tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

HEX_COLORS = ["F70A41", "4C00FF", "B39500", "07F7EB",
              "F707E3", "F78307", "076FF7", "FFFF00",
              "8C00FF", "6D706D", "FAD2DF", "32F20C"]
RGB_COLORS = [(int(h[:2], 16)/255.0, int(h[2:4], 16)/255.0, int(h[4:], 16)/255.0) for h in HEX_COLORS]
for i in range(len(tableau20)):
  r, g, b = tableau20[i]
  tableau20[i] = (r / 255., g / 255., b / 255.)

def get_spaced_colors(n):
  max_value = 16581375  # 255**3
  interval = int(max_value / n)
  colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]
  return [(int(index[:2], 16), int(index[2:4], 16), int(index[4:], 16)) for index in colors]

COLORS = get_spaced_colors(12)

#def get_color(index): return str(COLORS[index])

#def get_color(index): return tableau20[index]

# def get_color(index): return RGB_COLORS[index]

COLORS_7 = ["grey", "red", "blue", "green",
            "yellow", "magenta", "cyan", "black"]
def get_color(index):
  return COLORS_7[index]




def top_authors(graph, top_percent = 0.01, min_year=None):
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

def super_author(fig_prefix="super_author" ,top_percent=1.00):
  graph = cite_graph(GRAPH_CSV)
  miner = Miner(graph)
  lda_model, vocab = miner.lda(7, n_iter=100, alpha=0.847433736937, beta=0.763774618977)
  authors = graph.get_papers_by_authors()
  author_topics = {}
  tops = top_authors(graph, top_percent)
  for author_id, papers in authors.items():
    if author_id not in tops:
      continue
    topics = [0]*lda_model.n_topics
    for paper_id, _, __ in papers:
      document = miner.documents[paper_id]
      for index, topic_count in enumerate(document.topics_count):
        if topic_count >= TOPIC_THRESHOLD:
          topics[index] = 1
    author_topics[author_id] = sum(topics)
  vals = sorted(author_topics.values(), reverse=True)
  # x_axis = range(1, len(vals) + 1)
  # plt.ylabel("Topic Count")
  # plt.xlabel("Author ID")
  # plt.title("Super Author")
  # plt.ylim(min(vals)-1, max(vals)+1)
  # plt.plot(x_axis, vals)
  # plt.savefig("figs/super_author/%s.png"%fig_prefix)
  # plt.clf()
  fig = plt.figure(figsize=(8, 2), dpi=100)
  counter = Counter()
  for val in vals:
    counter[val] += 1
  bar_x = []
  bar_y = []
  for key in sorted(counter.keys()):
    bar_x.append(key)
    bar_y.append(counter[key])
  print(bar_x, bar_y)
  return
  fig, ax = plt.subplots()
  width = 2/3
  ax.bar(bar_x, bar_y, 2/3, color='blue', align='center')
  ax.set_xticks(np.arange(1,lda_model.n_topics+1))
  ax.set_xticklabels(np.arange(1,lda_model.n_topics+1))
  # for i, v in zip(bar_x,bar_y):
  #   ax.text(i, v + 0.25, str(v), color='red', fontweight='bold', fontsize=11, horizontalalignment='center')
  plt.xlabel("Topics")
  plt.ylabel("Authors Count")
  # plt.ylim(min(bar_y) - 1, max(bar_y) + 1)
  plt.savefig("figs/super_author/%s_bar.png"%fig_prefix)
  plt.clf()
  n_top_words = 10
  for index, topic_dist in enumerate(lda_model.topic_word_):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
    print('Topic {}: {}'.format(index, ', '.join(topic_words)))

def yearize(paps):
  paps = sorted(paps, key=lambda tup: tup[1], reverse=True)
  pap_dict = {}
  for pap in paps:
    year_paps = pap_dict.get(int(pap[1]), [])
    year_paps.append(pap[0])
    pap_dict[int(pap[1])] = year_paps
  return OrderedDict(sorted(pap_dict.items(), key=lambda t: t[0]))

def percent_sort(arr):
  sum_arr = sum(arr)
  sum_arr = sum_arr if sum_arr else 0.0001
  tmp = [(i, round(t * 100 / sum_arr, 2)) for i, t in enumerate(arr)]
  return sorted(tmp, key=lambda yp: yp[1], reverse=True)


def conference_diversity():
  graph = cite_graph(GRAPH_CSV)
  miner = Miner(graph)
  lda_model, vocab = miner.lda(7, n_iter=100, alpha=0.847433736937, beta=0.763774618977)
  conferences = graph.get_papers_by_venue()
  conference_topics = {}
  conference_heatmaps = {}
  for conference_id, papers in conferences.items():
    topics = np.array([0] * lda_model.n_topics)
    for tup in yearize(papers).items():
      for paper_id in tup[1]:
        topics = np.add(topics, miner.documents[paper_id].topics_count)
    conference_topics[conference_id] = percent_sort(topics)
    conference_heatmaps[conference_id] = topics
  #fig, ax = plt.subplots()
  bar_vals = []
  colors = []
  width = 0.75
  plts = []
  x_axis = np.arange(1, len(conference_topics.keys())+1)
  #x_axis = [c.acronym for c in mysql.get_conferences()]
  y_offset = np.array([0]*len(conference_topics.keys()))
  colors_dict = {}
  for index in range(7):
    bar_val = []
    color = []
    for conference_id in sorted(conference_topics.keys(), key=lambda x: int(x)):
      topic = conference_topics[conference_id][index]
      colors_dict[topic[0]] = get_color(topic[0])
      color.append(colors_dict[topic[0]])
      bar_val.append(topic[1])
    plts.append(plt.bar(x_axis, bar_val, width, color=color, bottom=y_offset))
    y_offset = np.add(y_offset, bar_val)
  plt.ylabel("Topic Coverage %")
  plt.xlabel("Conferences")
  plt.xticks(x_axis+width/2, [c.acronym for c in mysql.get_conferences()])
  plt.yticks(np.arange(0, 101, 10))
  plt.ylim([0,101])
  #Legends
  patches = []
  for topic, color in colors_dict.items():
    patches.append(mpatches.Patch(color=color, label='Topic %s' % str(topic)))
  plt.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=6, fontsize=7)
  plt.savefig("figs/diversity/conference_diversity_7topics.png")
  plt.clf()
  n_top_words = 10
  #Heatmap
  heatmap_arr = []
  for conference_id in sorted(conference_heatmaps.keys(), key=lambda x: int(x)):
    tot = sum(conference_heatmaps[conference_id])
    dist = [top/tot for top in conference_heatmaps[conference_id]]
    heatmap_arr.append(dist)
  fig, ax = plt.subplots()
  heatmap_arr = np.array(heatmap_arr)
  heatmap = ax.pcolor(heatmap_arr, cmap=plt.cm.Reds)
  plt.ylabel("Conferences")
  plt.xlabel("Topics")
  # row_labels = range(lda_model.n_topics)
  t_names= ["Testing", "Applications", "Program Analysis", "Tools and Projects",
            "Defect Analysis", "Modeling", "Maintenance"]
  row_labels = [str(ind) + "-" + name for ind, name in zip(range(lda_model.n_topics), t_names)]
  column_labels = [c.acronym for c in mysql.get_conferences()]
  ax.set_xticks(np.arange(heatmap_arr.shape[1])+0.5, minor=False)
  ax.set_yticks(np.arange(heatmap_arr.shape[0])+0.5, minor=False)
  ax.set_xticklabels(row_labels, minor=False)
  ax.set_yticklabels(column_labels, minor=False)
  plt.savefig("figs/diversity/heatmap_7topics.png")
  plt.clf()
  for index, topic_dist in enumerate(lda_model.topic_word_):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
    print('Topic {}: {}'.format(index, ', '.join(topic_words)))
  # make_dendo_heatmap(np.transpose(heatmap_arr), row_labels, column_labels, "figs/diversity/dend_heatmap_7topics.png")
  make_heatmap(np.transpose(heatmap_arr), row_labels, column_labels, "figs/diversity/heatmap2.png")


def conference_evolution_2(paper_range, figname):
  graph = cite_graph(GRAPH_CSV)
  miner = Miner(graph)
  lda_model, vocab = miner.lda(7, n_iter=100, alpha=0.847433736937, beta=0.763774618977)
  conferences = graph.get_papers_by_venue()
  conference_topics = {}
  conference_heatmaps = {}
  for conference_id, papers in conferences.items():
    topics = np.array([0] * lda_model.n_topics)
    for tup in yearize(papers).items():
      if tup[0] not in paper_range:
        continue
      for paper_id in tup[1]:
        topics = np.add(topics, miner.documents[paper_id].topics_count)
    conference_topics[conference_id] = percent_sort(topics)
    conference_heatmaps[conference_id] = topics
  n_top_words = 10
  #Heatmap
  heatmap_arr = []
  column_labels = []
  for conference_id, conf in zip(sorted(conference_heatmaps.keys(), key=lambda x: int(x)), mysql.get_conferences()):
    tot = sum(conference_heatmaps[conference_id])
    if tot == 0: continue
    column_labels.append(conf.acronym)
    dist = [top/tot for top in conference_heatmaps[conference_id]]
    heatmap_arr.append(dist)
  fig, ax = plt.subplots()
  heatmap_arr = np.array(heatmap_arr)
  heatmap = ax.pcolor(heatmap_arr, cmap=plt.cm.Reds)
  plt.ylabel("Conferences")
  plt.xlabel("Topics")
  row_labels = range(lda_model.n_topics)
  ax.set_xticks(np.arange(heatmap_arr.shape[1])+0.5, minor=False)
  ax.set_yticks(np.arange(heatmap_arr.shape[0])+0.5, minor=False)
  ax.set_xticklabels(row_labels, minor=False)
  ax.set_yticklabels(column_labels, minor=False)
  plt.savefig("figs/diversity/heatmap_7topics.png")
  plt.clf()
  for index, topic_dist in enumerate(lda_model.topic_word_):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
    print('Topic {}: {}'.format(index, ', '.join(topic_words)))
  # make_dendo_heatmap(np.transpose(heatmap_arr), row_labels, column_labels, "figs/diversity/dend_heatmap_7topics.png")
  make_dendo_heatmap(np.transpose(heatmap_arr), row_labels, column_labels, "figs/evolution/%s.png"%figname)



def make_heatmap(arr, row_labels, column_labels, figname):
  plt.figure(figsize=(4,3))
  df = pd.DataFrame(arr, columns=column_labels, index=row_labels)
  cax = plt.matshow(df, interpolation='nearest', cmap='hot_r')
  plt.colorbar(cax)
  plt.xticks(np.arange(len(list(df.columns))), list(df.columns), rotation="vertical")
  plt.yticks(np.arange(len(list(df.index))), list(df.index))
  plt.title("Topics to Conference Distribution", y=1.2)
  plt.savefig(figname, bbox_inches='tight')
  plt.clf()


fig_size_12 = (8, 8)
col_axes_12 = [0.25, 0.75, 0.50, 0.10]
row_axes_12 = [0.0, 0.123, 0.21, 0.555]
plot_axes_12 = [0.25, 0.1, 0.63, 0.6]
fig_size_7 = (7, 4)
col_axes_7 = [0.25, 0.78, 0.50, 0.16]
row_axes_7 = [0.0, 0.075, 0.21, 0.56]
plot_axes_7 = [0.25, 0.05, 0.63, 0.6]
col_axes_7_2 = [0.26, 0.8, 0.45, 0.16]
row_axes_7_2 = [0.0, 0.05, 0.21, 0.6]
plot_axes_7_2 = [0.2, 0.05, 0.63, 0.6]
def make_dendo_heatmap(arr, row_labels, column_labels, figname):
  df = pd.DataFrame(arr, columns=column_labels, index=row_labels)
  # Compute pairwise distances for columns
  col_clusters = linkage(pdist(df.T, metric='euclidean'), method='complete')
  # plot column dendrogram
  fig = plt.figure(figsize=fig_size_7)
  axd2 = fig.add_axes(col_axes_7)
  col_dendr = dendrogram(col_clusters, orientation='top',
                         color_threshold=np.inf)  # makes dendrogram black)
  axd2.set_xticks([])
  axd2.set_yticks([])
  # plot row dendrogram
  axd1 = fig.add_axes(row_axes_7)
  row_clusters = linkage(pdist(df, metric='euclidean'),  method='complete')
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
  axm = fig.add_axes(plot_axes_7)
  cax = axm.matshow(df_rowclust, interpolation='nearest', cmap='hot_r')
  fig.colorbar(cax)
  print(df_rowclust.columns)
  axm.set_xticks(np.arange(len(list(df_rowclust.columns))))
  axm.set_xticklabels(list(df_rowclust.columns), rotation="vertical")
  axm.set_yticks(np.arange(len(list(df_rowclust.index))))
  axm.set_yticklabels(list(df_rowclust.index))
  plt.savefig(figname, bbox_inches='tight')
  plt.clf()


def conference_evolution():
  legit_conferences = ["ICSE", "MSR", "FSE", "ASE"]
  non_legit_conferences = ["GPCE", "FASE"]
  TOP_TOPIC_COUNT = 7
  graph = cite_graph(GRAPH_CSV)
  miner = Miner(graph)
  lda_model, vocab = miner.lda(7, n_iter=100, alpha=0.847433736937, beta=0.763774618977)
  conferences = graph.get_papers_by_venue()
  f, subplts = plt.subplots(3, 3)
  f.tight_layout()
  y_counter = -1
  x_counter = 0
  for conf_index, conference in enumerate(mysql.get_conferences()):
    # if conference.acronym not in legit_conferences: continue
    if conference.acronym in non_legit_conferences: continue
    y_counter += 1
    if y_counter > 2:
      x_counter += 1
      y_counter = 0
    year_topics = {}
    year_heatmaps = {}
    for year, papers in yearize(conferences[conference.id]).items():
      topics = np.array([0]*lda_model.n_topics)
      for paper_id in papers:
        topics = np.add(topics, miner.documents[paper_id].topics_count)
      year_heatmaps[year] = topics
      year_topics[year] = percent_sort(topics)
    width = 0.8
    x_axis = np.arange(1, len(year_topics.keys()) + 1)
    # x_axis = [c.acronym for c in mysql.get_conferences()]
    y_offset = np.array([0] * len(year_topics.keys()))
    colors_dict={}
    for index in range(TOP_TOPIC_COUNT):
      bar_val, color = [], []
      for year in sorted(year_topics.keys(), key=lambda x:int(x)):
        topic = year_topics[year][index]
        colors_dict[topic[0]] = get_color(topic[0])
        color.append(colors_dict[topic[0]])
        bar_val.append(topic[1])
      subplts[x_counter, y_counter].bar(x_axis, bar_val, width, color=color, bottom=y_offset)
      y_offset = np.add(y_offset, bar_val)
    # subplts[x_counter, y_counter].set_ylabel("Topic Coverage %")
    #subplts[x_counter, y_counter].set_xlabel("Conferences")
    if len(year_topics.keys()) <= 14:
      subplts[x_counter, y_counter].set_xticks(x_axis + width / 2)
      subplts[x_counter, y_counter].set_xticklabels([str(y)[2:] for y in sorted(year_topics.keys(), key=lambda x: int(x))], fontsize=7)
    else:
      subplts[x_counter, y_counter].set_xticks(np.arange(1, len(year_topics.keys()) + 1, 2) + width / 2)
      subplts[x_counter, y_counter].set_xticklabels([str(y)[2:] for index, y in enumerate(sorted(year_topics.keys(), key=lambda x: int(x))) if index%2 == 0], fontsize=7)

    subplts[x_counter, y_counter].set_yticks(np.arange(0, 101, 20))
    subplts[x_counter, y_counter].set_ylim([0,101])
    subplts[x_counter, y_counter].set_title(conference.acronym)
  # Legends
  patches = []
  labels = []
  for topic in xrange(lda_model.n_topics):
    patches.append(mpatches.Patch(color=get_color(topic)))
    labels.append('Topic %s' % str(topic))
  f.legend(handles=patches, labels=labels, loc='upper center', bbox_to_anchor=(0.5, 0.04), ncol=12, fontsize=7)
  plt.savefig("figs/evolution/evolution_7topics.png")
  plt.clf()
  n_top_words = 10
  for index, topic_dist in enumerate(lda_model.topic_word_):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
    print('Topic {}: {}'.format(index, ', '.join(topic_words)))


def pc_bias_table():
  def index_by_year(tups):
    y_comm = {}
    for tup in tups:
      comm = y_comm.get(tup[1], set())
      comm.add(tup[0])
      y_comm[tup[1]] = comm
    return y_comm

  graph = cite_graph(GRAPH_CSV)
  max_len = 21
  start = 1993
  max_len = 5
  start = 2009
  p_conferences = graph.get_papers_by_venue()
  p_committees = graph.get_committee_by_conference()
  conf_year_scores = {}
  for conference in mysql.get_conferences():
    year_committees = index_by_year(p_committees[conference.id])
    year_papers = index_by_year(p_conferences[conference.id])
    year_scores = {}
    for year in sorted(year_committees.keys(), key=lambda y: int(y)):
      if year < start: continue
      papers = year_papers.get(year, None)
      if papers is None:
        year_scores[int(year)] = None
        continue
      committee = year_committees[year]
      comm_papers = 0
      non_comm_papers = 0
      for paper_id in papers:
        paper = graph.paper_nodes[paper_id]
        author_ids = set(paper.author_ids.strip().split(","))
        if author_ids.intersection(committee):
          comm_papers += 1
        else:
          non_comm_papers += 1
      year_scores[int(year)] = 0 if not comm_papers else int(round(comm_papers * 100 / (comm_papers + non_comm_papers)))
    conf_year_scores[conference.acronym] = year_scores
  header = ["conf"] + [str(start + i) for i in xrange(max_len)]
  table = PrettyTable(header)
  for conf, year_scores in conf_year_scores.items():
    row = [conf]
    for index in xrange(max_len):
      row.append(year_scores.get(start+index, None))
    table.add_row(row)
  print("```")
  print(table)
  print("```")
  exit()

def pc_paper_count_table():
  def index_by_year(tups):
    y_comm = {}
    for tup in tups:
      comm = y_comm.get(tup[1], set())
      comm.add(tup[0])
      y_comm[tup[1]] = comm
    return y_comm

  graph = cite_graph(GRAPH_CSV)
  max_len = 5
  start = 2009
  p_conferences = graph.get_papers_by_venue()
  p_committees = graph.get_committee_by_conference()
  header = ["conf", "# Accepted", "# from PC"]
  table = PrettyTable(header)
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
    table.add_row([conference.acronym, tot_papers, comm_papers])
  print("```")
  print(table)
  print("```")
  exit()



def pc_bias():
  def index_by_year(tups):
    y_comm = {}
    for tup in tups:
      comm = y_comm.get(tup[1], set())
      comm.add(tup[0])
      y_comm[tup[1]] = comm
    return y_comm

  legit_conferences = ["ICSE", "MSR", "FSE", "ASE"]
  colors = ['r', 'g', 'b', 'y']
  graph = cite_graph(GRAPH_CSV)
  width = 0.5
  space = 0.3
  p_conferences = graph.get_papers_by_venue()
  p_committees = graph.get_committee_by_conference()
  max_len = 21
  low = 1
  high = max_len * (len(legit_conferences)*width + space) + 1
  delta = (high - low)/max_len
  x_axis = np.arange(low, high, delta)
  x_ticks = np.arange(1993, 1993+max_len)
  conf_index = 0
  patches = []
  for conference in mysql.get_conferences():
    if conference.acronym not in legit_conferences: continue
    year_committees = index_by_year(p_committees[conference.id])
    year_papers = index_by_year(p_conferences[conference.id])
    year_scores = {}
    y_axis = []
    #x_axis = np.arange(1, len(year_committees.keys())+1)
    for year in sorted(year_committees.keys(), key=lambda y: int(y)):
      papers = year_papers.get(year,None)
      if papers is None:
        y_axis.append(0)
        continue
      committee = year_committees[year]
      comm_papers = 0
      non_comm_papers = 0
      for paper_id in papers:
        paper = graph.paper_nodes[paper_id]
        author_ids = set(paper.author_ids.strip().split(","))
        if author_ids.intersection(committee):
          comm_papers += 1
        else:
          non_comm_papers += 1
      year_scores[year] = (comm_papers, non_comm_papers)
      percent = 0 if not comm_papers else comm_papers*100/(comm_papers+non_comm_papers)
      y_axis.append(percent)
    y_axis = np.lib.pad(y_axis, (max_len-len(y_axis), 0), 'constant', constant_values=0)
    plt.bar(x_axis+conf_index*width, y_axis, width=width, color=colors[conf_index])
    patches.append(mpatches.Patch(color=colors[conf_index], label=conference.acronym))
    conf_index += 1
  plt.xlabel("Year")
  plt.ylabel("% of papers by PC")
  plt.xticks(x_axis + len(legit_conferences)*width/2, [str(y)[2:] for y in x_ticks])
  #plt.yticks(np.arange(0, 100, 10))
  #plt.title(conference.acronym)
  plt.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(legit_conferences), fontsize=7)
  plt.savefig("figs/pc/pc.png")
  plt.clf()

def topic_evolution():
  graph = cite_graph(GRAPH_CSV)
  miner = Miner(graph)
  lda_model, vocab = miner.lda(7, n_iter=100, alpha=0.847433736937, beta=0.763774618977)
  paper_nodes = graph.paper_nodes
  topics_map = {}
  n_topics = lda_model.n_topics
  for paper_id, paper in paper_nodes.items():
    document = miner.documents[paper_id]
    year_topics = topics_map.get(paper.year, np.array([0]*n_topics))
    topics_map[paper.year] = np.add(year_topics, document.topics_count)
  yt_map = {}
  for year, t_count in topics_map.items():
    yt_map[year] = percent_sort(t_count)
  width = 0.8
  plts = []
  x_axis = np.arange(1, len(yt_map.keys()) + 1)
  # x_axis = [c.acronym for c in mysql.get_conferences()]
  y_offset = np.array([0] * len(yt_map.keys()))
  colors_dict = {}
  TOP_TOPIC_COUNT = 7
  for index in range(TOP_TOPIC_COUNT):
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
  plt.xticks(x_axis + width/2, [str(y)[2:] for y in sorted(yt_map.keys(), key=lambda x: int(x))])
  plt.yticks(np.arange(0, 101, 10))
  plt.ylim([0, 101])
  # Legends
  patches = []
  squares = []
  names = []
  t_names = ["Testing", "Applications", "Program Analysis", "Tools and Projects",
             "Defect Analysis", "Modeling", "Maintenance"]
  for index, (topic, color) in enumerate(colors_dict.items()):
    print(topic)
    patches.append(mpatches.Patch(color=color, label='Topic %s' % str(topic)))
    squares.append(plts[index][0])
    # names.append('Topic %s' % str(topic))
    # names.append(t_names[index])
  # plt.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=7, fontsize=9)
  plt.legend(tuple(patches), tuple(t_names), loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=4, fontsize=11, handlelength=0.7)
  plt.savefig("figs/topic_evolution/topic_evolution_7_gib.png")
  plt.clf()
  n_top_words = 10
  for index, topic_dist in enumerate(lda_model.topic_word_):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
    print('Topic {}: {}'.format(index, ', '.join(topic_words)))


def pc_topics_heatmap(year_range=None):
  def index_by_year(tups):
    y_comm = {}
    for tup in tups:
      comm = y_comm.get(tup[1], set())
      comm.add(tup[0])
      y_comm[tup[1]] = comm
    return y_comm

  graph = cite_graph(GRAPH_CSV)
  miner = Miner(graph)
  lda_model, vocab = miner.lda(7, n_iter=100, alpha=0.847433736937, beta=0.763774618977)
  max_len = 21
  start = 1993
  p_conferences = graph.get_papers_by_venue()
  p_committees = graph.get_committee_by_conference()
  conference_topics = {}
  for conference in mysql.get_conferences():
    year_committees = index_by_year(p_committees[conference.id])
    year_papers = index_by_year(p_conferences[conference.id])
    year_scores = {}
    topics = np.array([0] * lda_model.n_topics)
    for year in sorted(year_committees.keys(), key=lambda y: int(y)):
      if (year_range is not None) and (int(year) not in year_range):
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
  for conference_id in sorted(conference_topics.keys(), key=lambda x: int(x)):
    tot = sum(conference_topics[conference_id])
    dist = [top / tot for top in conference_topics[conference_id]]
    heatmap_arr.append(dist)
  row_labels = range(lda_model.n_topics)
  column_labels = [c.acronym for c in mysql.get_conferences()]
  make_dendo_heatmap(np.transpose(heatmap_arr), row_labels, column_labels, "figs/pc/pc_heatmap_7topics.png")

def super_authors_top():
  y_1 = [0]*2+[2, 4, 6, 12, 77]
  y_10 = [3, 27, 62, 131, 140, 252, 403]
  y_20 = [13, 83, 195, 346, 356, 507, 537]
  y_all = [139, 621, 1454, 1886, 1605, 1210, 773]
  ind = np.arange(1,8)  # the x locations for the groups
  fig = plt.figure(figsize=(8, 2))
  width = 0.2  # the width of the bars
  fig, ax = plt.subplots()
  rects_1 = ax.bar(ind, y_1, width, color='r')
  rects_10 = ax.bar(ind+width, y_10, width, color='b')
  rects_20 = ax.bar(ind+2*width, y_20, width, color='g')
  rects_all = ax.bar(ind+3*width, y_all, width, color='y')
  ax.legend((rects_1[0], rects_10[0], rects_20[0], rects_all[0]), ('1%', '10%', '20%', '100%'), loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=4, fontsize=12)
  plt.xticks(np.arange(1,8)+2*width, ind)
  plt.xlabel("Cumulative # of Topics")
  plt.ylabel("Authors Count")
  plt.yscale('log')
  plt.savefig("figs/super_author/all_authors.png")
  plt.clf()


def get_top_papers():
  n_topics = 7
  top_papers = {}
  for index in range(n_topics):
    top_papers[index] = []
  graph = cite_graph(GRAPH_CSV)
  miner = Miner(graph)
  lda_model, vocab = miner.lda(7, n_iter=100, alpha=0.847433736937, beta=0.763774618977)
  for paper_id, paper in graph.paper_nodes.items():
    topics = miner.documents[paper_id].topics_count
    # if int(paper.year) < 2009: continue
    if max(topics) == 0:
      continue
    topic = topics.argmax()
    # cites = len(paper.cites.split(",")) if paper.cites else 0
    cites = paper.cited_counts
    top_papers[topic].append([(cites, paper.title, paper.authors, paper.year)])
  for index in range(n_topics):
    top_papers[index] = sorted(top_papers[index], reverse=True)[:4]
    print("***", index, "***")
    for paper in top_papers[index]:
      paper = paper[0]
      print(paper[0], paper[-1] + " - " + paper[1] + ", " + paper[2])



def print_top_authors(top_percent= None, min_year=None):
  graph = cite_graph(GRAPH_CSV)
  tops = top_authors(graph, top_percent=top_percent, min_year=min_year)
  author_papers = graph.get_papers_by_authors()
  top_tups = []
  for author_id, author in graph.author_nodes.items():
    if  author_id in tops:
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
  with open("temp_all.txt", "wb") as f:
    for top_tup in top_tups:
      f.write(str(top_tup))
      f.write("\n")


def paper_bar():
  graph = cite_graph(GRAPH_CSV)
  conferences = graph.get_papers_by_venue()
  start = 2001
  end = 2012
  year_count = {}
  for year in range(start, end+1):
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
  fig = plt.figure(figsize=(8,3))
  plt.bar(bar_x, bar_y, color = 'blue', align='center')
  plt.xlim([start-1, end+1])
  plt.xticks(bar_x, rotation = 45)
  plt.ylim(300, 800)
  plt.xlabel('Year')
  plt.ylabel('# of Papers')
  plt.savefig("figs/paper_count.png", bbox_inches='tight')
  plt.clf()


def lda_topics():
  graph = cite_graph(GRAPH_CSV)
  miner = Miner(graph)
  lda_model, vocab = miner.lda(11, n_iter=100, alpha=0.22359, beta=0.53915)
  # lda_model, vocab = miner.lda(11, n_iter=100, alpha=0.847433736937, beta=0.763774618977)
  n_top_words = 15
  for index, topic_dist in enumerate(lda_model.topic_word_):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
    print('Topic {}: {}'.format(index, ', '.join(topic_words)))


# pc_topics_heatmap([2009, 2010, 2011, 2012, 2013])
# print_top_authors()
# get_top_papers()
# paper_bar()


# conference_evolution_2(range(1993,2001), "heatmap_93_00")
# conference_evolution_2(range(2001,2009), "heatmap_01_08")
# conference_evolution_2(range(2009,2014), "heatmap_09_14")

# pc_topics_heatmap()
# pc_bias_table()
# pc_paper_count_table()
#
# super_author("super_author_7_2", 0.20)
# super_author("super_author_7_1", 0.10)
# super_author("super_author_7_top")
# super_authors_top()
# conference_evolution()
# conference_diversity()
# pc_bias()
# topic_evolution()
# y_counter = -1
# x_counter = 0
# for i in range(12):
#   y_counter += 1
#   if y_counter > 2:
#     y_counter = 0
#     x_counter += 1
#   print(i, x_counter, y_counter)
lda_topics()
