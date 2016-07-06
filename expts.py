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

__author__ = "panzer"

COLORS = sorted(clrs.cnames.keys())
TOPIC_THRESHOLD = 3

def get_color(index): return str(COLORS[index])

def super_author():
  graph = cite_graph()
  miner = Miner(graph)
  lda_model, vocab = miner.lda(12, n_iter=100, alpha=0.847433736937, beta=0.763774618977)
  authors = graph.get_papers_by_authors()
  author_topics = {}
  for author_id, papers in authors.items():
    topics = [0]*lda_model.n_topics
    for paper_id, _, __ in papers:
      document = miner.documents[paper_id]
      for index, topic_count in enumerate(document.topics_count):
        if topic_count >= TOPIC_THRESHOLD:
          topics[index] = 1
    author_topics[author_id] = sum(topics)
  vals = sorted(author_topics.values(), reverse=True)
  x_axis = range(1, len(vals) + 1)
  plt.ylabel("Topic Count")
  plt.xlabel("Author ID")
  plt.title("Super Author")
  plt.plot(x_axis, vals)
  plt.savefig("figs/super_author.png")
  plt.clf()
  counter = Counter()
  for val in vals:
    counter[val] += 1
  bar_x = []
  bar_y = []
  for key in sorted(counter.keys()):
    bar_x.append(key)
    bar_y.append(counter[key])
  fig, ax = plt.subplots()
  width = 2/3
  ax.bar(bar_x, bar_y, 2/3, color='blue', align='center')
  ax.set_xticks(np.arange(1,len(bar_x)+1))
  ax.set_xticklabels(bar_x)
  for i, v in zip(bar_x,bar_y):
    ax.text(i, v + 3, str(v), color='red', fontweight='bold', fontsize=7, horizontalalignment='center')
  plt.xlabel("Topics")
  plt.ylabel("Authors Count")
  plt.savefig("figs/super_author_bar.png")
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
  graph = cite_graph()
  miner = Miner(graph)
  lda_model, vocab = miner.lda(12, n_iter=100, alpha=0.847433736937, beta=0.763774618977)
  conferences = graph.get_papers_by_conference()
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
  for index in range(5):
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
  plt.yticks(np.arange(0,100,10))
  #Legends
  patches = []
  for topic, color in colors_dict.items():
    patches.append(mpatches.Patch(color=color, label='Topic %s' % str(topic)))
  plt.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=6, fontsize=7)
  plt.savefig("figs/diversity/conference_diversity_12topics.png")
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
  row_labels = range(lda_model.n_topics)
  column_labels = [c.acronym for c in mysql.get_conferences()]
  ax.set_xticks(np.arange(heatmap_arr.shape[1])+0.5, minor=False)
  ax.set_yticks(np.arange(heatmap_arr.shape[0])+0.5, minor=False)
  ax.set_xticklabels(row_labels, minor=False)
  ax.set_yticklabels(column_labels, minor=False)
  plt.savefig("figs/diversity/heatmap_12topics.png")
  plt.clf()
  for index, topic_dist in enumerate(lda_model.topic_word_):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
    print('Topic {}: {}'.format(index, ', '.join(topic_words)))
  make_heatmap(np.transpose(heatmap_arr), row_labels, column_labels, "figs/diversity/dend_heatmap_12topics.png")


def make_heatmap(arr, row_labels, column_labels, figname):
    df = pd.DataFrame(arr, columns=column_labels, index=row_labels)
    # Compute pairwise distances for columns
    col_clusters = linkage(pdist(df.T, metric='euclidean'), method='complete')
    # plot column dendrogram
    fig = plt.figure(figsize=(8, 8))
    axd2 = fig.add_axes([0.25, 0.75, 0.50, 0.10])
    col_dendr = dendrogram(col_clusters, orientation='top',
                           color_threshold=np.inf)  # makes dendrogram black)
    axd2.set_xticks([])
    axd2.set_yticks([])
    # plot row dendrogram
    axd1 = fig.add_axes([0.0, 0.123, 0.21, 0.555])
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
    axm = fig.add_axes([0.25, 0.1, 0.63, 0.6])
    cax = axm.matshow(df_rowclust, interpolation='nearest', cmap='hot_r')
    fig.colorbar(cax)
    print(df_rowclust.columns)
    axm.set_xticks(np.arange(len(list(df_rowclust.columns))))
    axm.set_xticklabels(list(df_rowclust.columns), rotation="vertical")
    axm.set_yticks(np.arange(len(list(df_rowclust.index))))
    axm.set_yticklabels(list(df_rowclust.index))
    plt.savefig(figname)
    plt.clf()


def conference_evolution():
  legit_conferences = ["ICSE", "MSR", "FSE", "ASE"]
  TOP_TOPIC_COUNT = 5
  graph = cite_graph()
  miner = Miner(graph)
  lda_model, vocab = miner.lda(12, n_iter=100, alpha=0.847433736937, beta=0.763774618977)
  conferences = graph.get_papers_by_conference()
  for conference in mysql.get_conferences():
    if conference.acronym not in legit_conferences: continue
    year_topics = {}
    year_heatmaps = {}
    for year, papers in yearize(conferences[conference.id]).items():
      topics = np.array([0]*lda_model.n_topics)
      for paper_id in papers:
        topics = np.add(topics, miner.documents[paper_id].topics_count)
      year_heatmaps[year] = topics
      year_topics[year] = percent_sort(topics)
    width = 0.8
    plts = []
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
      plts.append(plt.bar(x_axis, bar_val, width, color=color, bottom=y_offset))
      y_offset = np.add(y_offset, bar_val)
    plt.ylabel("Topic Coverage %")
    plt.xlabel("Conferences")
    plt.xticks(x_axis + width / 2, [str(y)[2:] for y in sorted(year_topics.keys(), key=lambda x: int(x))])
    plt.yticks(np.arange(0, 100, 10))
    plt.title(conference.acronym)
    #Legends
    patches = []
    for topic, color in colors_dict.items():
      patches.append(mpatches.Patch(color=color, label='Topic %s' % str(topic)))
    plt.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=6, fontsize=7)
    plt.savefig("figs/evolution/%s.png"%conference.acronym)
    plt.clf()
    #Heatmap
    heatmap_arr = []
    for year in sorted(year_heatmaps.keys(), key=lambda x: int(x)):
      tot = sum(year_heatmaps[year])
      dist = [top / tot for top in year_heatmaps[year]]
      heatmap_arr.append(dist)
    fig, ax = plt.subplots()
    heatmap_arr = np.array(heatmap_arr)
    heatmap_arr[np.isnan(heatmap_arr)] = 0
    heatmap = ax.pcolor(heatmap_arr, cmap=plt.cm.Reds)
    plt.ylabel("Year")
    plt.xlabel("Topics")
    row_labels = range(lda_model.n_topics)
    column_labels = [str(y)[2:] for y in sorted(year_heatmaps.keys(), key=lambda x: int(x))]
    ax.set_xticks(np.arange(heatmap_arr.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(heatmap_arr.shape[0]) + 0.5, minor=False)
    ax.set_xticklabels(row_labels, minor=False)
    ax.set_yticklabels(column_labels, minor=False)
    plt.savefig("figs/evolution/%s_heatmap.png"%conference.acronym)
    plt.clf()

  n_top_words = 10
  for index, topic_dist in enumerate(lda_model.topic_word_):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
    print('Topic {}: {}'.format(index, ', '.join(topic_words)))


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
  graph = cite_graph()
  width = 0.5
  space = 0.3
  p_conferences = graph.get_papers_by_conference()
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
  graph = cite_graph()
  miner = Miner(graph)
  lda_model, vocab = miner.lda(12, n_iter=100, alpha=0.847433736937, beta=0.763774618977)
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
  TOP_TOPIC_COUNT = 5
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
  plt.yticks(np.arange(0, 100, 10))
  # Legends
  patches = []
  for topic, color in colors_dict.items():
    patches.append(mpatches.Patch(color=color, label='Topic %s' % str(topic)))
  plt.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=6, fontsize=7)
  plt.savefig("figs/topic_evolution/topic_evolution.png")
  plt.clf()
  n_top_words = 10
  for index, topic_dist in enumerate(lda_model.topic_word_):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
    print('Topic {}: {}'.format(index, ', '.join(topic_words)))


#super_author()
#conference_evolution()
conference_diversity()
#pc_bias()
#topic_evolution()