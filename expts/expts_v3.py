from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

from utils.lib import O, Memoized
import numpy as np
from collections import OrderedDict, Counter
from network.mine import cite_graph, Miner
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from db import mysql
import pandas as pd
from sklearn.feature_extraction import text
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram
from expts.settings import dend as dend_settings
from matplotlib.colors import ColorConverter
import cPickle as pkl
from sklearn.externals import joblib

GRAPH_CSV = "data/citemap_v8.csv"

# For 11 TOPICS
ALPHA = 0.22359
BETA = 0.53915
ITERATIONS = 100
# TOPICS = ["TPC %d" % d for d in range(N_TOPICS)]
# TOPICS = ["Design", "Testing", "Modelling", "Mobile", "Energy", "Defects",
#           "SourceCode", "WebApps", "Configuration", "Developer", "Mining"]
TOPICS_JOURNALS = ["Requirements", "Applications", "Source Code", "Empirical", "Testing", "Security", "Modelling"]
TOPICS_ALL = ["Performance", "Management", "Metrics", "Requirements", "Empirical",
              "Security", "Applications", "Modelling", "Source Code", "Program Analysis",
              "Testing"]
TOPIC_THRESHOLD = 3

COLORS_JOURNAL = ["grey", "red", "blue", "green",
            "yellow", "magenta", "cyan", "black"]

COLORS_ALL = ["lightgray", "red", "blue", "darkslategray",
            "yellow", "darkmagenta", "cyan", "saddlebrown",
            "orange", "lime", "hotpink"]

COLOR_CONVERTER = ColorConverter()


def get_color(index):
  if THE.permitted == "journals":
    return COLORS_JOURNAL[index]
  if THE.permitted == "all":
    return COLORS_ALL[index]


def get_topics():
  if THE.permitted == "journals":
    return TOPICS_JOURNALS
  if THE.permitted == "all":
    return TOPICS_ALL


def get_n_topics():
  if THE.permitted == "journals":
    return 7
  if THE.permitted == "all":
    return 11


STOP_WORDS = text.ENGLISH_STOP_WORDS.union(['software', 'engineering', 'paper', 'study', 'based',
                                            'results', 'approach', 'case', 'workshop', 'international', 'research',
                                            'conference', 'introduction', 'editors', 'article', 'issue', 'month',
                                            'copyright', 'special', 'used', 'using', 'use', 'studies', 'review',
                                            'editorial', 'report', 'book', 'ieee', 'published', 'science', 'column',
                                            'author', 'proposed', 'icse', 'article', 'year', 'articles', 'page', '2000',
                                            '2004', 'papers', 'computer', 'held', 'editor'])

CONFERENCES = [venue.acronym for venue in mysql.get_conferences()]

# Config
THE = O()
THE.permitted = "all"


@Memoized
def get_graph_lda_data():
  graph = cite_graph(GRAPH_CSV)
  miner = Miner(graph, THE.permitted)
  lda_model, vocab = miner.lda(get_n_topics(), n_iter=ITERATIONS, alpha=ALPHA, beta=BETA, stop_words=STOP_WORDS)
  return miner, graph, lda_model, vocab


def store_graph_lda_data():
  miner, graph, lda_model, vocab = get_graph_lda_data()
  with open('cache/%s/graph.pkl' % THE.permitted, 'wb') as f:
    pkl.dump(graph, f, pkl.HIGHEST_PROTOCOL)
  with open('cache/%s/vectorizer.pkl' % THE.permitted, 'wb') as f:
    pkl.dump(miner.vectorizer, f, pkl.HIGHEST_PROTOCOL)
  with open('cache/%s/doc_2_vec.pkl' % THE.permitted, 'wb') as f:
    joblib.dump(miner.doc_2_vec, f)
  with open('cache/%s/documents.pkl' % THE.permitted, 'wb') as f:
    pkl.dump(miner.documents, f, pkl.HIGHEST_PROTOCOL)
  with open('cache/%s/lda_model.pkl' % THE.permitted, 'wb') as f:
    pkl.dump(lda_model, f, pkl.HIGHEST_PROTOCOL)
  with open('cache/%s/vocabulary.pkl' % THE.permitted, 'wb') as f:
    pkl.dump(vocab, f, pkl.HIGHEST_PROTOCOL)
  return miner, graph, lda_model, vocab


@Memoized
def retrieve_graph_lda_data():
  graph_file = 'cache/%s/graph.pkl' % THE.permitted
  vectorizer_file = 'cache/%s/vectorizer.pkl' % THE.permitted
  doc_2_vec_file = 'cache/%s/doc_2_vec.pkl' % THE.permitted
  documents_file = 'cache/%s/documents.pkl' % THE.permitted
  lda_model_file = 'cache/%s/lda_model.pkl' % THE.permitted
  vocabulary_file = 'cache/%s/vocabulary.pkl' % THE.permitted
  if os.path.isfile(graph_file) and os.path.isfile(vectorizer_file) \
          and os.path.isfile(doc_2_vec_file) and os.path.isfile(documents_file) \
          and os.path.isfile(lda_model_file) and os.path.isfile(vocabulary_file):
    with open(graph_file) as f:
      graph = pkl.load(f)
    miner = Miner(graph)
    with open(vectorizer_file) as f:
      miner.vectorizer = pkl.load(f)
    with open(doc_2_vec_file) as f:
      miner.doc_2_vec = joblib.load(f)
    with open(documents_file) as f:
      miner.documents = pkl.load(f)
    with open(lda_model_file) as f:
      lda_model = pkl.load(f)
    with open(vocabulary_file) as f:
      vocab = pkl.load(f)
  else:
    miner, graph, lda_model, vocab = store_graph_lda_data()
  return miner, graph, lda_model, vocab


def shorter_names(name):
  name_map = {
    "SOFTWARE" : "S/W",
    "SIGSOFT": "NOTES"
  }
  if name in name_map:
    return name_map[name]
  return name


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


def report(lda_model, vocab, fig_name="topic_dist", n_top_words=10, plot_terms=50):
  fig = plt.figure()
  x_axis = range(1, plot_terms + 1)
  legends = []
  for index, topic_dist in enumerate(lda_model.topic_word_):
    sorted_dist = np.sort(topic_dist)
    # scores = sorted_dist[:-(n_top_words + 1):-1]
    plot_scores = sorted_dist[:-(plot_terms + 1):-1]
    plot_scores = np.log(plot_scores)
    plt.plot(x_axis, plot_scores)
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
    legends.append("Topic %d" % index)
    print('Topic {}: {}'.format(index, ', '.join(topic_words)))
  plt.legend(legends, loc='upper right')
  plt.title(fig_name)
  plt.xlabel("Term Index")
  plt.ylabel("Log. Word Score")
  plt.savefig("figs/v3/%s/%s.png" % (THE.permitted, fig_name))
  fig.clf()


def make_heatmap(arr, row_labels, column_labels, figname, paper_range):
  plt.figure(figsize=(4, 3))
  df = pd.DataFrame(arr, columns=column_labels, index=row_labels)
  cax = plt.matshow(df, interpolation='nearest', cmap='hot_r')
  plt.colorbar(cax)
  plt.xticks(np.arange(len(list(df.columns))), list(df.columns), rotation="vertical")
  plt.yticks(np.arange(len(list(df.index))), list(df.index))
  [tick.set_color("red") if tick.get_text() in CONFERENCES else tick.set_color("green") for tick in plt.gca().get_xticklabels()]
  if paper_range:
    plt.title("Topics to Conference Distribution(%d - %d)" % (paper_range[0], paper_range[-1]), y=1.2)
  else:
    plt.title("Topics to Conference Distribution", y=1.2)
  plt.savefig(figname, bbox_inches='tight')
  plt.clf()


def make_dendo_heatmap(arr, row_labels, column_labels, figname, paper_range):
  settings = dend_settings.get("dend_%d_%d" % (len(row_labels), len(column_labels)), None)
  if settings is None:
    print("ERROR: Configure Dendogram settings for %d rows and %d columns" % (len(row_labels), len(column_labels)))
    return
  df = pd.DataFrame(arr, columns=column_labels, index=row_labels)
  # Compute pairwise distances for columns
  col_clusters = linkage(pdist(df.T, metric='euclidean'), method='complete')
  # plot column dendrogram
  fig = plt.figure(figsize=settings.fig_size)
  axd2 = fig.add_axes(settings.col_axes)
  col_dendr = dendrogram(col_clusters, orientation='top',
                         color_threshold=np.inf, link_color_func=lambda _: '#999999')  # makes dendrogram black
  axd2.set_xticks([])
  axd2.set_yticks([])
  # plot row dendrogram
  axd1 = fig.add_axes(settings.row_axes)
  row_clusters = linkage(pdist(df, metric='euclidean'), method='complete')
  row_dendr = dendrogram(row_clusters, orientation='left',
                         count_sort='ascending',
                         color_threshold=np.inf, link_color_func=lambda _: '#999999')  # makes dendrogram black
  axd1.set_xticks([])
  axd1.set_yticks([])
  # remove axes spines from dendrogram
  for i, j in zip(axd1.spines.values(), axd2.spines.values()):
    i.set_visible(False)
    j.set_visible(False)
  # reorder columns and rows with respect to the clustering
  df_rowclust = df.ix[row_dendr['leaves'][::-1]]
  # df_rowclust.columns = [df_rowclust.columns[col_dendr['leaves']]]
  df_rowclust = df_rowclust[col_dendr['leaves']]
  # plot heatmap
  axm = fig.add_axes(settings.plot_axes)
  cax = axm.matshow(df_rowclust, interpolation='nearest', cmap='hot_r')
  if THE.permitted == "all":
    fig.colorbar(cax, location='bottom')
  else:
    fig.colorbar(cax)
  axm.set_xticks(np.arange(len(list(df_rowclust.columns))))
  axm.set_xticklabels(list(df_rowclust.columns), rotation="vertical")
  [tick.set_color("red") if tick.get_text() in CONFERENCES else tick.set_color("green") for tick in axm.get_xticklabels()]
  axm.set_yticks(np.arange(len(list(df_rowclust.index))))
  axm.set_yticklabels(list(df_rowclust.index))
  if paper_range:
    plt.title("Clustered topics to Conference Distribution(%d - %d)" % (paper_range[0], paper_range[-1]), y=-0.2)
  else:
    plt.title("Clustered topics to Conference Distribution", y=-0.2)
  plt.savefig(figname, bbox_inches='tight')
  plt.clf()


def paper_bar():
  print("PAPER BAR for %s" % THE.permitted)
  graph = cite_graph(GRAPH_CSV)
  venues = graph.get_papers_by_venue(permitted=THE.permitted)
  start = 2001
  end = 2016
  year_count = {}
  for year in range(start, end + 1):
    year_count[year] = 0
  for conference_id, papers in venues.items():
    for tup in papers:
      count = year_count.get(int(tup[1]), None)
      if count is None: continue
      year_count[int(tup[1])] += 1
  bar_x, bar_y = [], []
  for year, count in year_count.items():
    bar_x.append(year)
    bar_y.append(count)
  plt.figure(figsize=(8, 3))
  plt.bar(bar_x, bar_y, color='blue', align='center')
  plt.xlim([start - 1, end + 1])
  plt.xticks(bar_x, rotation=45)
  plt.xlabel('Year')
  plt.ylabel('# of Papers')
  plt.savefig("figs/v3/%s/paper_count.png" % THE.permitted, bbox_inches='tight')
  plt.clf()


def diversity(fig_name, paper_range):
  if paper_range:
    print("DIVERSITY for %s between %d - %d" % (THE.permitted, paper_range[0], paper_range[-1]))
  else:
    print("DIVERSITY for %s" % THE.permitted)
  miner, graph, lda_model, vocab = retrieve_graph_lda_data()
  paper_map = graph.get_papers_by_venue(THE.permitted)
  venue_topics = {}
  venue_heatmaps = {}
  valid_conferences = []
  venues = mysql.get_venues()
  for conference_id, papers in paper_map.items():
    topics = np.array([0] * lda_model.n_topics)
    for tup in yearize(papers).items():
      if paper_range and tup[0] not in paper_range: continue
      for paper_id in tup[1]:
        topics = np.add(topics, miner.documents[paper_id].topics_count)
    if sum(topics) > 0:
      venue_topics[conference_id] = percent_sort(topics)
      venue_heatmaps[conference_id] = topics
      valid_conferences.append(conference_id)
  # row_labels = [str(ind) + "-" + name for ind, name in zip(range(lda_model.n_topics), get_topics())]
  row_labels = [name for ind, name in zip(range(lda_model.n_topics), get_topics())]
  # row_labels = ["%2d" % ind for ind in range(lda_model.n_topics)]
  column_labels = [shorter_names(venue.acronym) for c, venue in venues.items() if venue.id in valid_conferences]
  # Heatmap
  heatmap_arr = []
  for conference_id in sorted(venue_heatmaps.keys(), key=lambda x: int(x)):
    tot = sum(venue_heatmaps[conference_id])
    dist = [top / tot for top in venue_heatmaps[conference_id]]
    heatmap_arr.append(dist)
  # report(lda_model, vocab, n_top_words=15)
  make_dendo_heatmap(np.transpose(heatmap_arr), row_labels, column_labels,
                     "figs/v3/%s/diversity/%s_dend.png" % (THE.permitted, fig_name), paper_range)
  make_heatmap(np.transpose(heatmap_arr), row_labels, column_labels,
               "figs/v3/%s/diversity/%s.png" % (THE.permitted, fig_name), paper_range)


def topic_evolution(venue=THE.permitted):
  print("TOPIC EVOLUTION for %s" % venue)
  miner, graph, lda_model, vocab = retrieve_graph_lda_data()
  paper_nodes = graph.get_paper_nodes(venue)
  topics_map = {}
  n_topics = lda_model.n_topics
  for paper_id, paper in paper_nodes.items():
    if int(paper.year) < 1993 or int(paper.year) > 2016: continue
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
  top_topic_count = 7
  for index in range(top_topic_count):
    bar_val, color = [], []
    for year in sorted(yt_map.keys(), key=lambda x: int(x)):
      topic = yt_map[year][index]
      if topic[0] not in colors_dict:
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
  topics = []
  for index, (topic, color) in enumerate(colors_dict.items()):
    patches.append(mpatches.Patch(color=color, label='Topic %s' % str(topic)))
    topics.append(get_topics()[topic])
  plt.legend(tuple(patches), tuple(topics), loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=6, fontsize=10,
             handlelength=0.7)
  plt.savefig("figs/v3/%s/topic_evolution/topic_evolution_%s.png" % (THE.permitted, venue))
  plt.clf()
  # report(lda_model, vocab)


def top_authors(graph, top_percent=0.01, min_year=None):
  authors = graph.get_papers_by_authors(THE.permitted)
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


def super_author(top_percents):
  print("SUPER AUTHOR for %s; TOP PERCENTS : %s" % (THE.permitted, top_percents))
  miner, graph, lda_model, vocab = retrieve_graph_lda_data()
  authors = graph.get_papers_by_authors(THE.permitted)
  author_publications = OrderedDict()
  for top_percent in top_percents:
    p_key = "%d" % (int(top_percent * 100)) + ' %'
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
    bar_y = [0] * (lda_model.n_topics + 1 - len(bar_y)) + bar_y
    author_publications[p_key] = bar_y
  ind = np.arange(1, lda_model.n_topics + 2)  # the x locations for the groups
  fig = plt.figure(figsize=(8, 2))
  width = 0.2  # the width of the bars
  fig, ax = plt.subplots()
  colors = ['orange', 'cyan', 'green', 'hotpink']
  rects = []
  keys = []
  for i, (key, bar_y) in enumerate(author_publications.items()):
    rect = ax.bar(ind + width * i, bar_y, width, color=colors[i])
    rects.append(rect[0])
    keys.append(key)
  ax.legend(tuple(rects), tuple(keys), loc='upper center',
            bbox_to_anchor=(0.5, 1.12), ncol=4, fontsize=16)
  plt.xticks(ind + 2 * width, ind, fontsize=16)
  plt.xlabel("Cumulative # of Topics", fontsize=16)
  plt.ylabel("Authors Count", fontsize=16)
  plt.yticks(fontsize=16)
  plt.yscale('log')
  plt.savefig("figs/v3/%s/super_author.png" % THE.permitted)
  plt.clf()


def get_top_papers(top_count=5, year_from=None):
  if year_from:
    print("TOP %d PAPERS for %s from %d" % (top_count, THE.permitted, year_from))
  else:
    print("TOP %d ALL TIME PAPERS for %s" % (top_count, THE.permitted))
  top_papers = {}
  for index in range(get_n_topics()):
    top_papers[index] = []
  miner, graph, lda_model, vocab = retrieve_graph_lda_data()
  for paper_id, paper in graph.get_paper_nodes(THE.permitted).items():
    topics = miner.documents[paper_id].topics_count
    if year_from and int(paper.year) < year_from: continue
    if max(topics) == 0:
      continue
    topic = topics.argmax()
    cites = paper.cited_counts
    top_papers[topic].append([(cites, paper.title, paper.authors, int(paper.year))])
  suffix = str(year_from) if year_from else 'all'
  topic_names = get_topics()
  with open("figs/v3/%s/top_papers_%s.csv" % (THE.permitted, suffix), "wb") as f:
    f.write("Index, Cites, Year, Title, Authors\n")
    for index in range(get_n_topics()):
      top_papers[index] = sorted(top_papers[index], reverse=True)[:top_count]
      for paper in top_papers[index]:
        paper = paper[0]
        f.write("%s, %d, %d, \"%s\", \"%s\"\n" % (topic_names[index], paper[0], paper[-1], paper[1], paper[2]))


def reporter():
  print("TOPIC REPORTS for %s" % THE.permitted)
  miner, graph, lda_model, vocab = retrieve_graph_lda_data()
  report(lda_model, vocab)


def _main():
  reporter()
  paper_bar()
  diversity("heatmap_09_16", range(2009, 2017))
  diversity("heatmap_01_08", range(2001, 2009))
  diversity("heatmap_93_00", range(1993, 2000))
  topic_evolution(venue="all")
  topic_evolution(venue="conferences")
  topic_evolution(venue="journals")
  super_author([0.01, 0.1, 0.2, 1.0])
  get_top_papers(year_from=2009)
  get_top_papers()


def _store():
  store_graph_lda_data()

if __name__ == "__main__":
  _main()
  # _store()
