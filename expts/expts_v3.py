from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

from utils.lib import O, Memoized, shuffle
import numpy as np
from collections import OrderedDict, Counter
from network.mine import cite_graph, Miner
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from db import mysqldb
import pandas as pd
from sklearn.feature_extraction import text
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram
from expts.settings import dend as dend_settings
from matplotlib.colors import ColorConverter
import cPickle as pkl
from sklearn.externals import joblib
from utils.sk import rdivDemo as sk

GRAPH_CSV = "data/citemap_v9.csv"

# For 11 TOPICS
ALPHA = 0.22359
BETA = 0.53915
ITERATIONS = 100
# TOPICS = ["TPC %d" % d for d in range(N_TOPICS)]
# TOPICS = ["Design", "Testing", "Modelling", "Mobile", "Energy", "Defects",
#           "SourceCode", "WebApps", "Configuration", "Developer", "Mining"]
TOPICS_JOURNALS = ["Requirements", "Applications", "Source Code", "Empirical", "Testing", "Security", "Modelling"]
# TOPICS_ALL = ["Performance", "Management", "Metrics", "Requirements", "Empirical",
#               "Security", "Applications", "Modelling", "Source Code", "Program Analysis",
#               "Testing"]
TOPICS_ALL = ["Testing", "Requirements", "Applications", "Modeling", "Management",
              "Source Code", "Design", "Program Analysis", "Metrics", "Performance", "Security"]

TOPIC_THRESHOLD = 3

COLORS_JOURNAL = ["grey", "red", "blue", "green",
            "yellow", "magenta", "cyan", "black"]

COLORS_ALL = ["lightgray", "red", "blue", "darkslategray",
            "yellow", "darkmagenta", "cyan", "saddlebrown",
            "orange", "lime", "hotpink"]

COLOR_CONVERTER = ColorConverter()


def is_true(val):
  return val in [True, 'True', 'true']


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

CONFERENCES = [venue.acronym for venue in mysqldb.get_conferences()]

# Config
THE = O()
THE.permitted = "all"


def is_not_none(s):
  return s and s != 'None'


def harmonic_dist(n):
  dist = [1 / i for i in range(1, n + 1)]
  total = sum(dist)
  return [d / total for d in dist]


def uniform_dist(n):
  return [1 / n] * n


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


def retrieve_graph():
  graph_file = 'cache/%s/graph.pkl' % THE.permitted
  if os.path.isfile(graph_file):
    with open(graph_file) as f:
      graph = pkl.load(f)
  else:
    graph = cite_graph(GRAPH_CSV)
    with open(graph_file, 'wb') as f:
      pkl.dump(graph, f, pkl.HIGHEST_PROTOCOL)
  return graph


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
    # legends.append("Topic %d" % index)
    legends.append(TOPICS_ALL[index])
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
    plt.title("Topics to Venue Distribution(%d - %d)" % (paper_range[0], paper_range[-1]), y=1.2)
  else:
    plt.title("Topics to Venue Distribution", y=1.2)
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
    plt.title("Clustered topics to Venue Distribution(%d - %d)" % (paper_range[0], paper_range[-1]), y=-0.2)
  else:
    plt.title("Clustered topics to Venue Distribution", y=-0.2)
  plt.savefig(figname, bbox_inches='tight')
  plt.clf()


def paper_bar(start=1992, end=2016):
  print("PAPER BAR for %s" % THE.permitted)
  graph = cite_graph(GRAPH_CSV)
  venues = graph.get_papers_by_venue(permitted=THE.permitted)
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


def paper_and_author_growth(min_year=1992, max_year=2015):
  graph = cite_graph(GRAPH_CSV)
  year_authors_map = OrderedDict()
  year_papers_map = OrderedDict()
  for _, paper in graph.get_paper_nodes(permitted=THE.permitted).items():
    year = int(paper.year)
    if not (min_year < year <= max_year): continue
    authors = paper.authors.split(",")
    year_authors_map[year] = year_authors_map.get(year, set([])).union(authors)
    year_papers_map[year] = year_papers_map.get(year, 0) + 1
  x_axis = []
  papers = []
  authors = []
  seen = set(year_authors_map[sorted(year_authors_map.keys())[0]])
  f = open("figs/v3/%s/paper_author_count.csv" % THE.permitted, "wb")
  f.write("Year, # Papers, # Authors\n")
  for key in sorted(year_authors_map.keys())[1:]:
    x_axis.append(key)
    papers.append(year_papers_map[key])
    new_authors = set(year_authors_map[key]).difference(seen)
    authors.append(len(new_authors))
    seen = seen.union(set(year_authors_map[key]))
    f.write("%d, %d, %d\n" % (key, year_papers_map[key], len(new_authors)))
  plt.plot(x_axis, papers)
  plt.plot(x_axis, authors)
  legends = ['Papers', 'Authors']
  plt.legend(legends, loc='upper left')
  plt.title('Growth of Papers and Authors')
  plt.xlabel("Year")
  plt.ylabel(" Count")
  plt.savefig("figs/v3/%s/paper_author_count.png" % THE.permitted)
  plt.clf()
  f.close()


def diversity(fig_name, paper_range=None):
  if paper_range:
    print("DIVERSITY for %s between %d - %d" % (THE.permitted, paper_range[0], paper_range[-1]))
  else:
    print("DIVERSITY for %s" % THE.permitted)
  miner, graph, lda_model, vocab = retrieve_graph_lda_data()
  paper_map = graph.get_papers_by_venue(THE.permitted)
  venue_topics = {}
  venue_heatmaps = {}
  valid_conferences = []
  venues = mysqldb.get_venues()
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


def test_dendo_heatmap(col_size, paper_range):
  row_labels = TOPICS_ALL
  column_labels = ["VEN%d" % i for i in range(col_size)]
  heatmap_arr = np.random.rand(len(row_labels), len(column_labels))
  make_dendo_heatmap(heatmap_arr, row_labels, column_labels,
                     "temp.png", paper_range)


def topic_evolution(venue=THE.permitted):
  print("TOPIC EVOLUTION for %s" % venue)
  miner, graph, lda_model, vocab = retrieve_graph_lda_data()
  paper_nodes = graph.get_paper_nodes(venue)
  topics_map = {}
  n_topics = lda_model.n_topics
  for paper_id, paper in paper_nodes.items():
    # if int(paper.venue) == 38:
    #   continue
    if int(paper.year) < 1992 or int(paper.year) > 2016: continue
    document = miner.documents[paper_id]
    year_topics = topics_map.get(paper.year, np.array([0] * n_topics))
    topics_map[paper.year] = np.add(year_topics, document.topics_count)
  yt_map = {}
  for year, t_count in topics_map.items():
    yt_map[year] = percent_sort(t_count)
  width = 0.6
  plts = []
  x_axis = np.arange(0, len(yt_map.keys()))
  y_offset = np.array([0] * len(yt_map.keys()))
  colors_dict = {}
  top_topic_count = 9
  # plt.figure(figsize=(8, 8))
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
  plt.xticks(x_axis, [str(y)[2:] for y in sorted(yt_map.keys(), key=lambda x: int(x))], fontsize=9)
  plt.yticks(np.arange(0, 101, 10))
  plt.ylim([0, 101])
  # Legends
  patches = []
  topics = []
  for index, (topic, color) in enumerate(colors_dict.items()):
    patches.append(mpatches.Patch(color=color, label='Topic %s' % str(topic)))
    topics.append(get_topics()[topic])

  plt.legend(tuple(patches), tuple(topics), loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=5, fontsize=7,
             handlelength=0.7)
  plt.savefig("figs/v3/%s/topic_evolution/topic_evolution_%s.png" % (THE.permitted, venue), bbox_inches='tight')
  plt.clf()


def top_cited_authors(graph, top_percent=0.01, min_year=None):
  authors = graph.get_papers_by_authors(THE.permitted)
  author_cites = []
  if top_percent is None:
    top_percent = 1
  for author_id, papers in authors.items():
    cite_count = 0
    for paper_id, year, __ in papers:
      if min_year is not None and int(year) < min_year: continue
      cited = graph.paper_nodes[paper_id].cited_count
      if is_not_none(cited):
        cite_count += int(cited)
    author_cites.append((author_id, cite_count))
  tops = sorted(author_cites, key=lambda x: x[1], reverse=True)[:int(top_percent * len(author_cites))]
  return set([t[0] for t in tops])


def top_contributed_authors(graph=None, top_percent=0.01, min_year=None, with_scores=True):
  if graph is None:
    graph = retrieve_graph()
  author_contributions = OrderedDict()
  for paper_id, paper in graph.get_paper_nodes(THE.permitted).items():
    if min_year is not None and int(paper.year) < min_year: continue
    if not paper.authors: continue
    authors = paper.authors.split(",")
    dist = harmonic_dist(len(authors))
    for d, author in zip(dist, authors):
      author_contributions[author] = author_contributions.get(author, []) + [d]
  author_list = []
  for author, contribution in author_contributions.items():
    author_list.append((author, sum(contribution)))
  tops = sorted(author_list, key=lambda x: x[1], reverse=True)[:int(top_percent * len(author_list))]
  if with_scores:
    return tops
  return [t[0] for t in tops]


def top_cited_contributed_authors(graph=None, top_percent=0.01, min_year=None, with_scores=True):
  if graph is None:
    graph = retrieve_graph()
  author_contributions = OrderedDict()
  for paper_id, paper in graph.get_paper_nodes(THE.permitted).items():
    if min_year is not None and int(paper.year) < min_year: continue
    cites = int(paper.cited_count) + 1 if is_not_none(paper.cited_count) else 1
    if not paper.authors: continue
    authors = paper.authors.split(",")
    # dist = harmonic_dist(len(authors))
    dist = uniform_dist(len(authors))
    for d, author in zip(dist, authors):
      author_contributions[author] = author_contributions.get(author, []) + [d * cites]
  author_list = []
  for author, contribution in author_contributions.items():
    author_list.append((author, sum(contribution)))
  tops = sorted(author_list, key=lambda x: x[1], reverse=True)[:int(top_percent * len(author_list))]
  if with_scores:
    return tops
  return [t[0] for t in tops]


def super_author(top_percents):
  print("SUPER AUTHOR for %s; TOP PERCENTS : %s" % (THE.permitted, top_percents))
  miner, graph, lda_model, vocab = retrieve_graph_lda_data()
  authors = graph.get_papers_by_authors(THE.permitted)
  author_publications = OrderedDict()
  for top_percent in top_percents:
    p_key = "%d" % (int(top_percent * 100)) + ' %'
    author_topics = {}
    tops = top_cited_authors(graph, top_percent)
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
      if val > 0:
        counter[val] += 1
    bar_x = []
    bar_y = []
    for key in sorted(counter.keys()):
      bar_x.append(key)
      bar_y.append(counter[key])
    bar_y = [0] * (lda_model.n_topics - len(bar_y)) + bar_y
    author_publications[p_key] = bar_y
  ind = np.arange(1, lda_model.n_topics + 1)  # the x locations for the groups
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
            bbox_to_anchor=(0.5, 1.15), ncol=4, fontsize=12)
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
    cites = int(paper.cited_count) if is_not_none(paper.cited_count) else 0
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


def authors_percent_in_papers_year(min_year=1992):
  print("#AUTHOR PERCENT vs YEAR for %s" % THE.permitted)
  graph = retrieve_graph()
  year_authors_map = OrderedDict()
  for _, paper in graph.get_paper_nodes(permitted=THE.permitted).items():
    year = paper.year
    if int(year) < min_year or int(year) > 2015: continue
    num_authors = len(paper.authors.split(","))
    year_authors_count = year_authors_map.get(year, {})
    key = str(num_authors) if num_authors < 7 else "7+"
    year_authors_count[key] = year_authors_count.get(key, 0) + 1
    year_authors_map[year] = year_authors_count
  year_author_percent_map = OrderedDict()
  keys = ["1", "2", "3", "4", "5", "6", "7+"]
  authors_count_year_map = OrderedDict()
  for year in sorted(year_authors_map.keys()):
    authors_count = year_authors_map[year]
    total = sum(authors_count.values())
    percent_map = []
    for key in keys:
      percent = round(authors_count.get(key, 0) * 100.0 / total, 2)
      percent_map.append(percent)
      authors_count_year_map[key] = authors_count_year_map.get(key, []) + [percent]
    year_author_percent_map[year] = percent_map
  colors = ["red", "blue", "darkslategray", "yellow", "darkmagenta", "cyan", "saddlebrown"]
  x_axis = sorted(year_authors_map.keys())
  x_indices = np.arange(1, len(x_axis) + 1)
  legends = []
  for i, key in enumerate(authors_count_year_map.keys()):
    plt.plot(x_indices, authors_count_year_map[key], color=colors[i])
    legends.append(key)
  plt.legend(legends, loc='upper right', ncol=2, fontsize=10)
  fig_name = "figs/v3/%s/authors_vs_year.png" % THE.permitted
  plt.title("Percentage of number of authors per year")
  plt.xticks(x_indices, x_axis, rotation=60)
  plt.xlabel("Year")
  plt.ylabel("Percentage(%)")
  plt.savefig(fig_name, bbox_inches='tight')
  plt.clf()


def author_counts_vs_cites_per_year(min_year=1992):
  print("#AUTHOR PERCENT vs CITES for %s" % THE.permitted)
  graph = retrieve_graph()
  year_authors_map = OrderedDict()
  for _, paper in graph.get_paper_nodes(permitted=THE.permitted).items():
    year = paper.year
    if int(year) < min_year or int(year) > 2015: continue
    num_authors = len(paper.authors.split(","))
    year_authors_count = year_authors_map.get(year, {})
    key = str(num_authors) if num_authors < 7 else "7+"
    cites = int(paper.cited_count) if is_not_none(paper.cited_count) else 0
    years_since = 2017 - int(year)
    avg_cites = cites / years_since
    year_authors_count[key] = year_authors_count.get(key, []) + [avg_cites]
    year_authors_map[year] = year_authors_count
  year_author_percent_map = OrderedDict()
  keys = ["1", "2", "3", "4", "5", "6", "7+"]
  authors_count_year_map = OrderedDict()
  for year in sorted(year_authors_map.keys()):
    authors_count = year_authors_map[year]
    # total = sum(authors_count.values())
    percent_map = []
    for key in keys:
      avg_cites = authors_count.get(key, [0])
      percent = round(sum(avg_cites) / len(avg_cites), 2)
      percent_map.append(percent)
      authors_count_year_map[key] = authors_count_year_map.get(key, []) + [percent]
    year_author_percent_map[year] = percent_map
  colors = ["red", "blue", "darkslategray", "yellow", "darkmagenta", "cyan", "saddlebrown"]
  x_axis = sorted(year_authors_map.keys())

  x_indices = np.arange(1, len(x_axis) + 1)
  legends = []
  for i, key in enumerate(authors_count_year_map.keys()):
    plt.plot(x_indices, authors_count_year_map[key], color=colors[i])
    legends.append(key)
  plt.legend(legends, loc='upper left', ncol=2, fontsize=10)
  fig_name = "figs/v3/%s/authors_vs_cites_per_year.png" % THE.permitted
  plt.title("Average cites per year for different number of coauthors")
  plt.xticks(x_indices, x_axis, rotation=60)
  plt.xlabel("Year")
  plt.ylabel("Average Cites Per Year")
  plt.savefig(fig_name, bbox_inches='tight')
  plt.clf()


def venue_type_vs_cites_per_year(min_year=1992):
  print("#VENUE TYPE vs CITES for %s" % THE.permitted)
  graph = retrieve_graph()
  paper_cites_map = OrderedDict()
  for _, paper in graph.get_paper_nodes(permitted=THE.permitted).items():
    year = paper.year
    if int(year) < min_year or int(year) > 2015: continue
    paper_cites_count = paper_cites_map.get(year, {})
    cites = int(paper.cited_count) if is_not_none(paper.cited_count) else 0
    key = "Conference" if is_true(paper.is_conference) else "Journal"
    years_since = 2017 - int(year)
    avg_cites = cites / years_since
    paper_cites_count[key] = paper_cites_count.get(year, []) + [avg_cites]
    paper_cites_map[year] = paper_cites_count

  keys = ["Conference", "Journal"]
  cites_count_year_map = OrderedDict()
  for year in sorted(paper_cites_map.keys()):
    paper_cites_count = paper_cites_map[year]
    for key in keys:
      all_cites = paper_cites_count.get(key, [0])
      avg_cites = round(sum(all_cites) / len(all_cites), 2)
      # avg_cites = np.median(all_cites)
      cites_count_year_map[key] = cites_count_year_map.get(key, []) + [avg_cites]

  colors = ["green", "red"]
  x_axis = sorted(paper_cites_map.keys())
  x_indices = np.arange(1, len(x_axis) + 1)
  legends = []
  for i, key in enumerate(keys):
    plt.plot(x_indices, cites_count_year_map[key], color=colors[i])
    legends.append(key)
  plt.legend(legends, loc='upper left', ncol=2, fontsize=10)
  fig_name = "figs/v3/%s/venue_type_vs_cites_per_year.png" % THE.permitted
  plt.title("Average cites per year for Conferences & Journals")
  plt.xticks(x_indices, x_axis, rotation=60)
  plt.xlabel("Year")
  plt.ylabel("Average Cites Per Year")
  plt.savefig(fig_name, bbox_inches='tight')
  plt.clf()


def author_bar(min_year=1992):
  print("AUTHOR BAR for %s" % THE.permitted)
  graph = retrieve_graph()
  year_authors_map = OrderedDict()
  for _, paper in graph.get_paper_nodes(permitted=THE.permitted).items():
    year = paper.year
    if int(year) < min_year: continue
    authors = paper.authors.split(",")
    year_authors_map[year] = year_authors_map.get(year, set([])).union(authors)
  bar_x, bar_index, bar_y = [], [], []
  for index, year in enumerate(sorted(year_authors_map.keys())):
    bar_x.append(year)
    bar_index.append(index + 1)
    bar_y.append(len(year_authors_map[year]))
  plt.figure(figsize=(8, 3))
  plt.bar(bar_index, bar_y, color='blue', align='center')
  plt.xticks(bar_index, bar_x, rotation=45)
  plt.xlabel('Year')
  plt.ylabel('# of Authors')
  plt.savefig("figs/v3/%s/author_count.png" % THE.permitted, bbox_inches='tight')
  plt.clf()


def print_top_cited_authors(top_percent=None, min_year=None):
  graph = retrieve_graph()
  tops = top_cited_authors(graph, top_percent=top_percent, min_year=min_year)
  author_papers = graph.get_papers_by_authors(THE.permitted)
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
        if is_not_none(graph.paper_nodes[paper_id].cited_count):
          total_cites += int(graph.paper_nodes[paper_id].cited_count)
        counts += 1
      top_tups.append((author.name, counts, total_cites))
  top_tups = sorted(top_tups, key=lambda x: x[-1], reverse=True)
  if min_year:
    file_name = "figs/v3/%s/top_cited_authors_%s.txt" % (THE.permitted, min_year)
  else:
    file_name = "figs/v3/%s/top_cited_authors_all.txt" % THE.permitted
  with open(file_name, "wb") as f:
    for top_tup in top_tups:
      f.write(str(top_tup))
      f.write("\n")


def print_top_cited_contributed_authors(top_percent=0.01, min_year=None):
  graph = retrieve_graph()
  top_tups = top_cited_contributed_authors(graph, top_percent, min_year)
  if min_year:
    file_name = "figs/v3/%s/top_cited_contributed_authors_%s.txt" % (THE.permitted, min_year)
  else:
    file_name = "figs/v3/%s/top_cited_contributed_authors_all.txt" % THE.permitted
  with open(file_name, "wb") as f:
    for top_tup in top_tups:
      f.write(str(top_tup))
      f.write("\n")


def make_author_dict(author_tuples):
  names_to_ids, ids_to_names = OrderedDict(), OrderedDict()
  for index, tup in enumerate(author_tuples):
    names_to_ids[tup[0]] = index
    ids_to_names[index] = tup[0]
  n = len(author_tuples)
  author_scores = np.full(n, 1 / n, dtype=np.float64)
  return names_to_ids, ids_to_names, author_scores


def list_out(lst):
  for i in range(len(lst) - 1):
    yield lst[i], set(lst[:i] + lst[i + 1:])


def make_coauthor_map(graph, names_to_ids, min_year):
  top_authors = set(names_to_ids.keys())
  coauthor_count = {i: set() for i in range(len(top_authors))}
  author_cite_count = {i: 0 for i in range(len(top_authors))}
  for paper_id, paper in graph.paper_nodes.items():
    if min_year and int(paper.year) < min_year: continue
    authors = paper.authors.split(",")
    cites = int(paper.cited_count) if paper.cited_count and paper.cited_count != 'None' else 0
    if len(authors) <= 1: continue
    for author_name, co_authors_name in list_out(authors):
      if author_name not in top_authors: continue
      author_index = names_to_ids[author_name]
      co_authors_index = set([names_to_ids[co_author_name] for co_author_name in co_authors_name if co_author_name in top_authors])
      coauthor_count[author_index] = coauthor_count[author_index].union(co_authors_index)
      author_cite_count[author_index] += cites
  coauthor_count = {i: np.array(sorted(co_authors), dtype=np.int32) for i, co_authors in coauthor_count.items()}
  total_cites = sum(author_cite_count.values())
  author_cite_count = {i: author_cite_count[i] / total_cites for i in author_cite_count.keys()}
  return coauthor_count, author_cite_count


def page_rank(d=0.45, top_percent=0.01, min_year=None, iterations=1000):
  graph = retrieve_graph()
  top_tups = top_cited_contributed_authors(graph, top_percent, min_year)
  names_to_ids, ids_to_names, page_rank_scores = make_author_dict(top_tups)
  coauthor_count, author_cite_count = make_coauthor_map(graph, names_to_ids, min_year)
  self_score = 1 / page_rank_scores.shape[0]
  for _ in xrange(iterations):
    for i in shuffle(range(page_rank_scores.shape[0])):
      co_authors_i = coauthor_count[i]
      ext_score = 0
      for j in co_authors_i:
        links = len(coauthor_count[j])
        if links > 0:
          ext_score += page_rank_scores[j] / links
      page_rank_scores[i] = (1 - d) * self_score * author_cite_count[i] + d * ext_score
  top_author_indices = np.argsort(page_rank_scores)[::-1]
  if min_year:
    file_name = "figs/v3/%s/top_page_rank_authors_%s.txt" % (THE.permitted, min_year)
  else:
    file_name = "figs/v3/%s/top_page_rank_authors_all.txt" % THE.permitted
  with open(file_name, "wb") as f:
    f.write("Name, Score\n")
    for index in top_author_indices:
      f.write("%s, %f\n" % (ids_to_names[index], page_rank_scores[index]))
  top_author_names = [ids_to_names[index] for index in top_author_indices]
  return top_author_names, top_author_indices, page_rank_scores[top_author_indices]


def score_plotter(file):
  x, y = [], []
  with open(file) as f:
    line = f.readline().split(",")
    while len(line) > 1:
      x.append(int(line[0]))
      y.append(float(line[1]))
      line = f.readline().split(",")
  plt.plot(x, y, 'r--')
  plt.xlabel('Topics ->')
  plt.ylabel('Perplexity ->')
  plt.savefig("figs/v3/%s/perplexity.png" % THE.permitted, bbox_inches='tight')
  plt.clf()


def stat_author_counts_vs_cites_per_year(min_year=1992):
  print("#AUTHOR PERCENT vs CITES for %s" % THE.permitted)
  graph = retrieve_graph()
  year_authors_map = OrderedDict()
  for _, paper in graph.get_paper_nodes(permitted=THE.permitted).items():
    year = paper.year
    if int(year) < min_year or int(year) > 2015: continue
    num_authors = len(paper.authors.split(","))
    year_authors_count = year_authors_map.get(year, {})
    key = str(num_authors) if num_authors < 7 else "7+"
    cites = int(paper.cited_count) if is_not_none(paper.cited_count) else 0
    years_since = 2017 - int(year)
    avg_cites = cites / years_since
    year_authors_count[key] = year_authors_count.get(key, []) + [avg_cites]
    year_authors_map[year] = year_authors_count
  keys = ["1", "2", "3", "4", "5", "6", "7+"]
  file_name = "figs/v3/%s/stats/authors_vs_cites.txt" % THE.permitted
  f = open(file_name, "wb")
  for year in sorted(year_authors_map.keys()):
    authors_count = year_authors_map[year]
    percent_map = []
    for key in keys:
      cites = authors_count.get(key, [0])
      percent_map.append([key] + cites)
    f.write("\n## %s\n" % year)
    sk(percent_map, f)
  f.close()


def stat_venue_type_vs_cites_per_year(min_year=1992):
  print("#VENUE TYPE vs CITES for %s" % THE.permitted)
  graph = retrieve_graph()
  paper_cites_map = OrderedDict()
  for _, paper in graph.get_paper_nodes(permitted=THE.permitted).items():
    year = paper.year
    if int(year) < min_year or int(year) > 2015: continue
    paper_cites_count = paper_cites_map.get(year, {})
    cites = int(paper.cited_count) if is_not_none(paper.cited_count) else 0
    key = "Conference" if is_true(paper.is_conference) else "Journal"
    years_since = 2017 - int(year)
    avg_cites = cites / years_since
    paper_cites_count[key] = paper_cites_count.get(key, []) + [avg_cites]
    paper_cites_map[year] = paper_cites_count

  keys = ["Conference", "Journal"]
  file_name = "figs/v3/%s/stats/venue_type_vs_cites.txt" % THE.permitted
  f = open(file_name, "wb")
  for year in sorted(paper_cites_map.keys()):
    paper_cites_count = paper_cites_map[year]
    percent_map = []
    for key in keys:
      all_cites = paper_cites_count.get(key, [0])
      percent_map.append([key[:4]] + all_cites)
    print("\n## %s" % year)
    f.write("\n## %s\n" % year)
    sk(percent_map, f)
  f.close()


def _main():
  # test_dendo_heatmap(24, range(2009, 2017))
  # reporter()
  # paper_bar()
  # diversity("heatmap_09_16", range(2009, 2017))
  # diversity("heatmap_01_08", range(2001, 2009))
  # diversity("heatmap_93_00", range(1992, 2001))
  topic_evolution(venue="all")
  topic_evolution(venue="conferences")
  topic_evolution(venue="journals")
  # super_author([0.01, 0.1, 0.2, 1.0])
  # get_top_papers(year_from=2009)
  # get_top_papers()
  # authors_percent_in_papers_year(1993)
  # author_bar()
  # print_top_cited_authors(0.01)
  # print_top_cited_authors(0.01, 2009)
  # print_top_cited_contributed_authors(0.01)
  # print_top_cited_contributed_authors(0.01, 2009)
  # diversity("heatmap_all")
  # author_counts_vs_cites_per_year(1993)
  # page_rank()
  # page_rank(min_year=2009)
  # venue_type_vs_cites_per_year(1993)
  # stat_author_counts_vs_cites_per_year()
  # stat_venue_type_vs_cites_per_year()


def _store():
  store_graph_lda_data()

if __name__ == "__main__":
  # _main()
  # score_plotter("figs/v3/all/scores.csv")
  # _store()
  paper_and_author_growth()
