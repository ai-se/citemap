from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

__author__ = "bigfatnoob"

from utils.lib import O, Memoized
import numpy as np
from collections import OrderedDict
from network.mine import cite_graph, Miner
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from db import mysqldb
import pandas as pd
from sklearn.feature_extraction import text
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram
from expts.settings import dend as dend_settings
import cPickle as cPkl
from sklearn.externals import joblib
from utils.sk import bootstrap, qDemo

GRAPH_CSV = "data/citemap_v10.csv"

# For 11 TOPICS
ALPHA = 0.22359
BETA = 0.53915
ITERATIONS = 100

TOPICS_ALL = ["Program Analysis", "Requirements", "Metrics", "Applications",
              "Performance", "Miscellaneous", "Testing", "Source Code",
              "Architecture", "Modeling", "Developer"]

TOPIC_THRESHOLD = 3

# Global Settings
THE = O()
THE.permitted = "all"  # conference/journal/all
THE.version = "v4"
THE.use_numeric = False

STOP_WORDS = text.ENGLISH_STOP_WORDS.union(['software', 'engineering', 'paper', 'study', 'based',
                                            'results', 'approach', 'case', 'workshop', 'international', 'research',
                                            'conference', 'introduction', 'editors', 'article', 'issue', 'month',
                                            'copyright', 'special', 'used', 'using', 'use', 'studies', 'review',
                                            'editorial', 'report', 'book', 'ieee', 'published', 'science', 'column',
                                            'author', 'proposed', 'icse', 'article', 'year', 'articles', 'page', '2000',
                                            '2004', 'papers', 'computer', 'held', 'editor'])


COLORS_ALL = ["lightgray", "red", "blue", "darkslategray",
              "yellow", "darkmagenta", "cyan", "saddlebrown",
              "orange", "lime", "hotpink"]


MIN_DIVERSITY_SCORE = 0.075


def mkdir(directory):
  """
  Implements the "mkdir" linux function
  :param directory:
  :return:
  """
  if directory and not os.path.exists(directory):
    os.makedirs(directory)


def get_n_topics():
  if THE.permitted == "journals":
    return 7
  if THE.permitted == "all":
    return 11


def get_color(index):
  return COLORS_ALL[index]


def is_not_none(s):
  return s and s != 'None'


def is_true(val):
  return val in [True, 'True', 'true']


def shorter_names(name):
  name_map = {
      "SOFTWARE": "S/W",
      "SIGSOFT": "NOTES",
      "MODELS": "MDLS"
  }
  if name in name_map:
    return name_map[name]
  return name


CONFERENCES = [shorter_names(conf.acronym) for conf in mysqldb.get_conferences()]


@Memoized
def retrieve_graph_lda_data():
  """
  Fetch stored metadata
  :return:
  """
  graph_file = 'cache/%s/%s/graph.pkl' % (THE.version, THE.permitted)
  vectorizer_file = 'cache/%s/%s/vectorizer.pkl' % (THE.version, THE.permitted)
  doc_2_vec_file = 'cache/%s/%s/doc_2_vec.pkl' % (THE.version, THE.permitted)
  documents_file = 'cache/%s/%s/documents.pkl' % (THE.version, THE.permitted)
  lda_model_file = 'cache/%s/%s/lda_model.pkl' % (THE.version, THE.permitted)
  vocabulary_file = 'cache/%s/%s/vocabulary.pkl' % (THE.version, THE.permitted)
  if os.path.isfile(graph_file) and os.path.isfile(vectorizer_file) \
          and os.path.isfile(doc_2_vec_file) and os.path.isfile(documents_file) \
          and os.path.isfile(lda_model_file) and os.path.isfile(vocabulary_file):
    with open(graph_file) as f:
      graph = cPkl.load(f)
    miner = Miner(graph)
    with open(vectorizer_file) as f:
      miner.vectorizer = cPkl.load(f)
    with open(doc_2_vec_file) as f:
      miner.doc_2_vec = joblib.load(f)
    with open(documents_file) as f:
      miner.documents = cPkl.load(f)
    with open(lda_model_file) as f:
      lda_model = cPkl.load(f)
    with open(vocabulary_file) as f:
      vocab = cPkl.load(f)
  else:
    miner, graph, lda_model, vocab = store_graph_lda_data()
  return miner, graph, lda_model, vocab


def store_graph_lda_data():
  miner, graph, lda_model, vocab = get_graph_lda_data()
  folder_name = 'cache/%s/%s' % (THE.version, THE.permitted)
  mkdir(folder_name)
  with open('cache/%s/%s/graph.pkl' % (THE.version, THE.permitted), 'wb') as f:
    cPkl.dump(graph, f, cPkl.HIGHEST_PROTOCOL)
  with open('cache/%s/%s/vectorizer.pkl' % (THE.version, THE.permitted), 'wb') as f:
    cPkl.dump(miner.vectorizer, f, cPkl.HIGHEST_PROTOCOL)
  with open('cache/%s/%s/doc_2_vec.pkl' % (THE.version, THE.permitted), 'wb') as f:
    joblib.dump(miner.doc_2_vec, f)
  with open('cache/%s/%s/documents.pkl' % (THE.version, THE.permitted), 'wb') as f:
    cPkl.dump(miner.documents, f, cPkl.HIGHEST_PROTOCOL)
  with open('cache/%s/%s/lda_model.pkl' % (THE.version, THE.permitted), 'wb') as f:
    cPkl.dump(lda_model, f, cPkl.HIGHEST_PROTOCOL)
  with open('cache/%s/%s/vocabulary.pkl' % (THE.version, THE.permitted), 'wb') as f:
    cPkl.dump(vocab, f, cPkl.HIGHEST_PROTOCOL)
  return miner, graph, lda_model, vocab


def retrieve_graph():
  graph_file = 'cache/%s/%s/graph.pkl' % (THE.version, THE.permitted)
  if os.path.isfile(graph_file):
    with open(graph_file) as f:
      graph = cPkl.load(f)
  else:
    graph = cite_graph(GRAPH_CSV)
    with open(graph_file, 'wb') as f:
      cPkl.dump(graph, f, cPkl.HIGHEST_PROTOCOL)
  return graph


@Memoized
def get_graph_lda_data():
  graph = cite_graph(GRAPH_CSV)
  miner = Miner(graph, THE.permitted)
  lda_model, vocab = miner.lda(get_n_topics(), n_iter=ITERATIONS, alpha=ALPHA, beta=BETA, stop_words=STOP_WORDS)
  return miner, graph, lda_model, vocab


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
    # legends.append(TOPICS_ALL[index])
    print('%2d : %16s : %s' % (index, get_topics()[index].upper(), ', '.join(topic_words)))
  plt.legend(legends, loc='upper right')
  plt.title(fig_name)
  plt.xlabel("Term Index")
  plt.ylabel("Log. Word Score")
  plt.savefig("figs/%s/%s/%s.png" % (THE.version, THE.permitted, fig_name))
  fig.clf()


def reporter():
  print("TOPIC REPORTS for %s" % THE.permitted)
  miner, graph, lda_model, vocab = retrieve_graph_lda_data()
  report(lda_model, vocab, n_top_words=12)


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


def get_topics():
  if THE.use_numeric:
    return ["Topic-%2d" % i for i in range(get_n_topics())]
  if THE.permitted == "all":
    return TOPICS_ALL


def make_heatmap(arr, row_labels, column_labels, figname, paper_range):
  plt.figure(figsize=(4, 3))
  df = pd.DataFrame(arr, columns=column_labels, index=row_labels)
  cax = plt.matshow(df, interpolation='nearest', cmap='hot_r')
  plt.colorbar(cax)
  plt.xticks(np.arange(len(list(df.columns))), list(df.columns), rotation="vertical")
  plt.yticks(np.arange(len(list(df.index))), list(df.index))
  [tick.set_color("red") if tick.get_text() in CONFERENCES else tick.set_color("green")
      for tick in plt.gca().get_xticklabels()]
  if paper_range:
    plt.title("Topics to Venue Distribution(%d - %d)" % (paper_range[0], paper_range[-1]), y=1.2)
  else:
    plt.title("Topics to Venue Distribution", y=1.2)
  plt.savefig(figname, bbox_inches='tight')
  plt.clf()


def make_dendo_heatmap(arr, row_labels, column_labels, figname, paper_range, save_labels=False):
  settings = dend_settings.get("dend_%d_%d" % (len(row_labels), len(column_labels)), None)
  if settings is None:
    print("ERROR: Configure Dendogram settings for %d rows and %d columns" % (len(row_labels), len(column_labels)))
    return
  sums = np.sum(arr, axis=1)
  sorted_indices = np.argsort(sums)[::-1]
  arr = arr[sorted_indices]
  row_labels = np.array(row_labels)[sorted_indices]
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
  for i in axd2.spines.values():
    i.set_visible(False)
  df_rowclust = df.ix[range(len(row_labels))][col_dendr['leaves']]
  # Save Labels(This can be reused in reordering heatmap later)
  if save_labels:
    label_file = "figs/%s/%s/stats/heatmap_data.pkl" % (THE.version, THE.permitted)
    label_data = {
        "rows": df_rowclust.index.values,
        "columns": df_rowclust.columns.values,
        "data": df_rowclust
    }
    with open(label_file, "wb") as f:
      cPkl.dump(label_data, f, cPkl.HIGHEST_PROTOCOL)

  # plot heatmap
  axm = fig.add_axes(settings.plot_axes)
  cax = axm.matshow(df_rowclust, interpolation='nearest', cmap='hot_r')
  if THE.permitted == "all":
    fig.colorbar(cax, location='bottom')
  else:
    fig.colorbar(cax)
  axm.set_xticks(np.arange(len(list(df_rowclust.columns))))
  axm.set_xticklabels(list(df_rowclust.columns), rotation="vertical")
  [tick.set_color("red") if tick.get_text() in CONFERENCES else tick.set_color("green")
      for tick in axm.get_xticklabels()]
  axm.set_yticks(np.arange(len(list(df_rowclust.index))))
  axm.set_yticklabels(list(df_rowclust.index))
  if paper_range:
    plt.title("Clustered topics to Venue Distribution(%d - %d)" % (paper_range[0], paper_range[-1]), y=-0.2)
  else:
    plt.title("Clustered topics to Venue Distribution", y=-0.2)
  plt.savefig(figname, bbox_inches='tight')
  plt.clf()


def diversity(fig_name, paper_range=None, min_diversity_score=MIN_DIVERSITY_SCORE, save_labels=False):
  """
  Heat map in paper
  :param fig_name:
  :param paper_range:
  :param min_diversity_score:
  :param save_labels:
  :return:
  """
  if paper_range:
    print("DIVERSITY for %s between %d - %d" % (THE.permitted, paper_range[0], paper_range[-1]))
  else:
    print("DIVERSITY for %s" % THE.permitted)
  miner, graph, lda_model, vocab = retrieve_graph_lda_data()
  paper_map = graph.get_papers_by_venue(THE.permitted)
  venue_heatmaps = {}
  venues = mysqldb.get_venues()
  for conference_id, papers in paper_map.items():
    topics = np.array([0] * lda_model.n_topics)
    for tup in yearize(papers).items():
      if paper_range and tup[0] not in paper_range: continue
      for paper_id in tup[1]:
        topics = np.add(topics, miner.documents[paper_id].topics_count)
    if sum(topics) > 0:
      venue_heatmaps[conference_id] = topics
  # row_labels = [str(ind) + "-" + name for ind, name in zip(range(lda_model.n_topics), get_topics())]
  row_labels = np.array([name for ind, name in zip(range(lda_model.n_topics), get_topics())])
  # row_labels = ["%2d" % ind for ind in range(lda_model.n_topics)]
  column_labels = [shorter_names(v.acronym) for c, v in venues.items() if v.id in venue_heatmaps]
  # Heatmap
  heatmap_arr = []
  for conference_id in sorted(venue_heatmaps.keys(), key=lambda x: int(x)):
    tot = sum(venue_heatmaps[conference_id])
    dist = [top / tot for top in venue_heatmaps[conference_id]]
    heatmap_arr.append(dist)
  heatmap_arr = np.transpose(heatmap_arr)
  heatmap_arr[heatmap_arr < min_diversity_score] = 0.0
  row_sums = np.sum(heatmap_arr, axis=1)
  heatmap_arr = heatmap_arr[np.where(row_sums > 0)]
  row_labels = row_labels[np.where(row_sums > 0)]
  make_heatmap(heatmap_arr, row_labels, column_labels,
               "figs/%s/%s/diversity/%s.png" % (THE.version, THE.permitted, fig_name), paper_range)
  make_dendo_heatmap(heatmap_arr, row_labels, column_labels,
                     "figs/%s/%s/diversity/%s_dend.png" % (THE.version, THE.permitted, fig_name),
                     paper_range, save_labels)


def topic_evolution(venue=THE.permitted):
  """
  Stacked bar-charts
  :param venue:
  :return:
  """
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
  plt.figure(figsize=(8, 6))
  for index in range(top_topic_count):
    bar_val, color = [], []
    for year in sorted(yt_map.keys(), key=lambda x: int(x)):
      topic = yt_map[year][index]
      if topic[0] not in colors_dict:
        colors_dict[topic[0]] = get_color(topic[0])
      color.append(colors_dict[topic[0]])
      bar_val.append(topic[1])
    plts.append(plt.bar(x_axis, bar_val, width, color=color, bottom=y_offset))
    y_offset = np.add(y_offset, np.array(bar_val))
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
  plt.legend(tuple(patches), tuple(topics), loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=5, fontsize=9,
             handlelength=0.7)
  plt.savefig("figs/%s/%s/topic_evolution/topic_evolution_%s.png" % (THE.version, THE.permitted, venue),
              bbox_inches='tight')
  plt.clf()


def test_dendo_heatmap(col_size, paper_range, save_labels):
  row_labels = TOPICS_ALL
  column_labels = ["VEN%d" % i for i in range(col_size)]
  heatmap_arr = np.random.rand(len(row_labels), len(column_labels))
  make_dendo_heatmap(heatmap_arr, row_labels, column_labels,
                     "temp.png", paper_range, save_labels)


def get_author_genders():
  authors = mysqldb.get_authors()
  gender_map = {}
  for a_id, node in authors.items():
    if not node.gender:
      continue
    gender_map[a_id] = node.gender
  return gender_map


def gender_over_time(min_year=1992):
  print("Gender difference for %s since %d" % (THE.permitted, min_year))
  graph = retrieve_graph()
  gender_publish_map = {}
  author_gender_map = get_author_genders()
  for _, paper in graph.get_paper_nodes(permitted=THE.permitted).items():
    year = int(paper.year)
    if year < min_year or year > 2015: continue
    author_ids = paper.author_ids.split(",")
    gender_score = gender_publish_map.get(year, {'m': 0, 'f': 0})
    for author_id in author_ids:
      gender = author_gender_map.get(author_id, None)
      if gender:
        gender_score[gender] += 1
    gender_publish_map[year] = gender_score
  male_counts, female_counts = [], []
  male_percents, female_percents = [], []
  x_axis = []
  for year in sorted(gender_publish_map.keys()):
    x_axis.append(year)
    male_count = gender_publish_map[year]['m']
    female_count = gender_publish_map[year]['f']
    total = male_count + female_count
    male_counts.append(male_count)
    female_counts.append(female_count)
    male_percents.append(100 * male_count / total)
    female_percents.append(100 * female_count / total)
  plt.plot(x_axis, male_counts, color='red')
  plt.plot(x_axis, female_counts, color='blue')
  plt.legend(['male', 'female'], loc='upper right', ncol=2, fontsize=10)
  count_fig_name = "figs/%s/%s/gender/genders_vs_year.png" % (THE.version, THE.permitted)
  plt.title("Male and Female authors per year")
  plt.xlabel("Year")
  plt.ylabel("# Authors")
  plt.savefig(count_fig_name, bbox_inches='tight')
  plt.clf()
  plt.plot(x_axis, male_percents, color='red')
  plt.plot(x_axis, female_percents, color='blue')
  plt.legend(['male', 'female'], loc='upper right', ncol=2, fontsize=10)
  percent_fig_name = "figs/%s/%s/gender/percent_genders_vs_year.png" % (THE.version, THE.permitted)
  plt.title("% of Male and Female authors per year")
  plt.xlabel("Year")
  plt.ylabel("% of Authors")
  plt.savefig(percent_fig_name, bbox_inches='tight')
  plt.clf()


def gender_topics(paper_range=None, file_name="topic_contribution"):
  def index_by_year(tups):
    y_comm = {}
    for tup in tups:
      comm = y_comm.get(tup[1], set())
      comm.add(tup[0])
      y_comm[tup[1]] = comm
    return y_comm

  def normalize(arr):
    arr_tot = sum(arr)
    if arr_tot == 0:
      return [0] * len(arr)
    return [arr_i / arr_tot for arr_i in arr]

  miner, graph, lda_model, vocab = get_graph_lda_data()
  p_venues = graph.get_papers_by_venue(permitted=THE.permitted)
  author_gender_map = get_author_genders()
  venue_topics = OrderedDict()
  for venue_id in sorted(mysqldb.get_venues().keys()):
    venue_id = str(venue_id)
    year_papers = index_by_year(p_venues[venue_id])
    both_genders_topics = []
    male_topics = []
    female_topics = []
    for year in sorted(year_papers.keys(), key=lambda y: int(y)):
      if (paper_range is not None) and (int(year) not in paper_range):
        continue
      papers = year_papers.get(year, [])
      if len(papers) > 0:
        for paper_id in papers:
          paper = graph.paper_nodes[paper_id]
          author_ids = paper.author_ids.strip().split(",")
          paper_topics = miner.documents[paper_id].topics_count
          # unit_paper_topics = [t / len(author_ids) for t in paper_topics]
          male_count, female_count = 0, 0
          for author_id in author_ids:
            gender = author_gender_map.get(author_id, None)
            if gender == 'm':
              male_count += 1
            elif gender == 'f':
              female_count += 1
          normalized_topics = normalize(paper_topics)
          if sum(normalized_topics) > 0:
            both_genders_topics.append(normalized_topics)
          if male_count > 0:
            male_topics.append(normalized_topics)
          if female_count > 0:
            female_topics.append(normalized_topics)
          # male_paper_topics = [male_count * t for t in unit_paper_topics]
          # female_paper_topics = [female_count * t for t in unit_paper_topics]
          # both_genders_topics.append(normalize(unit_paper_topics))
          # male_topics.append(normalize(male_paper_topics))
          # female_topics.append(normalize(female_paper_topics))
    venue_topics[venue_id] = {
        "all": both_genders_topics,
        "male": male_topics,
        "female": female_topics
    }
  save_file = "figs/%s/%s/gender/%s.pkl" % (THE.version, THE.permitted, file_name)
  with open(save_file, "wb") as f:
    cPkl.dump(venue_topics, f, cPkl.HIGHEST_PROTOCOL)


def compare_gender_topics(source="topic_contribution", target="stat"):
  gender_file = "figs/%s/%s/gender/%s.pkl" % (THE.version, THE.permitted, source)
  with open(gender_file) as f:
    venue_topics = cPkl.load(f)
  stat_map = {}
  for venue_id, topic_map in venue_topics.items():
    both = np.transpose(topic_map["all"])
    male = np.transpose(topic_map["male"])
    female = np.transpose(topic_map["female"])
    stat_topic_map = {}
    for i in range(get_n_topics()):
      print("Venue %s, Topic %d" % (venue_id, i))
      both_i = both[i, ]
      male_i = male[i, ]
      female_i = female[i, ]
      a_v_m = bootstrap(both_i, male_i)
      a_v_f = bootstrap(both_i, female_i)
      m_v_f = bootstrap(female_i, male_i)
      stat_topic_map[i] = (a_v_m, a_v_f, m_v_f)
    stat_map[venue_id] = stat_topic_map
  stat_file = "figs/%s/%s/gender/%s.pkl" % (THE.version, THE.permitted, target)
  with open(stat_file, "wb") as f:
    cPkl.dump(stat_map, f, cPkl.HIGHEST_PROTOCOL)


def print_gender_topics(file_name):
  stat_file = "figs/%s/%s/gender/%s.pkl" % (THE.version, THE.permitted, file_name)
  with open(stat_file) as f:
    stat_map = cPkl.load(f)
  # print(stat_map.keys())
  # for venue_id in sorted(stat_map.keys(), lambda k: int(k)):
  venues = mysqldb.get_venues()
  venue_names = {shorter_names(venues[v_id].acronym): v_id for v_id in sorted(venues.keys(), key=lambda k: int(k))}
  with open("figs/%s/%s/stats/heatmap_data.pkl" % (THE.version, THE.permitted)) as f:
    axis_data = cPkl.load(f)
    permitted_topics = axis_data['rows']
    permitted_venues = axis_data['columns']
    valid_data = axis_data['data']
  print("," + ",".join(permitted_venues))
  topics = get_topics()
  for topic_name in permitted_topics:
    topic_id = topics.index(topic_name)
    arr = [topic_name]
    for venue in permitted_venues:
      venue_id = venue_names[venue]
      if venue_id in stat_map and valid_data[venue][topic_name] > 0:
        m_v_f = 'T' if stat_map[venue_id][topic_id][2] else 'F'
      else:
        m_v_f = ''
      # print("Venue: %s, Topic: %d, Stat: %s" % (venue_id, topic, m_v_f))
      arr.append(m_v_f)
    print(",".join(arr))


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
  with open("figs/%s/%s/top_papers/%s.csv" % (THE.version, THE.permitted, suffix), "wb") as f:
    f.write("Index, Cites, Year, Title, Authors\n")
    for index in range(get_n_topics()):
      top_papers[index] = sorted(top_papers[index], reverse=True)[:top_count]
      for paper in top_papers[index]:
        paper = paper[0]
        f.write("%s, %d, %d, \"%s\", \"%s\"\n" % (topic_names[index], paper[0], paper[-1], paper[1], paper[2]))


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
  file_name = "figs/%s/%s/stats/venue_type_vs_cites.txt" % (THE.version, THE.permitted)
  f = open(file_name, "wb")
  for year in sorted(paper_cites_map.keys()):
    paper_cites_count = paper_cites_map[year]
    percent_map = []
    for key in keys:
      all_cites = paper_cites_count.get(key, [0])
      percent_map.append([key[:4]] + all_cites)
    print("\n## %s" % year)
    f.write("\n## %s\n" % year)
    qDemo(percent_map, f)
  f.close()


def stat_cites_per_year(min_year=1992):
  print("#CITES PER YEAR for %s" % THE.permitted)
  graph = retrieve_graph()
  paper_cites_map = OrderedDict()
  for _, paper in graph.get_paper_nodes(permitted=THE.permitted).items():
    year = paper.year
    if int(year) < min_year or int(year) > 2015: continue
    paper_cites_count = paper_cites_map.get(year, [])
    cites = int(paper.cited_count) if is_not_none(paper.cited_count) else 0
    years_since = 2017 - int(year)
    avg_cites = cites / years_since
    paper_cites_count.append(avg_cites)
    paper_cites_map[year] = paper_cites_count

  file_name = "figs/%s/%s/stats/stats_cites.txt" % (THE.version, THE.permitted)
  f = open(file_name, "wb")
  for year in sorted(paper_cites_map.keys()):
    paper_cites_count = paper_cites_map[year]
    percent_map = [["C+J"] + paper_cites_count]
    print("\n## %s" % year)
    f.write("\n## %s\n" % year)
    qDemo(percent_map, f)
  f.close()


def percentile_lines_per_year():
  stats = [[0.12 , 0.36 , 1.36 , 0.52],
           [0.13, 0.33, 1.38, 0.46],
           [0.13, 0.3, 1.35, 0.43],
           [0.14, 0.36, 1.59, 0.5],
           [0.1, 0.29, 1.48, 0.43],
           [0.1, 0.3, 1.2, 0.45],
           [0.16, 0.53, 1.84, 0.68],
           [0.06, 0.28, 1.0, 0.39],
           [0.06, 0.35, 1.35, 0.53],
           [0.19, 0.5, 1.94, 0.69],
           [0.27, 0.6, 2.07, 0.8],
           [0.36, 0.86, 2.5, 0.93],
           [0.31, 0.77, 2.46, 1.08],
           [0.25, 0.75, 2.25, 0.92],
           [0.27, 0.91, 2.82, 1.18],
           [0.5, 1.1, 3.1, 1.3],
           [0.56, 1.22, 3.22, 1.44],
           [0.5, 1.13, 3.25, 1.38],
           [0.71, 1.29, 3.86, 1.57],
           [0.67, 1.5, 3.5, 1.67],
           [0.8, 1.6, 3.8, 1.8],
           [0.75, 1.75, 4.25, 1.75],
           [1.0, 1.67, 4.33, 2.0],
           [1.0, 1.5, 4.0, 2.0]]
  stats = map(list, zip(*stats))
  x_axis = range(1992, 2016)
  p50 = stats[0]
  p70 = stats[1]
  p90 = stats[2]
  iqr = stats[3]
  plt.plot(x_axis, p50, '-b', label='50P')
  plt.plot(x_axis, p70, '-r', label='70P')
  plt.plot(x_axis, p90, '-g', label='90P')
  # plt.plot(x_axis, iqr, linestyle='--', color='black', label='IQR')
  # plt.legend(loc='upper left')
  plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=5, fontsize=12,
             handlelength=0.7)
  plt.xlabel("Published Year", fontsize=12)
  plt.ylabel("Average Cites Per Year", fontsize=12)
  plt.savefig("figs/%s/%s/stats/average_cites_percentile.png" % (THE.version, THE.permitted), bbox_inches='tight')
  plt.clf()


def topical_gender_diff(paper_range=None, venue=THE.permitted):
  print("Gender TOPICAL EVOLUTION for %s" % venue)
  miner, graph, lda_model, vocab = retrieve_graph_lda_data()
  paper_nodes = graph.get_paper_nodes(venue)
  n_topics = lda_model.n_topics
  author_gender_map = get_author_genders()
  male_counts, female_counts = {}, {}
  male_contributions, female_contributions = {}, {}
  for paper_id, paper in paper_nodes.items():
    if paper_range is not None and int(paper.year) not in paper_range:
      continue
    document = miner.documents[paper_id]
    author_ids = paper.author_ids.strip().split(",")
    topics_count = document.topics_count / len(author_ids)
    male_count, female_count = 0, 0
    for author_id in author_ids:
      gender = author_gender_map.get(author_id, None)
      if gender == 'm':
        # male_count += 1
        male_count = 1
      elif gender == 'f':
        # female_count += 1
        female_count = 1
    if male_count > 0:
      male_contribution = male_contributions.get(paper.year, np.array([0] * n_topics))
      male_contributions[paper.year] = np.add(male_contribution, topics_count)
      male_counts[paper.year] = male_counts.get(paper.year, 0) + 1
    if female_count > 0:
      female_contribution = female_contributions.get(paper.year, np.array([0] * n_topics))
      female_contributions[paper.year] = np.add(female_contribution, topics_count)
      female_counts[paper.year] = female_counts.get(paper.year, 0) + 1
  delta_map = OrderedDict()
  for year in sorted(male_counts.keys(), key=lambda x: int(x)):
    delta_map[year] = (male_contributions[year] / male_counts[year]) - (female_contributions[year] / female_counts[year])
  with open("figs/%s/%s/gender/delta.pkl" % (THE.version, THE.permitted), "wb") as f:
    cPkl.dump(delta_map, f)
  plot_topical_gender_delta()


def plot_topical_gender_delta():
  with open("figs/%s/%s/gender/delta.pkl" % (THE.version, THE.permitted)) as f:
    delta_map = cPkl.load(f)
  with open("figs/%s/%s/stats/heatmap_data.pkl" % (THE.version, THE.permitted)) as f:
    axis_data = cPkl.load(f)
    permitted_topics = axis_data['rows']
  topics = get_topics()
  x_axis = map(int, delta_map.keys())
  arr = []
  for year, score in delta_map.items():
    arr.append(score)
  arr = map(list, zip(*arr))
  legend_names = []
  i = 0
  for topic, score in zip(topics, arr):
    if topic not in permitted_topics:
      continue
    color = get_color(i)
    legend_names.append(topic)
    plt.plot(x_axis, score, color=color)
    i += 1
  plt.legend(legend_names, loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=5, fontsize=7, handlelength=0.7)
  plt.xlabel("Year")
  plt.ylabel("Delta b/w average male and female contribution")
  plt.savefig("figs/%s/%s/gender/delta.png" % (THE.version, THE.permitted))
  plt.clf()


def yearly_gender_topics(paper_range=None, file_name="yearly_topic_contribution"):
  def index_by_year(tups):
    y_comm = {}
    for tup in tups:
      comm = y_comm.get(tup[1], set())
      comm.add(tup[0])
      y_comm[tup[1]] = comm
    return y_comm

  def normalize(arr):
    arr_tot = sum(arr)
    if arr_tot == 0:
      return [0] * len(arr)
    return [arr_i / arr_tot for arr_i in arr]

  miner, graph, lda_model, vocab = get_graph_lda_data()
  p_venues = graph.get_papers_by_venue(permitted=THE.permitted)
  author_gender_map = get_author_genders()
  venue_topics = OrderedDict()
  for venue_id in sorted(mysqldb.get_venues().keys()):
    venue_id = str(venue_id)
    year_papers = index_by_year(p_venues[venue_id])
    year_topics = OrderedDict()
    for year in sorted(year_papers.keys(), key=lambda y: int(y)):
      both_genders_topics = []
      male_topics = []
      female_topics = []
      if (paper_range is not None) and (int(year) not in paper_range):
        continue
      papers = year_papers.get(year, [])
      if len(papers) > 0:
        for paper_id in papers:
          paper = graph.paper_nodes[paper_id]
          author_ids = paper.author_ids.strip().split(",")
          paper_topics = miner.documents[paper_id].topics_count
          # unit_paper_topics = [t / len(author_ids) for t in paper_topics]
          male_count, female_count = 0, 0
          for author_id in author_ids:
            gender = author_gender_map.get(author_id, None)
            if gender == 'm':
              male_count += 1
            elif gender == 'f':
              female_count += 1
          normalized_topics = normalize(paper_topics)
          if sum(normalized_topics) > 0:
            both_genders_topics.append(normalized_topics)
          if male_count > 0:
            male_topics.append(normalized_topics)
          if female_count > 0:
            female_topics.append(normalized_topics)
      year_topics[year] = {
          "all": both_genders_topics,
          "male": male_topics,
          "female": female_topics
      }
    venue_topics[venue_id] = year_topics
  save_file = "figs/%s/%s/gender/%s.pkl" % (THE.version, THE.permitted, file_name)
  with open(save_file, "wb") as f:
    cPkl.dump(venue_topics, f, cPkl.HIGHEST_PROTOCOL)


def yearly_compare_gender_topics(years, source="topic_contribution", target="stat"):
  gender_file = "figs/%s/%s/gender/%s.pkl" % (THE.version, THE.permitted, source)
  with open(gender_file) as f:
    venue_topics = cPkl.load(f)
  stat_map = {}
  print("# Years : %s" % years)
  for venue_id, year_topics in venue_topics.items():
    both, male, female = [], [], []
    for year in years:
      year_str = str(year)
      if year_str not in year_topics: continue
      topic_map = year_topics[year_str]
      both += topic_map["all"]
      male += topic_map["male"]
      female += topic_map["female"]
    if len(both) == 0: continue
    both = np.transpose(both)
    male = np.transpose(male)
    female = np.transpose(female)
    stat_topic_map = {}
    for i in range(get_n_topics()):
      print("Venue %s, Topic %d" % (venue_id, i))
      both_i = both[i, ]
      male_i = male[i, ]
      female_i = female[i, ]
      a_v_m = bootstrap(both_i, male_i)
      a_v_f = bootstrap(both_i, female_i)
      m_v_f = bootstrap(female_i, male_i)
      stat_topic_map[i] = (a_v_m, a_v_f, m_v_f)
    stat_map[venue_id] = stat_topic_map
  stat_file = "figs/%s/%s/gender/%s.pkl" % (THE.version, THE.permitted, target)
  with open(stat_file, "wb") as f:
    cPkl.dump(stat_map, f, cPkl.HIGHEST_PROTOCOL)


# def print_gender_topics(file_name):
#   stat_file = "figs/%s/%s/gender/%s.pkl" % (THE.version, THE.permitted, file_name)
#   with open(stat_file) as f:
#     stat_map = cPkl.load(f)
#   # print(stat_map.keys())
#   # for venue_id in sorted(stat_map.keys(), lambda k: int(k)):
#   venues = mysqldb.get_venues()
#   venue_names = {shorter_names(venues[v_id].acronym): v_id for v_id in sorted(venues.keys(), key=lambda k: int(k))}
#   with open("figs/%s/%s/stats/heatmap_data.pkl" % (THE.version, THE.permitted)) as f:
#     axis_data = cPkl.load(f)
#     permitted_topics = axis_data['rows']
#     permitted_venues = axis_data['columns']
#     valid_data = axis_data['data']
#   print("," + ",".join(permitted_venues))
#   topics = get_topics()
#   for topic_name in permitted_topics:
#     topic_id = topics.index(topic_name)
#     arr = [topic_name]
#     for venue in permitted_venues:
#       venue_id = venue_names[venue]
#       if valid_data[venue][topic_name] > 0:
#         m_v_f = 'T' if stat_map[venue_id][topic_id][2] else 'F'
#       else:
#         m_v_f = ''
#       # print("Venue: %s, Topic: %d, Stat: %s" % (venue_id, topic, m_v_f))
#       arr.append(m_v_f)
#     print(",".join(arr))


def _main():
  reporter()
  diversity("heatmap_09_16", range(2009, 2017), save_labels=True)
  # diversity("heatmap_01_08", range(2001, 2009))
  # diversity("heatmap_93_00", range(1992, 2001))
  topic_evolution(venue="all")
  # topic_evolution(venue="conferences")
  # topic_evolution(venue="journals")
  # gender_over_time()
  # gender_topics(range(2009, 2017), "topic_contribution_09_17")
  # compare_gender_topics("topic_contribution_09_17", "stat_09_17")
  # print_gender_topics('stat_09_17')
  # topical_gender_diff(range(1992, 2017))
  # get_top_papers(year_from=2009)
  # get_top_papers()
  # stat_venue_type_vs_cites_per_year()
  # stat_cites_per_year()
  # percentile_lines_per_year()
  # test_dendo_heatmap(34, range(1992, 2016), True)
  # yearly_gender_topics(range(2009, 2017), "yearly_topic_contribution_09_16")
  # yearly_compare_gender_topics(range(2009, 2011), "yearly_topic_contribution_09_16", "yearly_stat_09_10")
  # yearly_compare_gender_topics(range(2011, 2013), "yearly_topic_contribution_09_16", "yearly_stat_11_12")
  # yearly_compare_gender_topics(range(2013, 2015), "yearly_topic_contribution_09_16", "yearly_stat_13_14")
  # yearly_compare_gender_topics(range(2015, 2017), "yearly_topic_contribution_09_16", "yearly_stat_15_16")
  # print_gender_topics("yearly_stat_09_10")
  # print_gender_topics("yearly_stat_11_12")
  # print_gender_topics("yearly_stat_13_14")
  # print_gender_topics("yearly_stat_15_16")


if __name__ == "__main__":
  _main()
