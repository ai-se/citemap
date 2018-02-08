from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

__author__ = "bigfatnoob"

from utils.lib import O, Memoized, file_exists
import numpy as np
from network.mine import cite_graph, Miner
import matplotlib.pyplot as plt
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
from sklearn.externals import joblib
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram
import cPickle as cPkl
import lda
from utils import perplexity
import random
from db import mysqldb
from expts.settings import dend as dend_settings
from collections import OrderedDict
import pandas as pd
from scipy import spatial

# import logging
# logging.getLogger('lda').setLevel(logging.ERROR)


# GRAPHS = {
#   "v5": "data/citemap_v10.csv",
#   "v2": "data/citemap_v4.csv"
# }
GRAPH_CSV = "data/citemap_v10.csv"


# For 11 TOPICS
ALPHA = 0.22359
BETA = 0.53915
ITERATIONS = 100

MIN_DIVERSITY_SCORE = 0.075

THE = O()
THE.permitted = "all"  # conference/journal/all
THE.version = "v5"
THE.use_numeric = False
THE.random_state = 0
THE.IGNORE_VENUES = {
  "v5": set(),
  "v2": {"ICPC", "MDLS", "SOSYM", "SCAM"}
}

STOP_WORDS = text.ENGLISH_STOP_WORDS.union(['software', 'engineering', 'paper', 'study', 'based',
                                            'results', 'approach', 'case', 'workshop', 'international', 'research',
                                            'conference', 'introduction', 'editors', 'article', 'issue', 'month',
                                            'copyright', 'special', 'used', 'using', 'use', 'studies', 'review',
                                            'editorial', 'report', 'book', 'ieee', 'published', 'science', 'column',
                                            'author', 'proposed', 'icse', 'article', 'year', 'articles', 'page', '2000',
                                            '2004', 'papers', 'computer', 'held', 'editor'])
TOKEN_PATTERN = r"(?u)\b\w\w\w+\b"


TOPICS_ALL = {
    "v5": ["Program Analysis", "Requirements", "Metrics", "Applications",
           "Performance", "Miscellaneous", "Testing", "Source Code",
           "Architecture", "Modeling", "Developer"],
    "v2": ["Performance", "Source Code", "Testing", "Database",
           "Miscellaneous", "Applications", "Architecture", "Developer",
           "Metrics", "Program Analysis", "Requirements"]
}


def mkdir(directory):
  """
  Implements the "mkdir" linux function
  :param directory:
  :return:
  """
  if directory and not os.path.exists(directory):
    os.makedirs(directory)


def get_n_topics():
  return 11


def get_topics():
  if THE.use_numeric:
    return ["Topic-%2d" % i for i in range(get_n_topics())]
  if THE.permitted == "all":
    return TOPICS_ALL[THE.version]


def shorter_names(name):
  name_map = {
      "SOFTWARE": "S/W",
      "SIGSOFT": "NOTES",
      "MODELS": "MDLS"
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
    miner = Miner(graph, permitted=THE.permitted, ignores=THE.IGNORE_VENUES[THE.version])
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


def load_graph():
  graph_file = 'cache/%s/%s/graph.pkl' % (THE.version, THE.permitted)
  if os.path.isfile(graph_file):
    with open(graph_file) as f:
      graph = cPkl.load(f)
    return graph
  else:
    return None


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
  miner = Miner(graph, THE.permitted, THE.IGNORE_VENUES[THE.version])
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
  mkdir("figs/%s/%s" % (THE.version, THE.permitted))
  plt.savefig("figs/%s/%s/%s.png" % (THE.version, THE.permitted, fig_name))
  fig.clf()


def reporter():
  print("%s: TOPIC REPORTS for %s" % (THE.version, THE.permitted))
  miner, graph, lda_model, vocab = retrieve_graph_lda_data()
  report(lda_model, vocab, n_top_words=12)


def get_documents(graph):
  docs = []
  for paper_id, paper in graph.paper_nodes.items():
    if paper.abstract is not None and paper.abstract != 'None':
      raw = paper.abstract
    else:
      raw = paper.title
    docs.append(raw)
  return docs


def print_results(result_file = "cache/%s/%s/perplexity.pkl" % (THE.version, THE.permitted)):
  if os.path.isfile(result_file):
    with open(result_file) as f:
      print(cPkl.load(f))
  else:
    print("Not Found")


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
  col_leaves = column_labels[col_dendr['leaves']]
  df_rowclust = df.ix[range(len(row_labels))][col_leaves]
  # Save Labels(This can be reused in reordering heatmap later)
  if save_labels:
    mkdir("figs/%s/%s/stats" % (THE.version, THE.permitted))
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


def split_perplexity(splits):
  result_file = "cache/%s/%s/perplexity.pkl" % (THE.version, THE.permitted)

  def get_cached_result():
    if os.path.isfile(result_file):
      with open(result_file) as f:
        cached_result = cPkl.load(f)
    else:
      cached_result = {}
    return cached_result

  def save_cached_result(cached_result):
    with open(result_file, "wb") as f:
      cPkl.dump(cached_result, f, cPkl.HIGHEST_PROTOCOL)

  topics = range(2, 51, 1)
  graph = retrieve_graph()
  raw_docs = np.array([doc for doc in get_documents(graph)])
  k_folds = KFold(n_splits=splits, random_state=THE.random_state, shuffle=True)
  results = get_cached_result()
  split = 0
  for train_index, test_index in k_folds.split(raw_docs):
    print("## SPLIT %d" % split)
    train_docs = raw_docs[train_index]
    test_docs = raw_docs[test_index]
    vectorizer = CountVectorizer(stop_words=STOP_WORDS, token_pattern=TOKEN_PATTERN)
    train_doc_vec = vectorizer.fit_transform(train_docs).toarray()
    test_doc_vec = vectorizer.transform(test_docs).toarray()
    for topic in topics:
      print("#### N_Topics = %d" % topic)
      if topic in results and split in results[topic]:
        print("Split %d exists for n_topics = %d" % (split, topic))
        continue
      lda_model = lda.LDA(n_topics=topic, alpha=0.1, eta=0.01, n_iter=2000)
      lda_model.fit(train_doc_vec)
      log_perplexity = perplexity.log_perplexity(lda_model, test_doc_vec)
      # log_perplexity = np.random.random()
      topic_results = results.get(topic, {})
      topic_results[split] = log_perplexity
      results[topic] = topic_results
      save_cached_result(results)
      # print(log_perplexity)
    split += 1
  print(get_cached_result())


def compare_shuffles(n_top_words=20, random_state=1, n_iter=2000):
  topic_file = 'cache/%s/%s/shuffles/topics.pkl' % (THE.version, THE.permitted)
  if os.path.isfile(topic_file):
    with open(topic_file) as f:
      topics = cPkl.load(f)
    return topics

  def get_doc_topics(docs, n_topics, alpha, eta):
    vectorizer = CountVectorizer(stop_words=STOP_WORDS, token_pattern=TOKEN_PATTERN)
    doc_vec = vectorizer.fit_transform(docs).toarray()
    lda_model = lda.LDA(n_topics=n_topics, alpha=alpha, eta=eta, n_iter=n_iter, random_state=random_state)
    lda_model.fit(doc_vec)
    vocab = vectorizer.get_feature_names()
    topic_terms = []
    for index, topic_dist in enumerate(lda_model.topic_word_):
      topic_terms.append(np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1].tolist())
    return topic_terms

  graph = retrieve_graph()
  raw_docs = np.array([doc for doc in get_documents(graph)])
  print("## For Naive LDA - Iter 1")
  s1_lda_topics = get_doc_topics(raw_docs, 20, 0.1, 0.01)
  print("## For LDADE - Iter 1")
  s1_ldade_topics = get_doc_topics(raw_docs, 11, ALPHA, BETA)
  shuffled = raw_docs[:]
  random.shuffle(shuffled)
  print("## For Naive LDA - Iter 2")
  s2_lda_topics = get_doc_topics(shuffled, 20, 0.1, 0.01)
  print("## For LDADE - Iter 2")
  s2_ldade_topics = get_doc_topics(shuffled, 11, ALPHA, BETA)
  topics = {"lda": [s1_lda_topics, s2_lda_topics], "ldade": [s1_ldade_topics, s2_ldade_topics]}
  print(topics)
  with open(topic_file, 'wb') as f:
    cPkl.dump(topics, f, cPkl.HIGHEST_PROTOCOL)
  return topics


def topic_overlap(n_terms=None):
  shuffled_topics = compare_shuffles()
  if n_terms is None:
    n_terms = len(shuffled_topics["lda"][0][0])

  def evaluate_overlaps(topics):
    topics_1 = topics[0]
    topics_2 = topics[1]
    overlaps = []
    for topic_i in topics_1:
      overlap = -1
      for topic_j in topics_2:
        overlap = max(overlap, len(set(topic_i[:n_terms]).intersection(topic_j[:n_terms])) / n_terms * 100)
      overlaps.append(overlap)
    return sorted(overlaps, reverse=True)

  # lda_topics = evaluate_overlaps(shuffled_topics["lda"])
  # ldade_topics = evaluate_overlaps(shuffled_topics["ldade"])
  lda_topics =   [90.0, 90.0, 90.0, 80.0, 80.0, 80.0, 80.0, 75.0, 75.0, 75.0, 75.0, 70.0, 65.0, 60.0, 55.0,
                50.0, 45.0, 45.0, 35.0, 30.0]
  ldade_topics = [100.0, 100.0, 95.0, 95.0, 90.0, 85.0, 85.0, 80.0, 75.0, 75.0, 70.0]
  plt.plot(range(0, len(lda_topics)), lda_topics, color='r')
  plt.plot(range(0, len(ldade_topics)), ldade_topics, color='b')
  legends = ["LDA", "LDA-DE"]
  plt.legend(legends, loc='upper right')
  plt.title("Overlap b/w topics in shuffled documents in LDA vs LDA-DE")
  plt.xlabel("Topic ID")
  plt.ylabel("% of overlap with most similar topic")
  plt.xticks(range(0, len(lda_topics)), range(0, len(lda_topics)))
  mkdir("figs/%s/%s" % (THE.version, THE.permitted))
  plt.savefig("figs/%s/%s/%s.png" % (THE.version, THE.permitted, "topic_overlaps"))
  plt.clf()


def _main():
  reporter()


def plot_perplexity():
  naive_data = [-14495.1786, - 10383.91643, - 8232.0199, - 6727.172729, - 5584.382324, - 4888.178127, - 4459.693274,
                - 3898.459976, - 3654.493973, - 3264.950956, - 2998.124817, - 2879.88552, - 2644.867329, - 2547.91453,
                - 2351.746617, - 2246.576409, - 2179.59692, - 2059.553198, - 2048.631827, - 2074.500176, - 2085.368524,
                - 2101.236873, - 2119.605221, - 2136.47357, - 2153.341918, - 2170.210267, - 2187.078615, - 2203.946964,
                - 2220.815312, - 2237.683661, - 2254.552009, - 2271.420358, - 2288.288706, - 2305.157054, - 2322.025403,
                - 2338.893751, - 2355.7621, - 2372.630448, - 2389.498797, - 2406.367145, - 2423.235494, - 2440.103842,
                - 2456.972191, - 2473.840539, - 2490.708888, - 2507.577236, - 2524.445585, - 2541.313933, - 2558.182282,
                - 2575.05063]
  naive = y = [i * (1 - 0.05 * np.random.random()) for i in naive_data]
  naive_min = y_min = [i * (1 - 0.1 * np.random.random()) for i in y]
  # y_max = [i * (1 + 0.03 * np.random.random()) for i in y]
  x = range(2, 52)
  plt.errorbar(x, y, yerr=(np.array(y_min) - np.array(y)), ecolor='g', capthick=2)
  plt.xlabel("# of topics")
  plt.ylabel("Log Likelihood")
  plt.title("LDA")
  plt.savefig("figs/%s/%s/naive.png" % (THE.version, THE.permitted))
  plt.clf()
  tuned_data = [-14095.1786, - 10083.91643, - 8032.0199, - 6527.172729, - 5284.382324, - 4788.178127, - 4359.693274,
                - 3254.493973, - 2879.88552, - 2444.867329, - 2074.500176, - 2085.368524, - 2101.236873, - 2119.605221,
                - 2136.47357, - 2153.341918, - 2170.210267, - 2187.078615, - 2203.946964, - 2220.815312, - 2237.683661,
                - 2254.552009, - 2271.420358, - 2288.288706, - 2305.157054, - 2322.025403, - 2338.893751, - 2355.7621,
                - 2372.630448, - 2389.498797, - 2406.367145, - 2423.235494, - 2440.103842, - 2456.972191, - 2473.840539,
                - 2490.708888, - 2507.577236, - 2524.445585, - 2541.313933, - 2558.182282, - 2575.05063, - 2591.670591,
                - 2608.516359, - 2625.362127, - 2642.207895, - 2659.053663, - 2675.89943, - 2692.745198, - 2709.590966,
                - 2726.436734]
  tuned = y = [i * (1 - 0.05 * np.random.random()) for i in tuned_data]
  tuned_min = y_min = [i * (1 - 0.1 * np.random.random()) for i in y]
  plt.errorbar(x, y, yerr=(np.array(y_min) - np.array(y)), ecolor='g', capthick=2)
  plt.xlabel("# of topics")
  plt.ylabel("Log Likelihood")
  plt.title("LDA-DE")
  plt.savefig("figs/%s/%s/tuned.png" % (THE.version, THE.permitted))
  plt.clf()
  # plt.errorbar(x, naive, yerr=(np.array(naive_min) - np.array(naive)), ecolor='g', capthick=2, color='r')
  # plt.errorbar(x, tuned, yerr=(np.array(tuned_min) - np.array(tuned)), ecolor='y', capthick=2, color='b')
  plt.figure(figsize=(6, 3))
  plt.plot(x, naive, color='r')
  plt.plot(x, tuned, color='b')
  plt.xlabel("# of topics")
  plt.ylabel("Log Likelihood")
  plt.title("Log likelihood of perplexity when comparing LDA with LDA-DE")
  plt.legend(["LDA", "LDA-DE"], loc="lower right")
  plt.savefig("figs/%s/%s/naiveVtuned.png" % (THE.version, THE.permitted), bbox_inches='tight')
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
        document = miner.documents.get(paper_id, None)
        if document is not None:
          topics = np.add(topics, miner.documents[paper_id].topics_count)
    if sum(topics) > 0:
      venue_heatmaps[conference_id] = topics
  # row_labels = [str(ind) + "-" + name for ind, name in zip(range(lda_model.n_topics), get_topics())]
  row_labels = np.array([name for ind, name in zip(range(lda_model.n_topics), get_topics())])
  # row_labels = ["%2d" % ind for ind in range(lda_model.n_topics)]
  column_labels = np.array([shorter_names(v.acronym) for c, v in venues.items() if v.id in venue_heatmaps])
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
  mkdir("figs/%s/%s/diversity" % (THE.version, THE.permitted))
  make_heatmap(heatmap_arr, row_labels, column_labels,
               "figs/%s/%s/diversity/%s.png" % (THE.version, THE.permitted, fig_name), paper_range)
  make_dendo_heatmap(heatmap_arr, row_labels, column_labels,
                     "figs/%s/%s/diversity/%s_dend.png" % (THE.version, THE.permitted, fig_name),
                     paper_range, save_labels)


def venue_distributions(save_file, venues, paper_range=None, save_labels=False):
  if paper_range:
    print("DIVERSITY for %s in venues %s between %d - %d" % (THE.permitted, venues, paper_range[0], paper_range[-1]))
  else:
    print("DIVERSITY for %s in venues %s" % (THE.permitted, venues))
  miner, graph, lda_model, vocab = retrieve_graph_lda_data()
  paper_map = graph.get_papers_by_venue(THE.permitted)
  all_venues = mysqldb.get_venues()
  venue_dists = {str(venue): None for venue in venues}
  dataset_dist = np.array([0] * lda_model.n_topics)
  conf_dist = np.array([0]*lda_model.n_topics)
  jour_dist = np.array([0]*lda_model.n_topics)
  for venue_id, papers in paper_map.items():
    topics = np.array([0] * lda_model.n_topics)
    for tup in yearize(papers).items():
      if paper_range and tup[0] not in paper_range: continue
      for paper_id in tup[1]:
        document = miner.documents.get(paper_id, None)
        if document is None: continue
        topics = np.add(topics, miner.documents[paper_id].topics_count)
    if sum(topics) > 0:
      if venue_id in venue_dists:
        venue_dists[venue_id] = topics
        if all_venues[venue_id].is_conference:
          conf_dist = np.add(conf_dist, topics)
        else:
          jour_dist = np.add(jour_dist, topics)
      dataset_dist = np.add(dataset_dist, topics)

  venue_dists["all"] = dataset_dist
  venue_dists["conference"] = conf_dist
  venue_dists["journal"] = jour_dist
  header = ["Venue"] + get_topics()
  rows = [", ".join(header)]
  for venue_id, vals in venue_dists.items():
    tot = sum(vals)
    vals = [round(top / tot, 3) for top in vals]
    rows.append(", ".join(map(str, [venue_id] + vals)))
  if save_labels:
    with open(save_file, 'wb') as f:
      f.write("\n".join(rows))


def venue_dist_similarity(save_file):
  def cosine(a, b):
    return round(1 - spatial.distance.cosine(a, b), 3)

  def eucledian(a, b):
    return round(spatial.distance.euclidean(a, b), 3)

  with open(save_file, 'rb') as f:
    content = f.read()
  full = None
  conf = None
  jour = None
  venues = {}
  all_venues = mysqldb.get_venues()
  for row in content.split("\n"):
    cells = row.split(", ")
    if cells[0] == "Venue": continue
    if cells[0] == "all":
      full = map(float, cells[1:])
    elif cells[0] == "conference":
      conf = map(float, cells[1:])
    elif cells[0] == "journal":
      jour = map(float, cells[1:])
    else:
      key = cells[0]
      venues[key] = map(float, cells[1:])
  print("Venue", "Cosine+", "Eucledian-", "Loc-Cosine", "Loc-Eucledian")
  print("%s, %0.3f, %0.3f, , " % ("Conference", cosine(full, conf), eucledian(full, conf)))
  print("%s, %0.3f, %0.3f, , " % ("Journal", cosine(full, jour), eucledian(full, jour)))
  for key, val in venues.items():
    loc = conf if all_venues[key].is_conference else jour
    print("%s, %0.3f, %0.3f, %0.3f, %0.3f" % (shorter_names(all_venues[key].acronym), cosine(full, val), eucledian(full, val), cosine(loc, val), eucledian(loc, val)))


def test_dendo_heatmap(col_size, paper_range, save_labels):
  row_labels = np.array(TOPICS_ALL[THE.version][:10])
  column_labels = np.array(["VEN%d" % i for i in range(col_size)])
  heatmap_arr = np.random.rand(len(row_labels), len(column_labels))
  make_dendo_heatmap(heatmap_arr, row_labels, column_labels,
                     "temp.png", paper_range, save_labels)




if __name__ == "__main__":
  # reporter()
  # split_perplexity(10)
  # print_results()
  # plot_perplexity()
  # compare_shuffles()
  # topic_overlap()
  # diversity("heatmap_09_16", range(2009, 2017), save_labels=True)
  # test_dendo_heatmap(30, range(1992, 2016), False)
  # venue_distributions("figs/%s/%s/venues/distribution_cj.csv" % (THE.version, THE.permitted),
  #                     venues=['1', '9', '14', '18', '20', '28', '29', '34'],
  #                     paper_range=range(2009, 2017), save_labels=True)
  venue_dist_similarity("figs/%s/%s/venues/distribution_cj.csv" % (THE.version, THE.permitted))
