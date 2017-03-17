from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

from utils.lib import O
import numpy as np
from collections import OrderedDict
from network.mine import cite_graph, Miner
import matplotlib.pyplot as plt
from db import mysql
import pandas as pd
from sklearn.feature_extraction import text

GRAPH_CSV = "data/citemap_v6.csv"

# For 11 TOPICS
N_TOPICS = 7
ALPHA = 0.22359
BETA = 0.53915
ITERATIONS = 100
# TOPICS = ["TPC %d" % d for d in range(N_TOPICS)]
# TOPICS = ["Design", "Testing", "Modelling", "Mobile", "Energy", "Defects",
#           "SourceCode", "WebApps", "Configuration", "Developer", "Mining"]
TOPICS = ["Modelling", "Empirical", "Requirements", "Theory", "Web", "Testing", "Applications"]
TOPIC_THRESHOLD = 3
STOP_WORDS = text.ENGLISH_STOP_WORDS.union(['software', 'engineering', 'paper', 'study', 'based',
                                            'results', 'approach', 'case', 'workshop', 'international', 'research',
                                            'conference', 'introduction', 'editors', 'article', 'issue', 'month',
                                            'copyright', 'special', 'used', 'using', 'use', 'studies'])

# Config
THE = O()
THE.permitted = "journals"


def get_graph_lda_data():
  graph = cite_graph(GRAPH_CSV)
  miner = Miner(graph, THE.permitted)
  lda_model, vocab = miner.lda(N_TOPICS, n_iter=ITERATIONS, alpha=ALPHA, beta=BETA, stop_words=STOP_WORDS)
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


def report(lda_model, vocab, n_top_words=10):
  for index, topic_dist in enumerate(lda_model.topic_word_):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
    print('Topic {}: {}'.format(index, ', '.join(topic_words)))


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
  print("DIVERSITY for %s" % THE.permitted)
  miner, graph, lda_model, vocab = get_graph_lda_data()
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
  row_labels = [str(ind) + "-" + name for ind, name in zip(range(lda_model.n_topics), TOPICS)]
  # row_labels = ["%2d" % ind for ind in range(lda_model.n_topics)]
  column_labels = [shorter_names(venue.acronym) for c, venue in venues.items() if venue.id in valid_conferences]
  # Heatmap
  heatmap_arr = []
  for conference_id in sorted(venue_heatmaps.keys(), key=lambda x: int(x)):
    tot = sum(venue_heatmaps[conference_id])
    dist = [top / tot for top in venue_heatmaps[conference_id]]
    heatmap_arr.append(dist)
  report(lda_model, vocab, 15)
  # make_dendo_heatmap(np.transpose(heatmap_arr), row_labels, column_labels,
  #                    "figs/v2/diversity/%s_dend.png" % fig_name, dend_settings)
  make_heatmap(np.transpose(heatmap_arr), row_labels, column_labels,
               "figs/v3/%s/diversity/%s.png" % (THE.permitted, fig_name))


def _main():
  paper_bar()
  diversity("heatmap_09_16", range(2009, 2017))
  diversity("heatmap_01_08", range(2001, 2009))
  diversity("heatmap_93_00", range(1993, 2000))


if __name__ == "__main__":
  _main()
