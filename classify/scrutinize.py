from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True
from network.mine import Miner, cite_graph
from sklearn.feature_extraction import text
from utils.lib import O
from collections import OrderedDict

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
DELIMITER = '$|$'
STOP_WORDS = text.ENGLISH_STOP_WORDS.union(['software', 'engineering'])
TOKEN_PATTERN = r"(?u)\b\w\w\w+\b"
K_BEST_RATE = 0.2
IS_INDEPENDENT_CONFERENCE = True


def top_authors(graph):
  authors = graph.get_papers_by_authors()
  author_cites = []
  for author_id, papers in authors.items():
    cite_count = 0
    for paper_id, year, __ in papers:
      cited = graph.paper_nodes[paper_id].cited_counts
      if cited:
        cite_count += cited
    author_cites.append((author_id, cite_count, graph.author_nodes[author_id].name))
  tops = sorted(author_cites, key=lambda x: x[1], reverse=True)
  author_dict = OrderedDict()
  for a_id, cites, name in tops[:10]:
    author_dict[name] = (a_id, cites, name)
  return author_dict


def reputation():
  graph = cite_graph(GRAPH_CSV)
  miner = Miner(graph)
  # model, vocab = miner.lda(n_topics=N_TOPICS, n_iter=ITERATIONS,
  #                          random_state=RANDOM_STATE, alpha=ALPHA, beta=BETA)
  print(top_authors(graph))

if __name__ == "__main__":
  reputation()