from __future__ import print_function, division
import os, sys
sys.path.append(os.path.abspath("."))
from network.mine import cite_graph, Miner
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

__author__ = "panzer"

TOPIC_THRESHOLD = 3
def super_author():
  graph = cite_graph()
  miner = Miner(graph)
  lda_model, vocab = miner.lda(22, n_iter=100, alpha=0.847433736937, beta=0.763774618977)
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



super_author()