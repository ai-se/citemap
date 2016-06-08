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

__author__ = "panzer"

COLORS = sorted(clrs.cnames.keys())
TOPIC_THRESHOLD = 3

def get_color(index): return str(COLORS[index])

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
  lda_model, vocab = miner.lda(22, n_iter=100, alpha=0.847433736937, beta=0.763774618977)
  conferences = graph.get_papers_by_conference()
  conference_topics = {}
  for conference_id, papers in conferences.items():
    topics = np.array([0] * lda_model.n_topics)
    for tup in yearize(papers).items():
      for paper_id in tup[1]:
        topics = np.add(topics, miner.documents[paper_id].topics_count)
    conference_topics[conference_id] = percent_sort(topics)
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
  plt.yticks(np.arange(0,90,10))
  #Legends
  patches = []
  for topic, color in colors_dict.items():
    patches.append(mpatches.Patch(color=color, label = 'Topic %s'%str(topic)))
  plt.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(patches), fontsize=7)
  plt.savefig("figs/conference_diversity.png")
  plt.clf()








def conference_evolution():
  graph = cite_graph()
  miner = Miner(graph)
  lda_model, vocab = miner.lda(22, n_iter=100, alpha=0.847433736937, beta=0.763774618977)
  conferences = graph.get_papers_by_conference()
  conference_topics = {}
  for conference_id, papers in conferences.items():
    topics = []
    for tup in yearize(papers).items():
      year = tup[0]
      year_topics = np.array([0]*lda_model.n_topics)
      for paper_id in tup[1]:
        topic_count = miner.documents[paper_id].topics_count
        year_topics = np.add(year_topics, topic_count)
      topics.append((year, percent_sort(year_topics)))
    conference_topics[conference_id] = topics
  print(conference_topics['1'][-1])
  print(conference_topics['1'][0])



conference_diversity()