from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

from utils.lib import O, shuffle
import cPickle as pkl
import numpy as np
from network.graph import Graph
from collections import OrderedDict

__author__ = "bigfatnoob"

GRAPH_CSV = "data/citemap_v8.csv"

THE = O()
THE.permitted = 'all'


def harmonic_dist(n):
  dist = [1 / i for i in range(1, n + 1)]
  total = sum(dist)
  return [d / total for d in dist]


def uniform_dist(n):
  return [1 / n] * n


def is_not_none(s):
  return s and s != 'None'


def cite_graph(file_name):
  return Graph.from_file(file_name)


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


def all_authors(graph, min_year=None):
  authors = graph.get_papers_by_authors(THE.permitted)
  author_cites = []
  for author_id, papers in authors.items():
    cite_count = 0
    paper_count = 0
    for paper_id, year, __ in papers:
      if min_year is not None and int(year) < min_year: continue
      cited = graph.paper_nodes[paper_id].cited_count
      paper_count += 1
      if is_not_none(cited):
        cite_count += int(cited)
    author_cites.append((author_id, paper_count, cite_count))
  # tops = sorted(author_cites, key=lambda x: x[1], reverse=True)[:int(top_percent * len(author_cites))]
  return author_cites


def most_cited_authors(graph, top_percent=None, min_year=None):
  authors = all_authors(graph, min_year)
  if top_percent is None:
    top_percent = 1
  authors = sorted(authors, key=lambda x: x[2], reverse=True)[:int(top_percent * len(authors))]
  return [a[0] for a in authors]


def most_published_authors(graph, top_percent=None, min_year=None):
  authors = all_authors(graph, min_year)
  if top_percent is None:
    top_percent = 1
  authors = sorted(authors, key=lambda x: x[1], reverse=True)[:int(top_percent * len(authors))]
  return [a[0] for a in authors]


def most_contributed_authors(graph, distribution, top_percent=0.01, min_year=None):
  author_contributions = OrderedDict()
  for paper_id, paper in graph.get_paper_nodes(THE.permitted).items():
    if min_year is not None and int(paper.year) < min_year: continue
    cites = int(paper.cited_count) + 1 if is_not_none(paper.cited_count) else 1
    if not paper.author_ids: continue
    authors = paper.author_ids.split(",")
    dist = distribution(len(authors))
    for d, author in zip(dist, authors):
      author_contributions[author] = author_contributions.get(author, []) + [d * cites]
  author_list = []
  for author, contribution in author_contributions.items():
    author_list.append((author, sum(contribution)))
  tops = sorted(author_list, key=lambda x: x[1], reverse=True)[:int(top_percent * len(all_authors(graph, min_year)))]
  return [t[0] for t in tops]


def make_author_dict(author_tuples):
  names_to_ids, ids_to_names = OrderedDict(), OrderedDict()
  for index, tup in enumerate(author_tuples):
    names_to_ids[tup] = index
    ids_to_names[index] = tup
  n = len(author_tuples)
  author_scores = np.full(n, 1 / n, dtype=np.float64)
  return names_to_ids, ids_to_names, author_scores


def list_out(lst):
  for i in range(len(lst) - 1):
    yield lst[i], set(lst[:i] + lst[i + 1:])


def make_coauthor_map(graph, top_authors, names_to_ids, min_year):
  coauthor_count = {i: set() for i in range(len(top_authors))}
  author_cite_count = {i: 0 for i in range(len(top_authors))}
  for paper_id, paper in graph.paper_nodes.items():
    if min_year and int(paper.year) < min_year: continue
    authors = paper.author_ids.split(",")
    cites = int(paper.cited_count) if paper.cited_count and paper.cited_count != 'None' else 0
    if len(authors) <= 1: continue
    for author_name, co_authors_name in list_out(authors):
      if author_name not in top_authors: continue
      author_index = names_to_ids[author_name]
      co_authors_index = set([names_to_ids[co_author_name]
                              for co_author_name in co_authors_name if co_author_name in top_authors])
      coauthor_count[author_index] = coauthor_count[author_index].union(co_authors_index)
      author_cite_count[author_index] += cites
  coauthor_count = {i: np.array(sorted(co_authors), dtype=np.int32) for i, co_authors in coauthor_count.items()}
  total_cites = sum(author_cite_count.values())
  author_cite_count = {i: author_cite_count[i] / total_cites for i in author_cite_count.keys()}
  return coauthor_count, author_cite_count


def page_rank(graph, d=0.45, top_percent=0.01, min_year=None, iterations=1000):
  factor = 50
  top_tups = most_cited_authors(graph, top_percent=factor * top_percent, min_year=min_year)
  names_to_ids, ids_to_names, page_rank_scores = make_author_dict(top_tups)
  coauthor_count, author_cite_count = make_coauthor_map(graph, top_tups, names_to_ids, min_year)
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
  top_author_indices = np.argsort(page_rank_scores)[::-1][:int(len(page_rank_scores) / factor)]
  top_author_names = [ids_to_names[index] for index in top_author_indices]
  return top_author_names


def __main():
  graph = retrieve_graph()
  cited = set(most_cited_authors(graph, top_percent=0.01, min_year=2009))
  published = set(most_published_authors(graph, top_percent=0.01, min_year=2009))
  harmonic = set(most_contributed_authors(graph, harmonic_dist, top_percent=0.01, min_year=2009))
  uniform = set(most_contributed_authors(graph, uniform_dist, top_percent=0.01, min_year=2009))
  pr = set(page_rank(graph, top_percent=0.01, min_year=2009))
  print("Cited: ", len(cited))
  print("Published: ", len(published))
  print("Harmonic: ", len(harmonic))
  print("Uniform: ", len(uniform))
  print("Page Rank: ", len(pr))
  metrics = {
    'cite' : cited,
    'publ' : published,
    'harm': harmonic,
    'unif': uniform,
    'page': pr
  }
  keys = sorted(metrics.keys())
  for key in keys:
    print("#", key)
    for key1 in keys:
      print("\t ##", key1, len(metrics[key].intersection(metrics[key1])))


if __name__ == "__main__":
  __main()
