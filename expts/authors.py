from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True
from utils.lib import O, Memoized, file_exists
import numpy as np
from collections import OrderedDict
from network.graph import Graph
import pandas as pd
from sklearn import preprocessing
import cPickle as cPkl
import db.mysqldb as mysql
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn.apionly as sns

# mpl.rc('axes', grid = False)
# Set backgound color to white
# mpl.rc('axes', facecolor = 'white')

__author__ = "bigfatnoob"


GRAPH_CSV = "data/citemap_v10.csv"

THE = O()
THE.permitted = 'all'
THE.version = 'v4'

@Memoized
def cite_graph(file_name):
  return Graph.from_file(file_name)


def is_not_none(s):
  return s and s != 'None'


def open_pkl(pkl_file):
  with open(pkl_file) as f:
    return cPkl.load(f)


def all_authors(graph, min_year=None):
  """
  :param graph:
  :param min_year:
  :return: [(str(author_id), paper_count, cite_count)]
  """
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
  """
  [(str(author_id), paper_count, cite_count)]
  :param graph:
  :param top_percent:
  :param min_year:
  :return:
  """
  authors = all_authors(graph, min_year)
  if top_percent is None:
    top_percent = 1
  return [a for a in sorted(authors, key=lambda x: x[2], reverse=True)[:int(top_percent * len(authors))]]


def link_matrix(graph, valid_authors, min_year=None):
  links_file = "figs/%s/%s/authors/links.pkl" % (THE.version, THE.permitted)
  collaborators_file = "figs/%s/%s/authors/collaborators.pkl" % (THE.version, THE.permitted)
  if file_exists(links_file) and file_exists(collaborators_file):
    with open(collaborators_file) as f:
      collaborators = cPkl.load(f)
    with open(links_file) as f:
      data = cPkl.load(f)
      if data.columns.values.tolist() == valid_authors:
        return data, collaborators
  papers = graph.get_paper_nodes(permitted=THE.permitted)
  valid_authors_set = set(valid_authors)
  matrix = pd.DataFrame(index=valid_authors, columns=valid_authors).fillna(0)
  collaborators = {a: set() for a in valid_authors}
  for paper_id, paper in papers.items():
    if min_year is not None and int(paper.year) < min_year:
      continue
    author_ids = paper.author_ids.strip().split(",")
    for i in range(len(author_ids) - 1):
      if author_ids[i] not in valid_authors_set:
        continue
      for j in range(i + 1, len(author_ids)):
        if author_ids[j] not in valid_authors_set:
          continue
        # Setting as 1 rather than forced increment to
        # avoid over collaboration
        matrix.ix[author_ids[i], author_ids[j]] += 1
        matrix.ix[author_ids[j], author_ids[i]] += 1
        i_collaborators = collaborators[author_ids[i]]
        i_collaborators.add(author_ids[j])
        collaborators[author_ids[i]] = i_collaborators
        j_collaborators = collaborators[author_ids[j]]
        j_collaborators.add(author_ids[i])
        collaborators[author_ids[j]] = j_collaborators
  with open(links_file, "wb") as f:
    cPkl.dump(matrix, f)
  with open(collaborators_file, "wb") as f:
    cPkl.dump(collaborators, f)
  return matrix, collaborators


def max_normalize(d):
  max_val = max(d.values())
  return {d_i: d[d_i] / max_val for d_i in d.keys()}


def page_rank(authors, links, author_collaborators, damp=0.5, iterations=100000,
              file_name="page_rank", prefix=None):
  init = 1 / len(authors)
  pr = {a: init for a in authors}
  for i in xrange(iterations):
    if i % 100 == 0:
      print("Iteration : %d" % i)
    recomputed_pr = {}
    for a in pr.keys():
      collaborators = author_collaborators[a]
      link_score = 0
      for collaborator in collaborators:
        # link_score += pr[collaborator] / sum(links.ix[:, collaborator] > 0)
        link_score += pr[collaborator] / len(author_collaborators[collaborator])
      link_score *= damp
      self_score = (1 - damp) * init
      recomputed_pr[a] = self_score + link_score
    pr = max_normalize(recomputed_pr)
  if prefix is None:
    f_name = "figs/%s/%s/authors/%s.pkl" % (THE.version, THE.permitted, file_name)
  else:
    f_name = "figs/%s/%s/authors/%s/%s.pkl" % (THE.version, THE.permitted, prefix, file_name)
  with open(f_name, "wb") as f:
    cPkl.dump(pr, f, cPkl.HIGHEST_PROTOCOL)
  return pr


def df_column_normalize(df):
  x = df.values  # returns a numpy array
  min_max_scaler = preprocessing.MinMaxScaler()
  x_scaled = min_max_scaler.fit_transform(x)
  return pd.DataFrame(x_scaled, index=df.index.values, columns=df.columns.values)


def weighted_page_rank(authors, links, weights, author_collaborators, damp=0.5, iterations=100000,
                       file_name="cite_page_rank", prefix=None, do_print=False):
  init = 1 / len(authors)
  pr = {a: init for a in authors}
  links = df_column_normalize(links).sum(axis=1)
  links = {a: links[a] for a in pr.keys()}
  total_weights = sum(weights.values())
  weights = {a_id: w / total_weights for a_id, w in weights.items()}
  for i in xrange(iterations):
    if i % 100 == 0:
      print("Iteration : %d" % i)
    recomputed_pr = {}
    for a in pr.keys():
      collaborators = author_collaborators[a]
      link_score = 0
      for collaborator in collaborators:
        # link_score += pr[collaborator] / sum(links.ix[:, collaborator] > 0)
        link_score += pr[collaborator] / links[collaborator]
      link_score *= damp
      self_score = (1 - damp) * weights[a]
      recomputed_pr[a] = self_score + link_score
    pr = max_normalize(recomputed_pr)
  if prefix is None:
    f_name = "figs/%s/%s/authors/%s.pkl" % (THE.version, THE.permitted, file_name)
  else:
    f_name = "figs/%s/%s/authors/%s/%s.pkl" % (THE.version, THE.permitted, prefix, file_name)
  with open(f_name, "wb") as f:
    cPkl.dump(pr, f, cPkl.HIGHEST_PROTOCOL)
  return pr


def run_page_rank(min_year, damp, top=0.01, iterations=100000, use_prefix=False, f_name=None):
  graph = cite_graph(GRAPH_CSV)
  authors = [a[0] for a in most_cited_authors(graph, top, min_year)]
  links, author_collaborators = link_matrix(graph, authors, min_year)
  if f_name:
    file_name = f_name
  elif use_prefix:
    file_name = "%s/page_rank_%0.2f" % ("naive", damp)
  else:
    file_name = "%s_page_rank_%0.2f" % ("naive", damp)
  page_rank(authors, links, author_collaborators, damp, iterations=iterations,
            file_name=file_name)


def run_weighted_page_rank(min_year, damp, top=0.01, iterations=100000, weight_param="cite", use_prefix=False, f_name=None):
  graph = cite_graph(GRAPH_CSV)
  authors, weights = [], OrderedDict()
  for a in most_cited_authors(graph, top, min_year):
    authors.append(a[0])
    if weight_param == "cite":
      weights[a[0]] = a[2]
    elif weight_param == "publ":
      weights[a[0]] = a[1]
    else:
      print ("Invalid weighing scheme : %s " % weight_param)
      exit()
  links, author_collaborators = link_matrix(graph, authors, min_year)
  if f_name:
    file_name = f_name
  elif use_prefix:
    file_name = "%s/page_rank_%0.2f" % (weight_param, damp)
  else:
    file_name = "%s_page_rank_%0.2f" % (weight_param, damp)
  weighted_page_rank(authors, links, weights, author_collaborators, damp,
                     iterations=iterations, file_name=file_name)


def plot_damp_top_authors(folder, damps, top, min_year, plot_author_count=20, show_legend=True):
  graph = cite_graph(GRAPH_CSV)
  top_authors = most_cited_authors(graph, top, min_year)[:plot_author_count]
  author_nodes = mysql.get_authors()
  x_labels = [author_nodes[a[0]].name for a in top_authors]
  x_axis = range(1, plot_author_count + 1)
  top_author_ids = np.array([a[0] for a in top_authors])
  folder_path = "figs/%s/%s/authors/%s" % (THE.version, THE.permitted, folder)
  palette = np.array(sns.color_palette("hls", plot_author_count))
  legends = []
  # for i, f_name in enumerate(os.listdir(folder_path)):
  y_axes = []
  means = np.array([0.0] * plot_author_count)
  plt.figure(figsize=(8, 2))
  for i, _ in enumerate(damps):
    # file_name = "%s/%s" % (folder_path, name)
    file_name = "%s/page_rank_%0.2f.pkl" % (folder_path, damps[i])
    with open(file_name) as f:
      pr_scores = cPkl.load(f)
      y_axis = np.array([pr_scores[a] for a in top_author_ids])
      y_axes.append(y_axis)
      means += y_axis
  indices = np.argsort(means)[::-1]
  top_author_ids = top_author_ids[indices]
  # sns.set_style("whitegrid", {'axes.grid': False})
  sns.set_style("white")
  for i, y_axis in enumerate(y_axes):
    plt.plot(x_axis, y_axis[indices], c=palette[i])
    legends.append("%0.2f" % damps[i])
  if show_legend:
    plt.legend(legends, bbox_to_anchor=(-0.1, 1.15, 1.15, 0.2), loc="lower left",
               mode="expand", borderaxespad=0, ncol=10)
  fig_name = "figs/%s/%s/authors/damp_%s.png" % (THE.version, THE.permitted, folder)
  plt.ylabel("Page Rank Score", fontsize=14)
  plt.xlabel("Author ID", fontsize=14)
  plt.xticks(x_axis, top_author_ids, rotation='vertical')
  plt.title("Page Rank Score for top %d cited author with varying damping factors" % plot_author_count)
  plt.savefig(fig_name, bbox_inches='tight')
  plt.clf()


def get_author_genders():
  authors = mysql.get_authors()
  gender_map = {}
  for a_id, node in authors.items():
    if not node.gender:
      gender_map[a_id] = "u"
    else:
      gender_map[a_id] = node.gender
  return gender_map


def top_authors_with_gender(metric, count):
  # TODO: Create a file with more than 1% top authors for all methods
  file_name = "figs/%s/%s/authors/for_gender/%s_page_rank.pkl" % (THE.version, THE.permitted, metric)
  gender_map = get_author_genders()
  with open(file_name) as f:
    author_map = cPkl.load(f)
    author_tups = []
    for key, val in author_map.items():
      author_tups.append((key, val, gender_map[key]))
    author_tups = sorted(author_tups, key=lambda x: x[1], reverse=True)
  mc, fc, uc = 0, 0, 0
  for author in author_tups[:count]:
    if author[2] == 'm':
      mc += 1
    elif author[2] == 'f':
      fc += 1
    else:
      uc += 1
  print("For %s and top %d => Males, Females, Unknowns, FP: %d, %d, %d, %0.2f" % (metric.upper(), count, mc, fc, uc,                                                                                  100 * fc / (fc + mc)))
  return [mc, fc, uc, 100 * fc / (fc + mc)]


def run_top_authors_with_gender():
  gender_map = get_author_genders()
  mc, fc, uc = 0, 0, 0
  for a_id, gender in gender_map.items():
    if gender == 'm':
      mc += 1
    elif gender == 'f':
      fc += 1
    else:
      uc += 1
  print("For all authors => Males, Females, Unknowns: %d, %d, %d, %0.2f" % (mc, fc, uc, 100 * fc / (fc + mc)))
  x_axis = [10, 20, 50, 100, 200, 500, 1000]
  y_axes = {}
  metrics = ["naive", "cite", "publ"]
  for metric in metrics:
    y_axis = []
    for count in x_axis:
      y_axis.append(top_authors_with_gender(metric, count))
    y_axes[metric] = y_axis
  width = 0.3
  print(y_axes)


def plot_top_authors_with_genders():
  def autolabel(rects, labels):
    """
    Attach a text label above each bar displaying its height
    """
    for rect, label in zip(rects, labels):
      height = rect.get_height()
      ax.text(rect.get_x() + rect.get_width() / 2., height + 1,
              '%d:%d' % (label[1], label[0]),
              ha='center', va='bottom', rotation='vertical')
  y_axes = {
      'naive': [[10, 0, 0, 0.0],
                [17, 3, 0, 15.0],
                [41, 8, 1, 16.3265306122449],
                [73, 22, 5, 23.157894736842106],
                [145, 43, 12, 22.872340425531913],
                [367, 100, 33, 21.41327623126338],
                [738, 188, 74, 20.302375809935207]],
      'cite': [[6, 3, 1, 33.333333333333336],
               [15, 4, 1, 21.05263157894737],
               [38, 9, 3, 19.148936170212767],
               [73, 22, 5, 23.157894736842106],
               [146, 38, 16, 20.652173913043477],
               [365, 97, 38, 20.995670995670995],
               [731, 194, 75, 20.972972972972972]],
      'publ': [[6, 3, 1, 33.333333333333336],
               [15, 4, 1, 21.05263157894737],
               [38, 9, 3, 19.148936170212767],
               [73, 22, 5, 23.157894736842106],
               [146, 38, 16, 20.652173913043477],
               [365, 97, 38, 20.995670995670995],
               [732, 194, 74, 20.950323974082075]]
  }
  fig, ax = plt.subplots()
  width = 0.5
  x_axis = [10, 20, 50, 100, 200, 500, 1000]
  metrics = ["naive", "cite", "publ"]
  ind = np.arange(len(x_axis))
  percenter = lambda x: round(x[-1], 2)
  naives = map(percenter, y_axes['naive'])
  cites = map(percenter, y_axes['cite'])
  publs = map(percenter, y_axes['publ'])
  naive_bar = ax.bar(2*ind, naives, width, color='r')
  cite_bar = ax.bar(2*ind + width, cites, width, color='y')
  publ_bar = ax.bar(2*ind + 2 * width, publs, width, color='b')
  ax.set_ylabel('% of women in top authors')
  ax.set_xlabel('top authors under consideration')
  ax.set_title('% of women in top X authors where X is varied b/w 10-1000')
  ax.set_xticks(2*ind + width)
  ax.set_xticklabels(x_axis)
  ax.set_ylim([0, 38])
  autolabel(naive_bar, y_axes['naive'])
  autolabel(cite_bar, y_axes['cite'])
  autolabel(publ_bar, y_axes['publ'])
  plt.legend((naive_bar, cite_bar, publ_bar), ("PR", "PR_cite", "PR_publ"))
  plt.savefig("figs/%s/%s/authors/for_gender/women.png" % (THE.version, THE.permitted), bbox_inches='tight')
  plt.clf()


def plot_single_top_authors_with_genders():
  def autolabel(line, labels):
    """
    Attach a text label above each bar displaying its height
    """
    for x_coord, y_coord, label in zip(line.get_xdata(), line.get_ydata(), labels):
      plt.text(x_coord, y_coord + 1,
               '%d:%d' % (label[1], label[0]),
               ha='center', va='bottom')
  pr = [[10, 0, 0, 0.0],
        [17, 3, 0, 15.0],
        [41, 8, 1, 16.3265306122449],
        [73, 22, 5, 23.157894736842106],
        [145, 43, 12, 22.872340425531913],
        [367, 100, 33, 21.41327623126338],
        [738, 188, 74, 20.302375809935207]]

  width = 0.5
  x_axis = [10, 20, 50, 100, 200, 500, 1000]
  metrics = ["naive", "cite", "publ"]
  ind = np.arange(len(x_axis))
  percenter = lambda x: round(x[-1], 2)
  scores = map(percenter, pr)
  plt.figure(figsize=(8, 2))
  lines = plt.plot(ind, scores, color='r')
  plt.ylabel('% of women in top authors')
  plt.xlabel('top authors under consideration')
  plt.title('% of women in top X authors where X is varied b/w 10-1000')
  plt.xticks(ind, x_axis)
  # plt.xticklabels(x_axis)
  plt.ylim([0, 38])
  autolabel(lines[0], pr)
  # autolabel(cite_bar, y_axes['cite'])
  # autolabel(publ_bar, y_axes['publ'])
  # plt.legend((naive_bar, cite_bar, publ_bar), ("PR", "PR_cite", "PR_publ"))
  plt.savefig("figs/%s/%s/authors/for_gender/women_line.png" % (THE.version, THE.permitted), bbox_inches='tight')
  plt.clf()


def get_h_index(cites):
  if len(cites) == 0:
    return 0
  cites = sorted(cites, reverse=True)
  for i, cite in enumerate(cites):
    if i > cite:
      return i
  return 0


def save_authors_by_h_index(save_file=None, min_year=1992):
  graph = cite_graph(GRAPH_CSV)
  papers_by_authors = graph.get_papers_by_authors(permitted=THE.permitted)
  paper_cite_map = {}
  author_map = {}
  for author, papers in papers_by_authors.items():
    author_cites = []
    for paper_tup in papers:
      year = int(paper_tup[1])
      if year < min_year:
        continue
      paper_id = paper_tup[0]
      if paper_id in paper_cite_map:
        cites = paper_cite_map[paper_id]
      else:
        cites = graph.paper_nodes[paper_id]['cited_count']
        cites = int(cites) if is_not_none(cites) else 0
        paper_cite_map[paper_id] = cites
      author_cites.append(cites)
    h_index = get_h_index(author_cites)
    author_map[author] = h_index
  if save_file:
    with open(save_file, "wb") as f:
      cPkl.dump(author_map, f, cPkl.HIGHEST_PROTOCOL)
  return author_map


def get_authors_by_h_index(save_file, top_count=100):
  with open(save_file) as f:
    author_data = cPkl.load(f)
  authors = mysql.get_authors()
  h_index_results = sorted([(a_id, authors[a_id]['name'], score) for a_id, score in author_data.items()],
                           key=lambda x: x[2], reverse=True)
  for i, result in enumerate(h_index_results[:top_count]):
    print(i + 1, result[1], result[2])


def _damp_plotter(damps=None):
  if damps is None:
    damps = np.arange(0.05, 1.00, 0.05)
  plot_damp_top_authors("naive", damps, top=0.01, min_year=1992)
  plot_damp_top_authors("cite", damps, top=0.01, min_year=1992, show_legend=False)
  plot_damp_top_authors("publ", damps, top=0.01, min_year=1992, show_legend=False)


def _damp_scores():
  iters = 1000
  damps = np.arange(0.05, 1.00, 0.05)
  for damp in damps:
    run_page_rank(1992, damp, use_prefix=True, iterations=iters)
    run_weighted_page_rank(1992, damp, weight_param="cite", use_prefix=True, iterations=iters)
    run_weighted_page_rank(1992, damp, weight_param="publ", use_prefix=True, iterations=iters)
  _damp_plotter(damps)


def _top_authors_for_gender(do_print=False):
  damp = 0.5
  iters = 1000
  run_page_rank(1992, damp, top=0.05, use_prefix=True, iterations=iters, f_name="for_gender/naive_page_rank")
  run_weighted_page_rank(1992, damp, top=0.05, weight_param="cite", iterations=iters, f_name="for_gender/cite_page_rank")
  run_weighted_page_rank(1992, damp, top=0.05, weight_param="publ", iterations=iters, f_name="for_gender/publ_page_rank")


def print_top_author_names(file_name):
  naive_pr_file = "figs/%s/%s/authors/%s.pkl" % (THE.version, THE.permitted, "for_gender/naive_page_rank")
  cite_pr_file = "figs/%s/%s/authors/%s.pkl" % (THE.version, THE.permitted, "for_gender/cite_page_rank")
  publ_pr_file = "figs/%s/%s/authors/%s.pkl" % (THE.version, THE.permitted, "for_gender/publ_page_rank")
  authors = mysql.get_authors()
  pr_results = sorted([
                        authors[a_id]['name']
                        # (a_id, authors[a_id]['name'], score)
                        for a_id, score in open_pkl(file_name).items()],
                      key=lambda x: x[1], reverse=True)
  print(pr_results[:100])


def author_bar(min_year=2000, max_year=2015):
  print("AUTHOR BAR for %s" % THE.permitted)
  graph = cite_graph(GRAPH_CSV)
  year_authors_map = OrderedDict()
  for _, paper in graph.get_paper_nodes(permitted=THE.permitted).items():
    year = paper.year
    if not min_year <= int(year) <= max_year: continue
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
  plt.savefig("figs/%s/%s/authors/author_count.png" % (THE.version, THE.permitted), bbox_inches='tight')
  plt.clf()

if __name__ == "__main__":
  # author_bar()
  # _damp_scores()
  # _damp_plotter()
  # save_authors_by_h_index("figs/%s/%s/authors/h-index_1992.pkl" % (THE.version, THE.permitted), min_year=1992)
  # save_authors_by_h_index("figs/%s/%s/authors/h-index_2009.pkl" % (THE.version, THE.permitted), min_year=2009)
  # get_authors_by_h_index("figs/%s/%s/authors/h-index_1992.pkl" % (THE.version, THE.permitted))
  # get_authors_by_h_index("figs/%s/%s/authors/h-index_2009.pkl" % (THE.version, THE.permitted))
  # _top_authors_for_gender()
  # plot_top_authors_with_genders()
  # print_top_author_names("figs/%s/%s/authors/%s.pkl" % (THE.version, THE.permitted, "for_gender/cite_page_rank"))
  plot_single_top_authors_with_genders()

