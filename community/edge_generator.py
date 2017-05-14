from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

__author__ = "bigfatnoob"

from community import truther
import itertools
import networkx as nx
import numpy as np


TRUTH_FILE = "community/results/ground_truth_ids.csv"


def avg(lst):
  return sum(lst) / len(lst)


def gen_ganaxis(inp, out):
  edge_map, _, __ = truther.map_for_nx(inp)
  with open(out, "wb") as f:
    for key, val in edge_map.items():
      [source, target] = key.split("-")
      # f.write("%s %s %d\n" % (source, target, val))
      f.write("%s %s\n" % (source, target))


def gen_top_gc(inp, out):
  edge_map, _, __ = truther.map_for_nx(inp)
  with open(out, "wb") as f:
    for key, val in edge_map.items():
      [source, target] = key.split("-")
      # f.write("%s\t%s\t%d\n" % (source, target, val))
      f.write("%s\t%s\n" % (source, target))


def parse_ground_truth(inp):
  communities = {}
  with open(inp) as f:
    for index, line in enumerate(f.readlines()):
      communities[index] = map(int, line.split(", "))
  return communities


def parse_ganxisw_community(inp):
  communities = {}
  with open(inp) as f:
    for line in f.readlines():
      [community, node] = map(int, line.split(" "))
      nodes = communities.get(community, [])
      nodes.append(node)
      communities[community] = nodes
  for community, nodes in communities.items():
    if len(nodes) < truther.MIN_SIZE:
      del communities[community]
  return communities


def parse_top_gc(inp):
  communities = {}
  with open(inp) as f:
    for index, line in enumerate(f.readlines()):
      communities[index] = map(int, line.split())
  return communities


def generate_cluster_pairs(communities, graph=None, size=None):
  components = communities.values()
  if size and graph and len(communities) > size:
    components = []
    median_degree = np.median(graph.degree(graph.nodes()).values())
    for component in communities.values():
      community = graph.subgraph(component)
      v_count = len(community.nodes())
      fomd = sum([1 for v in component if len(set(graph.neighbors(v)) & set(component)) > median_degree]) / v_count
      internal_density = nx.density(community)
      components.append((component, fomd, internal_density))
    components = sorted(components, key=lambda x: x[1], reverse=True)[:size]
    components = [c for c, _, __ in components]
  pairs = set()
  comms = []
  for nodes in components:
    comms.append(nodes)
    for combination in list(itertools.combinations(nodes, 2)):
      pairs.add(combination)
  return pairs, comms


def make_graph(inp):
  edge_map, _, __ = truther.map_for_nx(inp)
  graph = nx.Graph()
  weighted_edges = []
  for key, val in edge_map.items():
    [source, target] = map(int, key.split("-"))
    weighted_edges.append((source, target, val))
  graph.add_weighted_edges_from(weighted_edges)
  return graph


def goodness(graph, clusters):
  separabilities, densities, cohesivenesses, clustering_coeffs = [], [], [], []
  for i, cluster in enumerate(clusters):
    vertices = list(cluster)
    community = graph.subgraph(vertices)
    # community_size = len(community.nodes())
    inner_edges = []
    neighbors = []
    for vertex in vertices:
      inner_edges.append(len(set(graph.neighbors(vertex)).intersection(set(vertices))))
      neighbors.append(graph.neighbors(vertex))
    ms = len(community.edges())
    cs = sum([len(set(graph.neighbors(v)).difference(set(vertices))) for v in vertices])
    """
    Separability
    """
    separability = ms / cs if cs != 0 else ms
    separabilities.append(separability)
    """
    Density
    """
    density = nx.density(community) if len(vertices) > 1 else 0
    densities.append(density)
    """
    Cohesiveness
    """
    if len(vertices) == 1:
      cohesiveness = 1
    elif not nx.is_connected(community):
      cohesiveness = 0
    else:
      cohesiveness = len(nx.minimum_edge_cut(community)) / (len(vertices) - 1)
    cohesivenesses.append(cohesiveness)
    """
    Clustering Coefficient
    """
    clustering_coeff = nx.transitivity(community)
    clustering_coeffs.append(clustering_coeff)
  return avg(separabilities), avg(densities), avg(cohesivenesses), avg(clustering_coeffs)


def performance(graph, obtained_pairs, truth_pairs):
  n_vertices = len(graph.nodes())
  tp = len(obtained_pairs.intersection(truth_pairs))
  fp = len(obtained_pairs.difference(truth_pairs))
  fn = len(truth_pairs.difference(obtained_pairs))
  tn = ((n_vertices * (n_vertices - 1)) / 2) - (tp + fp + fn)
  precision = tp / (tp + fp)
  recall = tp / (tp + fn)
  f_score = 2 * precision * recall / (precision + recall)
  accuracy = (tp + tn) / (tp + tn + fp + fn)
  return precision, recall, f_score, accuracy


def results(predicted_communities, actual_communities):
  graph = make_graph(truther.CITEMAP_FILE)
  a_pairs, a_clusters = generate_cluster_pairs(actual_communities)
  p_pairs, p_clusters = generate_cluster_pairs(predicted_communities, graph, len(a_clusters))
  graph = make_graph(truther.CITEMAP_FILE)
  separability, density, cohesiveness, clustering_coeff = goodness(graph, p_clusters)
  precision, recall, f_score, accuracy = performance(graph, a_pairs, p_pairs)
  print("Separability: %.2f" % separability)
  print("Density: %.2f" % density)
  print("Cohessiveness: %.2f" % cohesiveness)
  print("Clustering Coefficient: %.2f" % clustering_coeff)
  print("Precision: %.2f" % precision)
  print("Recall: %.2f" % recall)
  print("F1 Score: %.2f" % f_score)
  print("Accuracy: %.2f" % accuracy)


def _validate_ganxisw():
  rs = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
  file_pref = "community/results/SLPAw_citemap_run1_r%s_v3_T100.icpm.com-node.txt"
  truth = parse_ground_truth("community/results/ground_truth_ids.csv")
  for r in rs:
    ganxisw_comm = parse_ganxisw_community(file_pref % r)
    print("## R = %s" % r)
    results(ganxisw_comm, truth)
    print("\n")


def _main():
  truth = parse_ground_truth("community/results/ground_truth_ids.csv")
  print("\nTOP GC")
  top_gc_comm = parse_top_gc("community/results/top_gc_formatted.ipairs.clusters_directed")
  results(top_gc_comm, truth)
  print ("\nGANXiSw")
  ganxisw_comm = parse_ganxisw_community("community/results/ganaxis.txt")
  results(ganxisw_comm, truth)


if __name__ == "__main__":
  # gen_ganaxis(truther.CITEMAP_FILE, "community/inputs/ganxisw.ipairs")
  # gen_top_gc(truther.CITEMAP_FILE, "community/inputs/top_gc.ipairs")
  # print(len(parse_ganxisw_community("community/results/SLPAw_citemap_run1_r0.25_v3_T100.icpm.com-node.txt")))
  # _validate_ganxisw()
  _main()
