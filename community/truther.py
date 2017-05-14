from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

from utils.lib import Node, Edge
import networkx as nx
import numpy as np

CITEMAP_FILE = 'data/citemap_v9.csv'
MIN_SIZE = 3
SAVE_AS_NAME = False
TRUTH_ID_FILE = 'community/results/ground_truth_ids.csv'
TRUTH_NAME_FILE = 'community/results/ground_truth_names.csv'
ID_INDEX = -5
NAME_INDEX = -4


def read(file_name, delimiter='$|$'):
  author_nodes = {}
  conference_edge_map = {}
  with open(file_name) as f:
    f.readline().strip().lower().split(delimiter)
    for line in f.readlines():
      columns = line.strip().split(delimiter)
      conference_id = int(columns[1])
      paper_authors = []
      for author_id, author in zip(columns[ID_INDEX].split(","), columns[NAME_INDEX].split(",")):
        if author in author_nodes:
          author_node = author_nodes[author]
        else:
          author_node = Node()
          author_node["id"] = author_id
          author_node["name"] = author
          author_nodes[author_id] = author_node
        paper_authors.append(author_node)
      if len(paper_authors) > 1:
        edges = []
        for i in range(len(paper_authors) - 1):
          for j in range(i + 1, len(paper_authors)):
            edge1 = Edge(source=paper_authors[i].id, target=paper_authors[j].id, edge_type='co_author')
            # edge2 = Edge(source=paper_authors[j].id, target=paper_authors[i].id, edge_type='co_author')
            edges.append(edge1)
            # edges.append(edge2)
        all_edges = conference_edge_map.get(conference_id, [])
        all_edges += edges
        conference_edge_map[conference_id] = all_edges
  return conference_edge_map, author_nodes


def map_for_nx(inp):
  venue_edge_map, node_map = read(inp)
  edge_map = {}
  for venue, edges in venue_edge_map.items():
    for edge in edges:
      key = edge.source + "-" + edge.target
      edge_map[key] = edge_map.get(key, 0) + 1
  return edge_map, venue_edge_map, node_map


def make_ground_truth():
  edge_map, venue_edge_map, node_map = map_for_nx(CITEMAP_FILE)
  components = []
  for conference in venue_edge_map.keys():
    edges = venue_edge_map[conference]
    graph = nx.Graph()
    edge_ids = [(int(edge.source), int(edge.target)) for edge in edges]
    graph.add_edges_from(edge_ids)
    median_degree = np.median(graph.degree(graph.nodes()).values())
    for component in nx.connected_components(graph):
      if len(component) >= MIN_SIZE:
        community = graph.subgraph(component)
        v_count = len(community.nodes())
        fomd = sum([1 for v in component if len(set(graph.neighbors(v)) & set(component)) > median_degree]) / v_count
        internal_density = nx.density(community)
        components.append((component, fomd, internal_density))
  components = sorted(components, key=lambda x: x[1], reverse=True)[:3000]
  components = sorted(components, key=lambda x: x[2], reverse=True)[:int(0.75 * len(components))]
  f_id = open(TRUTH_ID_FILE, 'wb')
  f_name = open(TRUTH_NAME_FILE, 'wb')
  for component, fomd, internal_density in components:
    component = map(str, component)
    author_names = ", ".join([node_map[node_id].name for node_id in component])
    author_ids = ", ".join(component)
    f_id.write(author_ids + "\n")
    f_name.write(author_names + "\n")
  f_id.close()
  f_name.close()


if __name__ == "__main__":
  make_ground_truth()
