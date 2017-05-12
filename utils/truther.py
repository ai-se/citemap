from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

from utils.lib import Node, Edge
import networkx as nx

CITEMAP_FILE = 'data/citemap_v9.csv'
MIN_SIZE = 3
SAVE_AS_NAME = False
TRUTH_ID_FILE = 'data/ground_truth_ids.csv'
TRUTH_NAME_FILE = 'data/ground_truth_names.csv'
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
        for i in range(len(paper_authors)):
          for j in range(i + 1, len(paper_authors)):
            edge = Edge(source=paper_authors[i].id, target=paper_authors[j].id, edge_type='co_author')
            edges.append(edge)
        all_edges = conference_edge_map.get(conference_id, [])
        all_edges += edges
        conference_edge_map[conference_id] = all_edges
  return conference_edge_map, author_nodes


def make_ground_truth():
  edge_map, node_map = read(CITEMAP_FILE)
  f_id = open(TRUTH_ID_FILE, 'wb')
  f_name = open(TRUTH_NAME_FILE, 'wb')
  for conference in edge_map.keys():
    edges = edge_map[conference]
    graph = nx.Graph()
    edge_ids = [(edge.source, edge.target) for edge in edges]
    graph.add_edges_from(edge_ids)
    for component in nx.connected_components(graph):
      if len(component) >= MIN_SIZE:
        author_names = ", ".join([node_map[node_id].name for node_id in component])
        author_ids = ", ".join(component)
        f_id.write(author_ids + "\n")
        f_name.write(author_names + "\n")
  f_id.close()
  f_name.close()


if __name__ == "__main__":
  make_ground_truth()
