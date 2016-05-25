from __future__ import print_function, division
import os, sys
sys.path.append(os.path.abspath("."))
from utils.lib import O

__author__ = "panzer"

class Node(O):
  id = 1
  def __init__(self, **params):
    O.__init__(self, **params)
    self.id = Node.id
    Node.id += 1

class Edge(O):
  id = 1
  def __init__(self, **params):
    O.__init__(self, **params)
    self.id =  Edge.id
    Edge.id += 1

class Graph(O):
  def __init__(self):
    O.__init__(self)
    self.paper_nodes = None
    self.author_nodes = None
    self.edges = None

  @staticmethod
  def from_file(file_name, delimiter='$|$'):
    paper_nodes = {}
    edges = {}
    author_nodes = {}
    ref_nodes = {}
    with open(file_name, 'rb') as f:
      column_names = f.readline().strip().split(delimiter)
      print(column_names)
      for line in f.readlines():
        columns = line.strip().split(delimiter)
        paper_node = Node()
        for name, val in zip(column_names, columns):
          paper_node[name] = val
        paper_node["type"] = "paper"
        if paper_node.Ref_ID:
          ref_nodes[paper_node.Ref_ID] = paper_node
        paper_nodes[paper_node.id] = paper_node
        for author in columns[-2].split(","):
          if author in author_nodes:
            author_node = author_nodes[author]
          else:
            author_node = Node()
            author_node["author_id"] = len(author_nodes)
            author_node["name"] = author
            author_node["type"] = "author"
            author_nodes[author_node.id] = author_node
          edge = Edge(source=author_node.id, target=paper_node.id, edge_type="author")
          edges[edge.id] = edge
      for paper_id, paper in paper_nodes.items():
        if not paper.Ref_ID: continue
        references = paper.Cites
        if not references: continue
        target = ref_nodes[paper.Ref_ID]
        for ref_id in references.split(","):
          if not ref_nodes.get(ref_id, None): continue
          source = ref_nodes[ref_id]
          edge = Edge(source=source.id, target=target.id, edge_type="cite")
          edges[edge.id] = edge
    graph = Graph()
    graph.paper_nodes = paper_nodes
    graph.author_nodes = author_nodes
    graph.edges = edges
    return graph

Graph.from_file("citemap.csv")