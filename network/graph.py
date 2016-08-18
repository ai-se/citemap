from __future__ import print_function, division
import os, sys
sys.path.append(os.path.abspath("."))
from utils.lib import O, Node, Edge
from db import mysql

__author__ = "panzer"

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
    author_nodes = mysql.get_authors()
    ref_nodes = {}
    with open(file_name, 'rb') as f:
      column_names = f.readline().strip().lower().split(delimiter)
      #print(column_names)
      for line in f.readlines():
        columns = line.strip().split(delimiter)
        paper_node = Node()
        for name, val in zip(column_names, columns):
          paper_node[name] = val
        paper_node["type"] = "paper"
        if paper_node.ref_id:
          ref_nodes[paper_node.ref_id] = paper_node
        paper_nodes[paper_node.id] = paper_node
        for author_id, author in zip(columns[-3].split(","), columns[-2].split(",")):
          #author_node = author_nodes[int(author_id)]
          author_node = author_nodes[author_id]
          # if author in author_nodes:
          #   author_node = author_nodes[author]
          # else:
          #   author_node = Node()
          #   author_node["id"] = author_id
          #   author_node["name"] = author
          #   author_node["type"] = "author"
          #   author_nodes[author_node.id] = author_node
          edge = Edge(source=author_node.id, target=paper_node.id, edge_type="author")
          edges[edge.id] = edge
      cited_counts = {}
      for paper_id, paper in paper_nodes.items():
        if not paper.ref_id: continue
        references = paper.cites
        if not references: continue
        target = ref_nodes[paper.ref_id]
        for ref_id in references.split(","):
          if not ref_nodes.get(ref_id, None): continue
          source = ref_nodes[ref_id]
          source_cited = cited_counts.get(source.id, 0)
          cited_counts[source.id] = source_cited + 1
          edge = Edge(source=source.id, target=target.id, edge_type="cite")
          edges[edge.id] = edge
      for paper_id, paper in paper_nodes.items():
        paper["cited_counts"] =cited_counts.get(paper_id, 0)
    graph = Graph()
    graph.paper_nodes = paper_nodes
    graph.author_nodes = author_nodes
    graph.edges = edges
    graph.add_pc_membership(mysql.get_pc_membership())
    return graph

  def add_pc_membership(self, pc_members):
    authors = self.author_nodes
    for member_id, membership in pc_members.items():
      author = authors[member_id]
      author.membership = membership

  def get_committee_by_conference(self):
    """
      :return:{
        <conference_id> : [(<author_id>, <year>, <role>) , ...]
      }
      """
    conferences = {}
    for author_id, author in self.author_nodes.items():
      if "membership" not in author.has():
        continue
      for pc in author.membership:
        conference = conferences.get(pc.conference_id, [])
        conference.append((pc.author_id, pc.year, pc.role))
        conferences[pc.conference_id] = conference
    for conference_id, conference in conferences.items():
      conferences[conference_id] = sorted(conference, key=lambda tup: (int(tup[1]), tup[2], tup[0]))
    return conferences


  def get_papers_by_conference(self):
    """
    :return: {
      <conference_id>: [(<paper>, <year>), ...]
    }
    """
    papers = {}
    for paper_id, paper in self.paper_nodes.items():
      c_papers = papers.get(paper.conference, [])
      c_papers.append((paper.id, paper.year))
      papers[paper.conference] = c_papers
    for conference_id, c_papers in papers.items():
      papers[conference_id] = sorted(c_papers, key=lambda tup: tup[1])
    return papers

  def get_papers_by_authors(self):
    """
    :return:{
      <author_id> : [(<paper>, <year>, <conference>) , ...]
    }
    """
    papers = {}
    for edge in self.edges.values():
      if edge.edge_type != "cite":
        continue
      author_papers = papers.get(edge.source, [])
      paper = self.paper_nodes[edge.target]
      author_papers.append((paper.id, paper.year, paper.conference))
      papers[edge.source] = author_papers
    for author_id, a_papers in papers.items():
      papers[author_id] = sorted(a_papers, key=lambda tup: (int(tup[2]), tup[1]))
    return papers




if __name__ == "__main__":
  g = Graph.from_file("citemap.csv")
  g.add_pc_membership(mysql.get_pc_membership())
  confs = g.get_committee_by_conference()
  paps = g.get_papers_by_authors()
  print(paps)
