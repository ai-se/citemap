from __future__ import print_function, division
import os, sys
sys.path.append(os.path.abspath("."))
from utils.lib import O, Node, Edge
from db import mysql
from utils.unicoder import UnicodeReader

__author__ = "panzer"


class Graph(O):
  def __init__(self):
    O.__init__(self)
    self.paper_nodes = None           # Paper Nodes
    self.author_nodes = None          # Author Nodes
    self.author_edges = None          # Directed Edges between author and paper
    self.cite_edges = None            # Directed Edges between reference paper and base paper
    self.collaborator_edges = None    # Weighted Undirected edges between authors

  @staticmethod
  def from_file(file_name, delimiter='$|$'):

    paper_nodes = {}
    author_edges = {}
    cite_edges = {}
    collaborator_edges = {}
    author_nodes = mysql.get_authors()
    ref_nodes = {}

    def add_collaborator_edges(authors):
      if len(authors) <= 1:
        return
      for i in range(len(authors)):
        for j in range(i + 1, len(authors)):
          low, high = min(authors[i].id, authors[j].id), max(authors[i].id, authors[j].id)
          key = low + "-" + high
          e = collaborator_edges.get(key, None)
          if e is None:
            e = Edge(source=low, target=high, edge_type="collaborator", count=1)
          else:
            e.count += 1
          collaborator_edges[key] = e

    with open(file_name, 'rb') as f:
      column_names = f.readline().strip().lower().split(delimiter)
      for line in f.readlines():
        line = line.decode('utf-8', 'ignore').encode("utf-8")
        columns = line.strip().split(delimiter)
        paper_node = Node()
        for name, val in zip(column_names, columns):
          paper_node[name] = val
        paper_node["type"] = "paper"
        if paper_node.ref_id:
          ref_nodes[paper_node.ref_id] = paper_node
        paper_nodes[paper_node.id] = paper_node
        paper_authors = []
        for author_id, author in zip(columns[-3].split(","), columns[-2].split(",")):
          author_node = author_nodes[author_id]
          paper_authors.append(author_node)
          edge = Edge(source=author_node.id, target=paper_node.id, edge_type="author")
          author_edges[edge.id] = edge
        add_collaborator_edges(paper_authors)

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
          cite_edges[edge.id] = edge
      for paper_id, paper in paper_nodes.items():
        paper["cited_counts"] =cited_counts.get(paper_id, 0)
    graph = Graph()
    graph.paper_nodes = paper_nodes
    graph.author_nodes = author_nodes
    graph.author_edges = author_edges
    graph.cite_edges = cite_edges
    graph.collaborator_edges = collaborator_edges
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

  def get_papers_by_venue(self, permitted='conferences'):
    """
    :param permitted: ['conferences', 'journals', 'all']
    :return: {
      <conference_id>: [(<paper>, <year>), ...]
    }
    """
    papers = {}
    venues = mysql.get_venues()
    for paper_id, paper in self.paper_nodes.items():
      venue = venues[paper.venue]
      if venue.is_conference and permitted == 'journals': continue
      if not venue.is_conference and permitted == 'conferences': continue
      c_papers = papers.get(paper.venue, [])
      c_papers.append((paper.id, paper.year))
      papers[paper.venue] = c_papers
    for conference_id, c_papers in papers.items():
      papers[conference_id] = sorted(c_papers, key=lambda tup: tup[1])
    return papers

  def get_paper_nodes(self, permitted='conferences'):
    paper_nodes = {}
    venues = mysql.get_venues()
    for p_id, paper in self.paper_nodes.items():
      venue = venues[paper.venue]
      if venue.is_conference and permitted == 'journals':
        continue
      if not venue.is_conference and permitted == 'conferences':
        continue
      paper_nodes[p_id] = paper
    return paper_nodes

  def get_papers_by_authors(self, permitted='conferences'):
    """
    :return:{
      <author_id> : [(<paper>, <year>, <conference>) , ...]
    }
    """
    papers = {}
    venues = mysql.get_venues()
    for edge in self.author_edges.values():
      author_papers = papers.get(edge.source, [])
      paper = self.paper_nodes[edge.target]
      venue = venues[paper.venue]
      if venue.is_conference and permitted == 'journals': continue
      if not venue.is_conference and permitted == 'conferences': continue
      author_papers.append((paper.id, paper.year, paper.venue))
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
