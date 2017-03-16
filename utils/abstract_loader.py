from __future__ import print_function, division
import os, sys
sys.path.append(os.path.abspath("."))
from db import mongo, mysql

__author__ = "panzer"


def abstracts():
  paper_names = mysql.get_paper_names()
  paper_map = mongo.get_papers_with_titles(paper_names)
  count = 0
  papers = {}
  for i, name in enumerate(paper_names):
    paper = paper_map.get(name, None)
    details = {}
    if paper:
      details["ref_id"] = str(paper["id"])
      if paper["refs"]:
        details["cites"] = str(",".join(paper["refs"]))
      else:
        details["cites"] = ""
      details["abstract"] = str(paper["abstract"].encode("ascii", "ignore"))
      papers[name] = details
    if papers and (i + 1) % 1000 == 0:
      print("Batch : %d" % ((i + 1) / 1000))
      mysql.update_papers(papers)
      papers = {}
  print("\n%d / %d" % (count, len(paper_names)))


abstracts()
