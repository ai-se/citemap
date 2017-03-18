from __future__ import print_function, division
import os, sys
sys.path.append(os.path.abspath("."))
from db import mongo, mysql

__author__ = "panzer"


def abstracts():
  def convert(t):
    return t.decode('utf-8', 'ignore').encode("utf-8").lower()
  paper_names, years = mysql.get_paper_names_and_years()
  paper_map = mongo.get_papers_with_titles(paper_names)
  count = 0
  papers = {}
  for i, (name, year) in enumerate(zip(paper_names, years)):
    key = mongo.KEY_SEPERATOR.join([convert(name), str(year)])
    paper = paper_map.get(key, None)
    if paper is None:
      paper = paper_map.get(convert(name), None)
    details = {}
    if paper:
      details["ref_id"] = str(paper["id"])
      details["cites"] = None
      if paper["refs"]:
        details["cites"] = str(",".join(paper["refs"]))
      details["abstract"] = None
      if 'abstract' in paper:
        details["abstract"] = str(paper["abstract"].encode("ascii", "ignore"))
      papers[(name, year)] = details
    if papers and (i + 1) % 1000 == 0:
      print("Batch : %d" % ((i + 1) / 1000))
      mysql.update_papers(papers)
      papers = {}
  print("\n%d / %d" % (count, len(paper_names)))


abstracts()
