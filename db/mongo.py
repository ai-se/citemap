from __future__ import print_function, division
import sys, os
sys.path.append(os.path.abspath("."))
import pymongo
import bson

__author__ = "panzer"

MONGO_URI = "mongodb://localhost:27017/"

client = pymongo.MongoClient(MONGO_URI)
db = client["citemap"]


def index_paper_collection():
  """
  Index the Paper Collection
  :return:
  """
  db.drop_collection('paper')
  db.paper.create_index([('id', pymongo.ASCENDING),
                        ('refs', pymongo.ASCENDING),
                         ('title', pymongo.ASCENDING)], background=True)


def insert_papers(papers):
  db.paper.insert_many(papers)


def get_paper(title):
  papers = [paper for paper in db.paper.find({"title": title})]
  if papers:
    return papers[0]
  return None


def get_papers_with_titles(titles):
  tits = [title.decode('utf-8', 'ignore').encode("utf-8") for title in titles]
  cursor = db.paper.find({"title": {"$in": tits}})
  paper_map = {}
  print("Fetched All")
  count = 0
  for paper in cursor:
    if 'abstract' in paper:
      paper_map[paper["title"]] = paper
      count += 1
  print(count)
  return paper_map




