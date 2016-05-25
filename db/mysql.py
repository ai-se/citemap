from __future__ import print_function, division
import sys, os
sys.path.append(os.path.abspath("."))
import MySQLdb
from utils.lib import O, Paper
import csv

class DB(O):
  _db = None

  @staticmethod
  def get():
    if DB._db is None:
      DB._db =  MySQLdb.connect(host="localhost",
                         user="root",
                         passwd="root",
                         db="conferences")
    return DB._db

  @staticmethod
  def close():
    if DB._db is not None:
      DB._db.close()
      DB._db = None



def get_paper_names():
  db = DB.get()
  cur = db.cursor()
  cur.execute("SELECT title from papers")
  rows = [row[0] for row in cur.fetchall()]
  DB.close()
  return rows

def update_papers(paper_map):
  db = DB.get()
  cur = db.cursor()
  count = 0
  for title, details in paper_map.items():
    cur.execute("UPDATE papers SET ref_id=%s, cites=%s, abstract=%s WHERE title=%s", (details["ref_id"], details["cites"], details["abstract"], title))
    count += 1
    print("Statement : ", count)
  db.commit()
  DB.close()


def dump(to_csv=True, file_name='citemap.csv', delimiter="$|$"):
  db = DB.get()
  cur = db.cursor()
  cur.execute("SELECT * FROM papers")
  papers = []
  for row in cur.fetchall():
    paper = Paper()
    paper.id = row[0]
    paper.conference_id = row[1]
    paper.year = row[2]
    paper.title = row[3]
    paper.h2 = row[6]
    paper.h3 = row[7]
    paper.ref_id = row[9]
    paper.cites = row[10].split(",") if row[10] else []
    paper.authors = []
    paper.abstract = row[11]
    cur_authors = db.cursor()
    cur_authors.execute("SELECT persons.name "
                "FROM persons, authorship "
                "WHERE persons.id = authorship.person_id AND authorship.paper_id = %d"%int(paper.id))
    for authors in cur_authors.fetchall():
      paper.authors.append(authors[0])
    papers.append(paper)
  if not to_csv:
    return papers
  header = ["ID", "Conference", "Year", "Title", "H2", "H3", "Ref_ID", "Cites", "Authors", "Abstract"]
  with open(file_name, 'wb') as f:
    f.write(delimiter.join(header)+"\n")
    for i, paper in enumerate(papers):
      cites = ",".join(paper.cites) if paper.cites else ""
      authors = ",".join(paper.authors) if paper.authors else ""
      row = [paper.id, paper.conference_id, paper.year, paper.title, paper.h2, paper.h3, paper.ref_id, cites, authors,
             paper.abstract]
      f.write(delimiter.join(map(str, row)) + "\n")





dump()