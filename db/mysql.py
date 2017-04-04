from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
import MySQLdb
from utils.lib import O, Paper, PC, Node, Venue
from collections import OrderedDict
import cPickle as pkl

# SCHEMA_NAME = "conferences"
# SCHEMA_NAME = "conferences_dummy"
SCHEMA_NAME = "se"


class DB(O):
  _db = None

  @staticmethod
  def get():
    if DB._db is None:
      DB._db = MySQLdb.connect(host="localhost",
                               user="root",
                               passwd="root",
                               db=SCHEMA_NAME)
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


def get_paper_names_and_years():
  db = DB.get()
  cur = db.cursor()
  cur.execute("SELECT title, year from papers")
  names = []
  years = []
  for row in cur.fetchall():
    names.append(row[0])
    years.append(row[1])
  DB.close()
  return names, years


def update_papers(paper_map):
  db = DB.get()
  cur = db.cursor()
  count = 0
  stmt = "UPDATE papers SET ref_id=%s, cites=%s, abstract=%s WHERE title=%s AND year=%s"
  values = []
  for (title, year), details in paper_map.items():
    # cur.execute(stmt,
    #             (details["ref_id"], details["cites"], details["abstract"], title, year))
    values.append((details["ref_id"], details["cites"], details["abstract"], title, year))
    count += 1
    # print("Statement : ", count)
  cur.executemany(stmt, values)
  db.commit()
  DB.close()


def get_venues():
  db = DB.get()
  cur = db.cursor()
  cur.execute('SELECT * FROM venues')
  venues = OrderedDict()
  for row in cur.fetchall():
    venue = O()
    venue.id = str(row[0])
    venue.acronym = row[1]
    venue.name = row[2]
    venue.impact = row[3]
    venue.is_conference = True if row[4] == 1 else False
    venues[venue.id] = venue
  DB.close()
  return venues


def dump(to_csv=True, file_name='data/citemap.csv', delimiter="$|$"):
  cur = DB.get().cursor()
  cur.execute("SELECT * FROM papers")
  papers = []
  venues = get_venues()
  for row in cur.fetchall():
    paper = Paper()
    paper.id = row[0]
    paper.venue_id = row[1]
    paper.year = row[2]
    paper.title = row[3]
    paper.h2 = row[6]
    paper.h3 = row[7]
    paper.ref_id = row[9]
    paper.cites = row[10].split(",") if row[10] else []
    paper.authors = []
    paper.author_sql_ids = []
    paper.abstract = row[11]
    paper.doi_url = row[12]
    paper.cited_count = row[13]
    paper.is_conference = True if venues[str(paper.venue_id)].is_conference == 1 else False
    cur_authors = DB.get().cursor()
    cur_authors.execute("SELECT persons.id, persons.name FROM persons, authorship "
                        "WHERE persons.id = authorship.person_id AND authorship.paper_id = %d" % int(paper.id))
    for authors in cur_authors.fetchall():
      paper.author_sql_ids.append(str(int(authors[0])))
      paper.authors.append(authors[1])
    papers.append(paper)
  if not to_csv:
    return papers
  header = ["ID", "Venue", "Is_Conference", "Year", "Title", "H2",
            "H3", "Ref_ID", "Cites", "Author_IDs", "Authors", "Abstract", "DOI_URL", "Cited_Count"]
  with open(file_name, 'wb') as f:
    f.write(delimiter.join(header) + "\n")
    for i, paper in enumerate(papers):
      cites = ",".join(paper.cites) if paper.cites else ""
      authors = ",".join(paper.authors) if paper.authors else ""
      author_ids = ",".join(paper.author_sql_ids) if paper.author_sql_ids else ""
      row = [paper.id, paper.venue_id, paper.is_conference, paper.year, paper.title, paper.h2,
             paper.h3, paper.ref_id, cites, author_ids, authors, paper.abstract, paper.doi_url, paper.cited_count]
      f.write(delimiter.join(map(str, row)) + "\n")
  DB.close()


def get_pc_membership():
  db = DB.get()
  cur = db.cursor()
  cur.execute("SELECT * FROM pc_membership")
  pc_members = {}
  for row in cur.fetchall():
    pc = PC()
    pc.author_id = str(int(row[1]))
    pc.conference_id = str(int(row[2]))
    pc.year = str(int(row[3]))
    pc.set_short_role(row[4])
    programs = pc_members.get(pc.author_id, [])
    programs.append(pc)
    pc_members[pc.author_id] = programs
  DB.close()
  return pc_members


def get_authors():
  db = DB.get()
  cur = db.cursor()
  cur.execute("SELECT * FROM persons")
  author_nodes = {}
  for row in cur.fetchall():
    author_node = Node()
    author_node.id = str(int(row[0]))
    author_node.name = row[1]
    author_node.type = "author"
    author_nodes[str(int(row[0]))] = author_node
  DB.close()
  return author_nodes


def get_conferences():
  db = DB.get()
  cur = db.cursor()
  cur.execute("SELECT * FROM venues WHERE is_conference=1")
  conferences = []
  for row in cur.fetchall():
    conference = Venue()
    conference.id = str(row[0])
    conference.acronym = str(row[1])
    conference.name = str(row[2])
    conference.impact = int(row[3])
    conference.is_conference = True
    conferences.append(conference)
  DB.close()
  return conferences


def get_papers():
  db = DB.get()
  cur = db.cursor()
  cur.execute("SELECT * from papers")
  rows = [row[:] for row in cur.fetchall()]
  DB.close()
  return rows


def update_citation_count():
  db = DB.get()
  cur = db.cursor()
  count = 0
  stmt = "UPDATE papers SET citation_count=%s WHERE id=%s"
  with open("data/cross_ref.pkl") as f:
    data = pkl.load(f)
    values = []
    for key, value in data.items():
      values.append((value['cited_count'], key))
      count += 1
      print("Statement : ", count)
    cur.executemany(stmt, values)
    db.commit()
  DB.close()



if __name__ == "__main__":
  # get_conferences()
  dump(file_name='data/citemap_v8.csv')
  # print(get_venues())
  # update_citation_count()
