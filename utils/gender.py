from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

import requests
import json
from db import mysqldb as mysql
import gender_guesser.detector
from utils.lib import O
import cPickle as pkl


__author__ = "bigfatnoob"

US_GENDER_FILE = "data/names_gender.pkl"
INDIAN_GENDER_FILE = "data/indian_names_gender.pkl"
US_STATES_GENDER_FILE = "data/us_states_gender.pkl"


def fetch_db(pkl_file):
  with open(pkl_file) as f:
    db = pkl.load(f)
    return db


US_GENDER_DB = fetch_db(US_GENDER_FILE)
INDIAN_GENDER_DB = fetch_db(INDIAN_GENDER_FILE)
US_STATES_GENDER_DB = fetch_db(US_STATES_GENDER_FILE)


def from_db(name):
  node = US_GENDER_DB.get(name, None)
  if node is None:
    node = INDIAN_GENDER_DB.get(name, None)
  if node is None:
    return US_STATES_GENDER_DB.get(name, None)
  if node.males < node.females:
    return 'female'
  elif node.males > node.females:
    return 'male'
  return None


def get_first_name(name):
  parts = name.split()
  n = None
  for part in parts:
    n = part
    if n[-1] == ".":
      n = n[:-1]
    if len(n) > 1:
      break
  return n.lower().split("-")[0]


def identify():
  d = gender_guesser.detector.Detector(case_sensitive=False)
  authors = mysql.get_authors()
  counts = {}
  for a_id, node in authors.items():
    if node.name:
      f_name = get_first_name(node.name)
      g = from_db(f_name)
      if g is None:
        g = d.get_gender(f_name)
      counts[g] = counts.get(g, []) + [node]
      node.f_name = f_name
  author_genders = []
  for key, nodes in counts.items():
    if "female" in key:
      gender = "f"
    elif "male" in key:
      gender = "m"
    else:
      continue
    for node in nodes:
      author_genders.append((gender, int(node.id)))
  # flattened = counts['unknown'] + counts['andy']
  # nodes = []
  mysql.update_genders(author_genders)


def from_api(names):
  url = ""
  cnt = 0
  if not isinstance(names, list):
    names = [names, ]

  for name in names:
    if url == "":
      url = "name[0]=" + name
    else:
      cnt += 1
      url = url + "&name[" + str(cnt) + "]=" + name

  req = requests.get("https://api.genderize.io?" + url)
  results = json.loads(req.text)

  retrn = []
  for result in results:
    if result["gender"] is not None:
      # retrn.append((result["gender"], result["probability"], result["count"]))
      retrn.append(result["gender"])
    else:
      # retrn.append((None, u'0.0', 0.0))
      retrn.append(None)
  return retrn


def make_name_db():
  root_folder = "data/us_names"
  pkl_file = US_GENDER_FILE
  db = {}
  for f_name in os.listdir(root_folder):
    f_name = "%s/%s" % (root_folder, f_name)
    with open(f_name) as f:
      print(f_name)
      for line in f.readlines():
        [name, gender, count] = line.split(",")
        name = name.lower()
        node = db.get(name, None)
        if node is None:
          node = O()
          node.name = name
          node.females = 0
          node.males = 0
        if gender == 'F':
          node.females += int(count)
        elif gender == 'M':
          node.males += int(count)
        db[name] = node
  with open(pkl_file, "wb") as f:
    pkl.dump(db, f, pkl.HIGHEST_PROTOCOL)
  return db


def make_indian_name_db():
  def split(l):
    splits = l.split()
    return int(splits[2]), int(splits[3]), splits[4].lower()

  inp_file = "data/ind_names.txt"
  db = {}
  with open(inp_file) as f:
    index = 0
    for line in f.readlines():
      index += 1
      if index % 1000 == 0:
        print("Line : %d", index)
      males, females, name = split(line)
      node = db.get(name, None)
      if node is None:
        node = O()
        node.name = name
        node.females = 0
        node.males = 0
      node.females += females
      node.males += males
      db[name] = node
  pkl_file = INDIAN_GENDER_FILE
  with open(pkl_file, "wb") as f:
    pkl.dump(db, f, pkl.HIGHEST_PROTOCOL)
  return db


def make_us_states_name_db():
  def split(l):
    splits = l.split(",")
    return splits[1], splits[3].lower()

  root_folder = "data/us_states"
  pkl_file = US_STATES_GENDER_FILE
  db = {}
  for f_name in os.listdir(root_folder):
    f_name = "%s/%s" % (root_folder, f_name)
    with open(f_name) as f:
      print(f_name)
      for line in f.readlines():
        gender, name = split(line)
        node = db.get(name, None)
        if node is None:
          node = O()
          node.name = name
          node.females = 0
          node.males = 0
        if gender == 'F':
          node.females += 1
        elif gender == 'M':
          node.males += 1
        db[name] = node
  with open(pkl_file, "wb") as f:
    pkl.dump(db, f, pkl.HIGHEST_PROTOCOL)
  return db


if __name__ == "__main__":
  identify()
  # make_us_states_name_db()
  # make_indian_name_db()
  # make_name_db()
  # print(len(fetch_db()))
