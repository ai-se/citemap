from __future__ import print_function, division
import os, sys
sys.path.append(os.path.abspath("."))
from itertools import groupby
import unicodedata
import string
import warnings
from unicoder import UTF8Recoder
from db import mongo
from utils.lib import Paper

warnings.filterwarnings("ignore")

__author__ = "panzer"

def parse_acm_generate(file_name):
  with open(file_name, 'rb') as f:
    f = UTF8Recoder(f)
    count = 0
    for key, group in groupby(f, key=lambda l: l.strip(' \n\r') == ''):
      paper = Paper()
      refs = []
      if key: continue
      for item in group:
        item = item.strip(' \r\n')
        if item.startswith('#*'):
          paper.title = de_punctuate(item[2:])
        elif item.startswith('#@'):
          paper.authors = remove_accents(item[2:]).split(',')
        elif item.startswith('#t'):
          paper.year = int(item[2:])
        elif item.startswith('#c'):
          paper.venue = item[2:]
        elif item.startswith('#index'):
          paper.id = item[6:]
        elif item.startswith('#!'):
          paper.abstract = item[2:]
        elif item.startswith('#%'):
          refs.append(item[2:])
      paper.refs = refs
      count+=1
      yield paper

def parse_acm(filename):
  paper_map = {}
  for paper in parse_acm_generate(filename):
    paper_map[paper.title] = paper
  return paper_map


def insert_papers(filename):
  batch = []
  batch_no=0
  for paper in parse_acm_generate(filename):
    batch.append(paper.has())
    if len(batch) == 1000:
      mongo.insert_papers(batch)
      batch = []
      batch_no += 1
      print("Batch %d"%batch_no)
  if batch:
    mongo.insert_papers(batch)


def de_punctuate(s):
  return reduce(lambda s1, c: s1.replace(c, ''), string.punctuation, s).strip()

def remove_accents(s):
  if isinstance(s, unicode):
    return ''.join(x for x in unicodedata.normalize('NFKD', s) if x in string.ascii_letters).lower().strip()
  else:
    return s.lower().strip()

insert_papers("acm/acm.txt")