from __future__ import print_function, division
import sys
import os

sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True
from db.mysql import get_papers
from urlparse import urlparse
import requests
from bs4 import BeautifulSoup
import cPickle as pkl
import time

__author__ = "bigfatnoob"

INVALID_SOURCES = ["com", "computer", "ieee"]
NAMESPACES = {'doi':'http://www.crossref.org/qrschema/3.0'}

def make_doi_pkl():
  papers = get_papers()
  source_map = {}
  pkl_map = {}
  for paper in papers:
    print("Paper ID:", paper[0])
    doi_url = paper[-2]
    if not doi_url: continue
    source, doi = process_url(doi_url)
    if source in INVALID_SOURCES:
      continue
    else:
      from_doi = fetch_from_crossref(doi)
    if from_doi:
      pkl_map[paper[0]] = from_doi
    cnt = source_map.get(source, 0)
    cnt += 1
    source_map[source] = cnt
  print(source_map)
  with open("data/cross_ref.pkl", "wb") as f:
    pkl.dump(pkl_map, f, pkl.HIGHEST_PROTOCOL)


MAX_RETRIES = 5


def fetch_from_crossref(doi, retries=0):
  if retries >= MAX_RETRIES:
    raise Exception("Exceeded max run times.")
  try:
    url = "http://api.crossref.org/works/%s.xml" % doi
    response = requests.get(url)
    if response.status_code == 404:
      return None
    elif response.status_code != 200:
      print("Unknown response: %d" % response.status_code)
      print(response.content)
    soup = BeautifulSoup(response.content, 'lxml')
    cited_node = soup.find("crm-item", attrs={"name": "citedby-count"})
    cited_count = None
    if cited_node is not None:
      cited_count = int(cited_node.get_text())
    abstract_node = soup.find("jats:abstract")
    abstract = None
    if abstract_node is not None:
      abstract = " ".join(abstract_node.get_text().split()).decode('utf-8', 'ignore').encode("utf-8")
    from_crossref = {
      "cited_count": cited_count,
      "abstract": abstract
    }
    return from_crossref
  except:
    print("Sleeping for a minute")
    time.sleep(60)
    return fetch_from_crossref(doi, retries+1)


# ACM_HEADERS = {
#   "Host": "doi.acm.org",
#   "Connection": "keep-alive",
#   "Upgrade-Insecure-Requests": "1",
#   "User-Agent": "Mozilla/5.0 (X11; U; Linux i686) Gecko/20071127 Firefox/2.0.0.11",
#   "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
#   "Accept-Encoding": "gzip, deflate, sdch",
#   "Accept-Language": "en-US,en;q=0.8",
#   "Cookie": "_ga=GA1.2.1873985683.1490550293"
# }
# headers = {"User-Agent":"Mozilla/5.0 (X11; U; Linux i686) Gecko/20071127 Firefox/2.0.0.11"}

# def fetch_from_acm(url):
#   s = requests.Session()
#   s.headers = ACM_HEADERS
#   s.max_redirects = 1000
#   response = s.get(url)
#   print(response.content)


def process_url(url):
  parsed = urlparse(url)
  source = parsed.netloc.split(".")[-2]
  doi = "/".join(parsed.path.split("/")[-2:])
  return source, doi


def read_pkl():
  with open("data/cross_ref.pkl") as f:
    data = pkl.load(f)
    print(sum([1 if value['abstract'] else 0 for value in data.values()]))

if __name__ == "__main__":
  # fetch_from_crossref("10.1109/ICSE.2015.24")
  # print(fetch_from_crossref("10.1109/TASC.2010.2088091"))
  # fetch_from_acm("http://doi.acm.org/10.1145/2884781.2884828")
  # make_doi_pkl()
  read_pkl()
