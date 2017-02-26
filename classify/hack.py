from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

import requests
from bs4 import BeautifulSoup
from utils.lib import O


SEPERATOR = "$|$"

COOKIE = "g=DkYf04vK5NTlEgNIg3Ig; cool1=zUcwPOePMVabIhJiyo8f; cool2=OZkWCKyhip5VtBEks1pQ"
BASE_URL = "https://easychair.org/conferences"
FILE_NAME = "classify/data.csv"

CONFERENCES = [("MSR", 2015, "https://easychair.org/conferences/submission_show_all.cgi?a=7612389"),
               ("FSE", 2015, "https://easychair.org/conferences/submission_show_all.cgi?a=8115951"),
               ("ICSE", 2015, "https://easychair.org/conferences/submission_show_all.cgi?a=5178156")]

HEADERS = {
    "host": "easychair.org",
    "Connection": "keep-alive",
    "Cache-Control": "max-age=0",
    "Upgrade-Insecure-Requests": "1",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_1) AppleWebKit/537.36 (KHTML, like Gecko)\
     Chrome/55.0.2883.95 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Referer": "https://easychair.org/my/roles.cgi?a=12766663",
    "Accept-Encoding": "gzip, deflate, sdch, br",
    "Accept-Language": "en-GB,en-US;q=0.8,en;q=0.6",
    "Cookie": COOKIE
}


class Submission(O):
  def __init__(self, **kwargs):
    self.title = None
    self.keywords = None
    self.abstract = None
    self.category = None
    self.decision = "reject"
    self.conference = None
    self.year = None
    self.authors = None
    O.__init__(self, **kwargs)

  def add_keyword(self, keyword):
    if self.keywords is None:
      self.keywords = []
    self.keywords.append(keyword)

  def add_author(self, f_name, l_name):
    if self.authors is None:
      self.authors = []
    self.authors.append("%s %s" % (f_name, l_name))

  @staticmethod
  def make_header():
    return SEPERATOR.join(["Conference", "Year", "Title", "Authors", "Keywords", "Abstract", "Category", "Decision"])

  def to_csv(self):
    return SEPERATOR.join([self.conference, str(self.year), self.title, ",".join(self.authors), ",".join(self.keywords),
                           self.abstract, str(self.category), self.decision]).encode("utf8")


def parse_conference(name, year, url, fil):
  resp = requests.get(url, headers=HEADERS)
  if resp.status_code != 200:
    print("Failed to retrieve conference. Status Code: %s" % resp.status_code)
    return
  soup = BeautifulSoup(resp.content)
  table = soup.find("table", id="ec:table1")
  header = table.thead.find_all("tr")[-1]
  info_index = 0
  for th in header.find_all("th"):
    if th and th.img and th.img['alt'] == 'information':
      break
    info_index += 1
  print(info_index)
  tbody = None
  for row in table.children:
    if row.name == 'tbody':
      tbody = row
      break
  for i, row in enumerate(tbody.find_all("tr")):
    print(i)
    info_link = "%s/%s" % (BASE_URL, row.find_all("td")[info_index].find("a")["href"])
    submission = parse_submission(info_link)
    if submission is None:
      continue
    submission.conference = name
    submission.year = year
    fil.write(submission.to_csv() + "\n")
  print("DONE PARSING")


def parse_submission(submission_link):
  sub = requests.get(submission_link, headers=HEADERS)
  if sub.status_code != 200:
    print("Failed to retrieve submission. Status Code: %s" % sub.status_code)
    return None
  soup = BeautifulSoup(sub.content)
  table = soup.find(id="ec:table1")
  submission = Submission()
  for row in table.tbody.find_all("tr"):
    cells = row.find_all("td")
    if len(cells) != 2:
      continue
    text = cells[0].get_text().split(":")[0]
    if text == "Title":
      submission.title = cells[1].get_text().strip()
    elif text == "Author keywords":
      for div in cells[1].children:
        submission.add_keyword(div.get_text().strip())
    elif text == "Abstract":
      submission.abstract = cells[1].get_text(strip=True).strip()
    elif text == "Category":
      submission.category = cells[1].get_text().strip()
    elif text == "Decision":
      value = cells[1].get_text().strip().lower()
      submission.decision = "accept" if "accept" in value else "reject"
  author_table = soup.find(id="ec:table2")
  for row in author_table.tbody.find_all("tr")[2:]:
    cells = row.find_all("td")
    f_name = cells[0].get_text().strip().lower()
    l_name = cells[1].get_text().strip().lower()
    submission.add_author(f_name, l_name)
  return submission


def conference_parser(conferences):
  header = Submission.make_header()
  with open(FILE_NAME, 'wb') as f:
    f.write(header + "\n")
    for c in conferences:
      parse_conference(c[0], c[1], c[2], f)

if __name__ == "__main__":
  conference_parser(CONFERENCES)
