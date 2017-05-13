from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

__author__ = "bigfatnoob"

from community import truther


def gen_ganaxis(inp, out):
  venue_edge_map, node_map = truther.read(inp)
  edge_map = {}
  for venue, edges in venue_edge_map.items():
    for edge in edges:
      key = edge.source + "-" + edge.target
      edge_map[key] = edge_map.get(key, 0) + 1
  with open(out, "wb") as f:
    for key, val in edge_map.items():
      [source, target] = key.split("-")
      f.write("%s %s %d\n" % (source, target, val))


if __name__ == "__main__":
  gen_ganaxis(truther.CITEMAP_FILE, "community/citemap.ipairs")
