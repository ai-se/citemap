from __future__ import print_function, division
import sys, os
sys.path.append(os.path.abspath("."))

__author__ = "panzer"

class O:
  def __init__(self, **d): self.has().update(**d)
  def has(self): return self.__dict__
  def update(self, **d) : self.has().update(d); return self
  def __repr__(self)   :
    show=[':%s %s' % (k,self.has()[k])
      for k in sorted(self.has().keys() )
      if k[0] is not "_"]
    txt = ' '.join(show)
    if len(txt) > 60:
      show=map(lambda x: '\t'+x+'\n',show)
    return '{'+' '.join(show)+'}'
  def __getitem__(self, item):
    return self.has().get(item)
  def __setitem__(self, key, value):
    self.has()[key] = value


def say(*lst):
  print(*lst, end="")
  sys.stdout.flush()

class Paper(O):
  def __init__(self, **kwargs):
    """
    :param kwargs:
     title - str
     authors - [str]
     year - int
     venue - str
     idx - str
     abstract - str
     refs = [str]
    """
    O.__init__(self, **kwargs)

class PC(O):
  def __init__(self):
    """

    """
    O.__init__(self)
    self.author_id = None
    self.conference_id = None
    self.year = None
    self.role = None

  def set_short_role(self, role):
    if role == "General Chair":
      self.role = "GC"
    elif role == "Program Chair":
      self.role = "PC"
    elif role == "PC member main track":
      self.role = "PCM"
    else:
      raise RuntimeError("Invalid role  %s"%role)

class Conference(O):
  def __init__(self):
    """
    Initialize Conference
    """
    O.__init__(self)
    self.id = None
    self.acronym = None
    self.name = None
    self.impact = None

class Node(O):
  id = 1
  def __init__(self, **params):
    O.__init__(self, **params)
    #self.id = Node.id
    Node.id += 1

class Edge(O):
  id = 1
  def __init__(self, **params):
    O.__init__(self, **params)
    #self.id =  Edge.id
    Edge.id += 1