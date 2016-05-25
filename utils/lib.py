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