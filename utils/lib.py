from __future__ import print_function, division
import sys, os
sys.path.append(os.path.abspath("."))
import collections
import functools
import random

__author__ = "panzer"


class O:
  def __init__(self, **d): self.has().update(**d)

  def has(self): return self.__dict__

  def update(self, **d) : self.has().update(d); return self

  def __repr__(self)   :
    show = [':%s %s' % (k, self.has()[k])
            for k in sorted(self.has().keys())
            if k[0] is not "_"]
    txt = ' '.join(show)
    if len(txt) > 60:
      show = map(lambda x: '\t' + x + '\n', show)
    return '{' + ' '.join(show) + '}'

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


class Venue(O):
  def __init__(self):
    """
    Initialize Conference
    """
    O.__init__(self)
    self.id = None
    self.acronym = None
    self.name = None
    self.impact = None
    self.is_conference = True


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


class Memoized(object):
  '''Decorator. Caches a function's return value each time it is called.
  If called later with the same arguments, the cached value is returned
  (not reevaluated).
  '''

  def __init__(self, func):
    self.func = func
    self.cache = {}

  def __call__(self, *args):
    if not isinstance(args, collections.Hashable):
      # uncacheable. a list, for instance.
      # better to not cache than blow up.
      return self.func(*args)
    if args in self.cache:
      return self.cache[args]
    else:
      value = self.func(*args)
      self.cache[args] = value
      return value

  def __repr__(self):
    '''Return the function's docstring.'''
    return self.func.__doc__

  def __get__(self, obj, objtype):
    '''Support instance methods.'''
    return functools.partial(self.__call__, obj)


def shuffle(lst):
  if lst:
    random.shuffle(lst)
  return lst


def file_exists(file_name):
  return os.path.isfile(file_name)
