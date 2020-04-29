import collections
from helpers import pipeline
from functools import partial

class SyntaxTree:
  def __init__(self, negation, conjunction, disjunction):
    self.__RIGHT, self.__LEFT = range(2)
    self.__Op = collections.namedtuple('Op', ['prec', 'assoc', 'arity', 'calc'])
    self.OPERATORS = self.__operatorsFactory(negation, conjunction, disjunction)
  
  def __operatorsFactory(
    self,
    negation = lambda x: x,
    conjunction = lambda x: x,
    disjunction = lambda x: x) -> dict:
      return {
      '!': self.__Op(prec=4, assoc=self.__RIGHT, arity=1, calc=negation),
      '&': self.__Op(prec=3, assoc=self.__LEFT, arity=2, calc=conjunction),
      '|': self.__Op(prec=2, assoc=self.__LEFT, arity=2, calc=disjunction)}

  def parsed_terms(self, query, adapter):
      res = []
      term = ''
      for s in query:
        if not (s in self.OPERATORS or s in "()"):
          term += s
        elif term:
          if len(term.strip()) > 0:
            res.append(adapter(term.strip()))
          term = ''
      if term:
        res.append(adapter(term.strip()))
      return res

  def __parse(self, adapter, query):
      res = []
      term = ''
      for s in query:
        if not (s in self.OPERATORS or s in "()"):
          term += s
        elif term:
          if len(term.strip()) > 0:
            res.append(adapter(term.strip()))
          term = ''
        if s in self.OPERATORS or s in "()":
          res.append(s)
      if term:
        res.append(adapter(term.strip()))
      return res


  def __greaterPrecedence(self, a, b):
    return ((self.OPERATORS[b].assoc == self.__RIGHT and
             self.OPERATORS[a].prec > self.OPERATORS[b].prec) or
            (self.OPERATORS[b].assoc == self.__LEFT and
             self.OPERATORS[a].prec >= self.OPERATORS[b].prec))

  def __shuntingYard(self, parsed):
      res = []
      stack = []
      for token in parsed:
        if token in self.OPERATORS:
          while stack and stack[-1] != "(" and not self.__greaterPrecedence(token,stack[-1]):
            res.append(stack.pop())
          stack.append(token)
        elif token == ")":
          while stack:
            x = stack.pop()
            if x == "(":
              break
            res.append(x)
        elif token == "(":
          stack.append(token)
        else:
          res.append(token)
      while stack:
        res.append(stack.pop())
      return res

  def __calc(self, polish):
      stack = []
      for token in polish:
        if token in self.OPERATORS:
          operator = self.OPERATORS[token]
          if (operator.arity == 1):
            e = stack.pop()
            stack.append(self.OPERATORS[token].calc(e))
          else:
            x = stack.pop()
            y = stack.pop()
            stack.append(self.OPERATORS[token].calc(x, y))
        else:
          stack.append(token)
      return list(stack[0])  

  def build(self, adapter):
    return pipeline(
      partial(self.__parse, adapter),
      self.__shuntingYard,
      self.__calc)