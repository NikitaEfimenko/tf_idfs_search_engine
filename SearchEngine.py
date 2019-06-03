from multiprocessing import Pool
from functools import partial, reduce
from collections import defaultdict as dict
import re

def pipeline(*f):
	return lambda x: reduce(lambda y, f: f(y), f, x)

def lemmanisation(tokens):
	def lingvaf(token):
		return pipeline(
			lambda x: x.lower()
		)(token)
	return list(map(lingvaf, tokens))

class SearchEngine:
	def __init__(self, index):
		self.index = index
	
	def index(self):
		print(self.index)

	def _and(self, res1, res2):
		result = []
		i = 0
		j = 0
		if len(res1) == 0 or len(res2) == 0:
			return result
		while (i < len(res1) and j < len(res2)):
			condition = res1[i] - res2[j]
			if (condition > 0 and i < len(res1)):
				j += 1
			elif (condition < 0 and j < len(res2)):
				i += 1
			elif (condition == 0):
				result.append(res1[i])
				i += 1
				j += 1
		return result

	def _not(self, res):
		pass

	def _and(self, res1, res2):
		pass

	def searchMapper(self, query):
		terms = lemmanisation(query.split(' '))
		return list(map(lambda word: self.index[word] , terms))
	
	def andReducer(self, docs):
		return reduce(self._and, docs)
	
	def search(self, query):
		return pipeline(
			self.searchMapper,
			self.andReducer
		)(query)


def tokenisation(docs):
	formatter = ' '
	return map(lambda doc: doc.split(formatter), docs)

def buildIndex(listOfTerms):
	d = dict(list)
	for i, terms in enumerate(listOfTerms):
		for term in terms:
			d[term].append(i)
	return d
def readFile(file):
	return file

def engineBuilder(index): 
	return SearchEngine(index)

documents = ['sDf sdgd fghd s fsdss', 'sdfb sdF dhgf gjh', 'sdfb sd dhgf gjh']


booleanSearch = pipeline(
  tokenisation,
  partial(map, lemmanisation),
  buildIndex
)

fetchData = pipeline(
	readFile
)

searchEngine = pipeline(
	fetchData,
	booleanSearch,
	engineBuilder
)

engine = searchEngine(documents)


def main():
	print(engine.index)
	print(engine.search('sdf sDF'))

if __name__ == '__main__':
	main()