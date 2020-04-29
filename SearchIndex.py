
import math
import re
from collections import defaultdict as dict
from helpers import pipeline, logging
from collections import Counter
from nltk.corpus import stopwords
from num2words import num2words
from helpers import set_interval

def tf_func(freq):
    return 1. + math.log10(freq)
def idf_func(all_doc_count, doc_count_with_term):
    return math.log10(1. + (all_doc_count / float(doc_count_with_term)))


def compute_tf(terms):
    return list(map(lambda v: (v[0], tf_func(int(v[1]))), Counter(terms).items()))

def compute_idf(coef1, term, index):
    return idf_func(coef1, len(index[term]))


@logging('Tokenization...')
def tokenization(docs):
    pattern = r"\w+(?:'\w+)?|[^\w\s]"
    return map(lambda doc: re.findall(pattern, doc), docs)

@logging('Build boolean index...')
def buildBooleanIndex(listOfTerms):
    draftIndex = dict(list)
    for i, terms in enumerate(listOfTerms):
        for term in terms:
            draftIndex[term].append(i)
    return (draftIndex, len(listOfTerms))

@logging('Build index with tf-idf extension...')
def frequencyRelevance(pair, with_logger = False):
    def logger():
        print("completed by {:.2f}%...".format(idx / len(index) * 100.))
    complete = 0
    idx = 0
    if with_logger:
        timer = set_interval(logger, 10)
    (index, coef1) = pair
    for token in index:
        idx += 1
        index[token] = (compute_idf(coef1, token, index), compute_tf(index[token]))
    return (index, coef1)

@logging('Reduce stop words...')
def reduceStopWords(listOfTerms):
    stopTerms = stopwords.words('english')
    listOfTerms = list(listOfTerms)
    for i, terms in enumerate(listOfTerms):
        listOfTerms[i] = [term  for term in terms if (term not in stopTerms) and len(term) > 1]
    return listOfTerms
