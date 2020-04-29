from multiprocessing import Pool
from functools import partial, reduce
from collections import defaultdict as dict
from numpy import dot
from numpy.linalg import norm
import numpy as np
import os
import pickle
from collections import Counter
import gc
import sys
from nltk.corpus import stopwords

from SearchIndex import tf_func, idf_func, \
    tokenization, buildBooleanIndex, frequencyRelevance, reduceStopWords

from argparse import ArgumentParser

from helpers import Page, pipeline, logging
from ShuntingYard import SyntaxTree

from nltk.stem import WordNetLemmatizer

DATA_DIR = "data"
DIRECT_INDEX_PATH = 'direct_index'

lemma = WordNetLemmatizer()

class SearchEngine:
    def __init__(self, index):
        self.index = index[0]
        self.raw_len = index[1]
        self.stopTerms = stopwords.words('english')
        self.tree = SyntaxTree(
            self.__neg,
            self.__conj,
            self.__disj)
        self.evaluteQuery = self.tree.build(self.__adapter)
        self.wholeSet = set(range(self.raw_len))
    def __tkn_preprocessing(self, tkn):
        return lemma.lemmatize(tkn.lower())
    def __adapter(self, tkn):
        preprocessed = self.__tkn_preprocessing(tkn)
        return tuple(list(map(lambda x: x[0], self.index[preprocessed][1])))  if (preprocessed in self.index.keys()) else tuple(self.wholeSet)

    def __conj(self, s1, s2):
        return set(s1) & set(s2)

    def __neg(self, s):
        return self.wholeSet - set(s)

    def __disj(self, s1, s2):
        return set(s1) | set(s2)

    def __build_vector_from_query(self, terms):
        count_dict = Counter(terms)
        tf = list(map(lambda term: tf_func(count_dict[term]), terms))
        idfs = [self.index[term][0] if self.index[term] else 1. for term in terms]
        return np.array(tf) * np.array(idfs)
        
    def __build_vectors_from_result(self, terms, result):
        def tf(i):
            res = list(map(lambda v: 1., terms))
            for idx,term in enumerate(terms):
                if self.index[term]:
                    for pair in self.index[term][1]:
                        if pair[0] == i:
                            res[idx] = pair[1]
            return res
        idfs = [self.index[term][0] if self.index[term] else 1. for term in terms]
        return [(i, np.array(tf(i)) * np.array(idfs)) for i in result]

    def __ranging(self, query, result_list):
        def cosine_similarity(v1, v2):
            return dot(v1, v2)
        terms = list(self.tree.parsed_terms(query, self.__tkn_preprocessing))
        
        q_vect = self.__build_vector_from_query(terms)
        res_vectors = self.__build_vectors_from_result(terms, result_list)
        
        return sorted([(i, cosine_similarity(q_vect, v)) for i, v in res_vectors], key=lambda kv: kv[1], reverse=True)


    def search(self, query):
        return pipeline(
            self.evaluteQuery,
            partial(self.__ranging, query)
        )(query)

def _init_argparser() -> ArgumentParser:
    text = "--tokenization - токенизация\n" + \
            "--build - построить индекс\n" + \
            "--pure - без ранжирования\n"
    parser = ArgumentParser(description=text)
    
    parser.add_argument(
        "--tokenization", "-t",
        help=f"Build tokenization block and save in data/tokenization",
        action="store_true")
    parser.add_argument(
        "--tiny", "-tiny",
        help=f"Build tokenization block and save in data/tokenization",
        action="store_true")
    parser.add_argument(
        "--pure", "-p",
        help=f"Modify all engine blocks as pure boolean index/search",
        action="store_true")
    parser.add_argument(
        "--build", "-i",
        help="Build invert index block and save in data/invert_index.out",
        action="store_true")
    return parser

@logging("save in data/...")
def save(data, path):
    gc.collect()
    with open(os.path.join(DATA_DIR, path), "wb") as file:
        pickle.dump(data, file)

@logging('loading direct index ...')
def extract_content():
    with open(os.path.join(DATA_DIR, DIRECT_INDEX_PATH), "rb") as db:
        pages = pickle.load(db)
    print(len(pages))
    return list(map(lambda page: page.title, pages))

@logging('loading tokens ...')
def extract_tokenize():
    TOKENIZE_PATH = 'tokenization'
    with open(os.path.join(DATA_DIR, TOKENIZE_PATH), "rb") as db:
        tokens = pickle.load(db)
    return tokens

@logging('loading tf-idf index...')
def extract_tf_idf_index():
    INDEX_PATH = 'invert_index_tf-idf'
    with open(os.path.join(DATA_DIR, INDEX_PATH), "rb") as db:
        index = pickle.load(db)
    return index

@logging('loading boolean index...')
def extract_boolean_index():
    INDEX_PATH = 'invert_index_boolean'
    with open(os.path.join(DATA_DIR, INDEX_PATH), "rb") as db:
        index = pickle.load(db)
    return index


if __name__ == '__main__':
    engine = None
    query = None
    contents = None

    args = _init_argparser().parse_args()
    with_ranging = False if args.pure else True

    if args.tokenization:
        contents = extract_content()
        tokenize = pipeline(
            tokenization,
            partial(map, partial(map, lambda tkn: lemma.lemmatize(tkn.lower()))),
            reduceStopWords,
        ) if not args.pure else tokenization
        tokens = list(tokenize(contents))
        save(tokens, "tokenization_tiny")
    elif args.build and not with_ranging:
        tokens = extract_tokenize()
        index = buildBooleanIndex(tokens)
        save(index, "invert_index_boolean")
    elif args.build and with_ranging:
        bool_index = extract_boolean_index()
        index = frequencyRelevance(bool_index)
        save(index, "invert_index_tf-idf")
    else:
        contents = extract_content()
        extracter = extract_tf_idf_index if with_ranging else extract_boolean_index 
        index = extracter()
        engine = SearchEngine(index)
        while(True):
            query = input('query: ')
            res = list(map(lambda x: x[0], engine.search(query)[:5]))
            print(res)
            print(list(map(lambda i: contents[i], res)))