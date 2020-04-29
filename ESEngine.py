from elasticsearch import Elasticsearch
from argparse import ArgumentParser
from helpers import Page, pipeline, logging
from functools import partial, reduce
import pickle
import sys
import os

INDEX = 'docs'
INDEX_TYPE = 'docs_type'
DATA_DIR = "data"
DIRECT_INDEX_PATH = 'direct_index'
conf = {'host': 'localhost', 'port': 9200}


def _init_argparser() -> ArgumentParser:
    text = "--build - построить индекс\n" + \
        "--search - поиск\n"
    parser = ArgumentParser(description=text)

    parser.add_argument(
        "--build", "-b",
        help=f"Build tokenization block and save in data/tokenization",
        action="store_true")
    parser.add_argument(
        "--search", "-s",
        help=f"Build tokenization block and save in data/tokenization",
        action="store_true")
    return parser


@logging('loading direct index ...')
def extract_corpus():
    with open(os.path.join(DATA_DIR, DIRECT_INDEX_PATH), "rb") as db:
        pages = pickle.load(db)
    print(len(pages))
    return pages


def es_adapter(es_res):
    hit = es_res
    return (int(hit['_source']['id']), hit['_score'])


if __name__ == '__main__':
    corpus = []
    args = _init_argparser().parse_args()
    es = Elasticsearch([conf])

    def search(q):
        return es.search(index=INDEX, doc_type=INDEX_TYPE, body={
        'query': {
            'match': {
                'content': q
            }
        }
    })
    corpus = extract_corpus()
    if args.build:
        for id, doc in enumerate(corpus):
            es.index(index='docs', doc_type='docs_type', id=id, body={
                'id': id,
                'title': doc.title,
                'content': doc.content
            })
    elif args.search:
        while(True):
            query = input('query: ')
            res = list(map(es_adapter, search(query)['hits']['hits']))[:5]
            print(list(map(lambda i: i[0], res)))
            print(list(map(lambda i: corpus[i[0]].title, res)))

