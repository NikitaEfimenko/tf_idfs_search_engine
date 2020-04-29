#! /evn/bin/python3.6
import pickle
import wikipediaapi
from helpers import set_interval
import sys
from helpers import Page

wiki_wiki = wikipediaapi.Wikipedia('en')
TINY = False

count = 0
speed = 0

def getPageFromCategory(categorymembers, level=0, max_level=3):
        global count
        global speed
        store = []
        for idx in categorymembers:
            c = categorymembers[idx]
            if (count >= 25000):
                continue
            if c.ns != wikipediaapi.Namespace.CATEGORY:
                count += 1
                speed += 1
                store.append(Page(c.title, sys.intern(c.text)))
            elif level < max_level:
                store.extend(getPageFromCategory(c.categorymembers, level=level + 1))
        return store

def logger():
    global count
    global speed
    print("fetch: {}/{} docs...".format(count, speed))
    speed = 0

if __name__ == '__main__':
    category = 'Category:Computer programming'
    cat = wiki_wiki.page(category)
    pages = []
    print('fetching...')
    timer = set_interval(logger, 10)
    pages = getPageFromCategory(cat.categorymembers)
    timer.cancel()
    print('Fetch done: {} documents'.format(len(pages)))
    with open("data/direct_index" + ("_tiny" if TINY else ""), "wb") as db:
        pickle.dump(pages, db)
        print('save!')
        exit(0)
