from functools import wraps, reduce
from time import time
import threading
from bisect import bisect_left

def bi_contains(lst, item):
    return (item <= lst[-1]) and (lst[bisect_left(lst, item)] == item)

def set_interval(func, sec):
    def func_wrapper():
        set_interval(func, sec)
        func()
    t = threading.Timer(sec, func_wrapper)
    t.start()
    return t

class Page:
    def __init__(self, title, content):
        self.title = title
        self.content = content


def logging(start):
    def inner_function(f):
        def timed(*args, **kw):
            print(start)
            ts = time()
            result = f(*args, **kw)
            te = time()
            print("  time=[{0:.2f}ms]".format((te - ts) * 1000))
            return result
        return timed
    return inner_function


def pipeline(*f):
    return lambda x: reduce(lambda y, f: f(y), f, x)
