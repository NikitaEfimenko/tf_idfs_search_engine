import pickle
import os
import sys
import tkinter
import math
import matplotlib.pyplot as plt
import matplotlib
from functools import reduce
from collections import Counter
import numpy as np

matplotlib.use("TkAgg")

TOKENS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '/data/tokenization')

def get_sorted_freq(tokens_freq):
    tokens_list = [(token, freq) for token, freq in tokens_freq.items()]
    return sorted(tokens_list, key=lambda x: x[1], reverse=True)

def get_plot_data(sorted_tokens):
    return [(rang, freq[1]) for rang, freq in enumerate(sorted_tokens, 1)]


def calc_tcipfa(path):
    tokens = []
    tokens_freq = {}
    with open(path, 'rb') as input_:
        tokens = pickle.load(input_)
    for token in tokens:
        for term in token:
            if term in tokens_freq:
                tokens_freq[term] += 1
            else:
                tokens_freq[term] = 1
    sorted_freq = get_sorted_freq(tokens_freq)
    plot_data = get_plot_data(sorted_freq)
    x_arr = [x for x, _ in plot_data]
    y_arr = [y for _, y in plot_data]
    print(len(x_arr), "terms")
    for i in [0, 1, 2, 3, 5, 99, 999]:
        print("rank #{}".format(i + 1), sorted_freq[i])
    x_tcipfa = x_arr
    y_tcipfa = [float(y_arr[0] / i) for i in x_tcipfa]
    plt.plot(x_arr[:200], y_arr[:200])
    plt.xlabel("Ранг")
    plt.ylabel('Частота')
    plt.title('Корпус (200 термов)')
    plt.plot(x_tcipfa[:200], y_tcipfa[:200])
    plt.title('Закон Ципфа (200 термов)')
    plt.show()

    x_arr = np.log(x_arr)
    y_arr = np.log(y_arr)
    plt.plot(x_arr, y_arr)
    plt.xlabel("log ранг")
    plt.ylabel('log частота')
    plt.title('Корпус с логарифмической шкалой')
    
    plt.plot(np.log(x_tcipfa), np.log(y_tcipfa))
    plt.title('Закон Ципфа с логарифмической шкалой')
    plt.show()


if __name__ == "__main__":
    calc_tcipfa(sys.argv[1] if sys.argv[1] else TOKENS_PATH)