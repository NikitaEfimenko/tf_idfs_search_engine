from Metrics import Metrics
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functools import partial

def extract_marks(file):
    marks_obj = []
    lines = []
    with open(file, "r") as _file:
        lines = _file.readlines()
    for idx in range(0, len(lines), 8):
        parsing_lines = lines[idx: idx + 7]
        query = parsing_lines[0]
        my_mark = list(map(int, parsing_lines[3].split(',')))
        es_mark = list(map(int, parsing_lines[6].split(',')))
        marks_obj.append((my_mark, es_mark, query))
    return marks_obj
      
def data_to_plot_metric(queries, tensor, q_idxs, metric_type, lvl_type=0):
    result = []
    q_sample = queries[q_idxs]
    my_mark= tensor[q_idxs, 0, lvl_type, metric_type]
    es_mark= tensor[q_idxs, 1, lvl_type, metric_type]
    result = np.column_stack((q_sample, my_mark, es_mark))
    return [[r[0], float(r[1]), float(r[2])] for r in result]

def draw_plot(data1, data3, data5, title, x, y, kind="bar"):
    fig, axes = plt.subplots(nrows=1, ncols=3)
    
    df1 = pd.DataFrame(data1, columns = x + y)
    df3 = pd.DataFrame(data3, columns = x + y)
    df5 = pd.DataFrame(data5, columns = x + y)
    plt.subplots_adjust(bottom=0.5)
    df1.plot(ax=axes[0], x=x[0], y=y, kind=kind)
    axes[0].set_title('@1')
    df3.plot(ax=axes[1], x=x[0], y=y, kind=kind)
    axes[1].set_title('@3')
    df5.plot(ax=axes[2], x=x[0], y=y, kind=kind)
    axes[2].set_title('@5')
    fig.suptitle(title, fontsize=24)

if __name__ == '__main__':
    file_name = sys.argv[1]
    marks_obj = extract_marks(file_name)
    metric_tensor = np.ndarray((len(marks_obj), 2, 3, 5))
    m1 = Metrics(1)
    m3 = Metrics(3)
    m5 = Metrics(5)
    queries = np.array(list(map(lambda x: x[2], marks_obj)))
    for i, mark in enumerate(marks_obj):
      my_mark, es_mark, query = mark
      
      metric_tensor[i, 0, 0, :] = m1.measure(my_mark[:1])
      metric_tensor[i, 0, 1, :] = m3.measure(my_mark[:3])
      metric_tensor[i, 0, 2, :] = m5.measure(my_mark[:5])

      metric_tensor[i, 1, 0, :] = m1.measure(es_mark[:1])
      metric_tensor[i, 1, 1, :] = m3.measure(es_mark[:3])
      metric_tensor[i, 1, 2, :] = m5.measure(es_mark[:5])
    
    partial_data_extractor = partial( data_to_plot_metric, queries,metric_tensor, np.arange(1, 20, 2))
    
    dP1 = partial_data_extractor(0, lvl_type = 0)
    dP3 = partial_data_extractor(0, lvl_type = 1)
    dP5 = partial_data_extractor(0, lvl_type = 2)
    
    dCG1 = partial_data_extractor(1, lvl_type = 0)
    dCG3 = partial_data_extractor(1, lvl_type = 1)
    dCG5 = partial_data_extractor(1, lvl_type = 2)
    
    dDCG1 = partial_data_extractor(2, lvl_type = 0)
    dDCG3 = partial_data_extractor(2, lvl_type = 1)
    dDCG5 = partial_data_extractor(2, lvl_type = 2)
    
    dNDCG1 = partial_data_extractor(3, lvl_type = 0)
    dNDCG3 = partial_data_extractor(3, lvl_type = 1)
    dNDCG5 = partial_data_extractor(3, lvl_type = 2)
    
    dERR1 = partial_data_extractor(4, lvl_type = 0)
    dERR3 = partial_data_extractor(4, lvl_type = 1)
    dERR5 = partial_data_extractor(4, lvl_type = 2)
    
    draw_plot(
        dP1,
        dP3,
        dP5,
    title = "P", x = ["queries"], y = ["ElasticSearch", "mySearch"])
    
    draw_plot(
        dCG1,
        dCG3,
        dCG5,
    title = "CG", x = ["queries"], y = ["ElasticSearch", "mySearch"])
    
    draw_plot(
        dDCG1,
        dDCG3,
        dDCG5,
    title = "DCG", x = ["queries"], y = ["ElasticSearch", "mySearch"])
    
    draw_plot(
        dNDCG1,
        dNDCG3,
        dNDCG5,
    title = "NDCG", x = ["queries"], y = ["ElasticSearch", "mySearch"])
    
    draw_plot(
        dERR1,
        dERR3,
        dERR5,
    title = "ERR", x = ["queries"], y = ["ElasticSearch", "mySearch"])
    
    plt.show()