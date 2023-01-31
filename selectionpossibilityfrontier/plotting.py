import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def plot_frontier(ds, qs, dlabel='Diversity', qlabel='Observed Quality', title='Selection Possibility Frontier', lims=False):
    plt.scatter(qs, ds)
    if lims:
        plt.xlim([0, 1])
        plt.ylim([0, 1])
    plt.xlabel(qlabel)
    plt.ylabel(dlabel)
    plt.title(title)
    plt.show()
