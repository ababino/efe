#!/usr/bin/env python
"""Resuleve el ejercicio 13 de la guia 8."""
import argparse
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
from scipy import optimize
from scipy.stats import norm, chi2
from __future__ import division




def main(args):
    data = pd.read_csv('datos-G9E13.dat',squeeze=True, header=None,
                       names=['Data'])
    histdata = np.histogram(data[:3000], 56, range=(-7, 7))
    ax = sns.distplot(data[:3000], 56, kde=False, hist_kws={'range':(-7, 7)})
    N = sum(histdata[0])
    y_data = histdata[0]
    yerr2 = y_data * ( 1 - y_data/ N)
    x = histdata[1][:-1] + np.diff(histdata[1])/2
    norm025 = norm(0, 2.5)
    y_fit = N*(norm025.cdf(histdata[1][1:]) - norm025.cdf(histdata[1][:-1]))
    ax.plot(x, y_fit)
    chi2_m = sum((y_data - y_fit)**2 / yerr2)
    print(chi2_m, chi2.sf(chi2_m, len(x)))
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Resuleve el ejercicio 13 de la guia 8.')
    parser.add_argument('items', metavar='I', type=str, nargs='+',
                        help='Los items a resolver.')
    args = parser.parse_args()
    main(args)
