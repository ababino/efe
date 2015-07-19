#!/usr/bin/env python
"""Resuleve el ejercicio 13 de la guia 8."""
from __future__ import division

import argparse
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
from scipy import optimize
from scipy.stats import norm, chi2
from scipy.stats import kstest
from scipy import integrate
from statsmodels.distributions.empirical_distribution import ECDF
from scipy import optimize


def cramer_von_mises(data):
    ecdf = ECDF(data)
    fun= lambda x: (ecdf(x) - norm.cdf(x, 0, 2.5))**2
    data[-1] = np.inf
    data[len(data)] = np.inf
    jumps = sorted(data.unique())
    cvm = 0
    cvm = integrate.quad(fun, -np.inf, min(data))[0]
    cvm += integrate.quad(fun, min(data), max(data))[0]
    cvm += integrate.quad(fun, max(data), np.inf)[0]
    #for a, b in zip(jumps[:-1], jumps[1:]):
    #    cvm += integrate.quad(fun, a, b)[0]
    return cvm


def kolmolgorov(data):
    ecdf = ECDF(data)
    d1 = abs(ecdf(data) - norm.cdf(data, 0, 2.5))
    d2 = abs(ecdf(data) - norm.cdf(data, 0, 2.5) - 1/len(data))
    i1 = abs(d1).argmax()
    i2 = abs(d2).argmax()
    d = max(d1[i1], d2[i2])
    return d


def chi2_from_sample(data):
    histdata = np.histogram(data, 56, range=(-7, 7))
    N = sum(histdata[0])
    y_data = histdata[0]
    yerr2 = y_data * ( 1 - y_data/ N)
    x = histdata[1][:-1] + np.diff(histdata[1])/2
    norm025 = norm(0, 2.5)
    y_fit = N*(norm025.cdf(histdata[1][1:]) - norm025.cdf(histdata[1][:-1]))
    chi2_m = sum((y_data - y_fit)**2 / yerr2)
    return chi2_m, chi2.sf(chi2_m, len(x)), len(x)


def plot_hist(data):
    ax = sns.distplot(data, 56, kde=False, hist_kws={'range':(-7, 7),
                                                     'label':'Datos'})
    histdata = np.histogram(data, 56, range=(-7, 7))
    x = histdata[1][:-1] + np.diff(histdata[1])/2
    N = sum(histdata[0])
    norm025 = norm(0, 2.5)
    y_fit = N*(norm025.cdf(histdata[1][1:]) - norm025.cdf(histdata[1][:-1]))
    ax.plot(x, y_fit, label=str(sum(histdata[0])) + 'N(0, 2.5)')
    return ax


def main(args):
    data = pd.read_csv('datos-G9E13.dat',squeeze=True, header=None,
                       names=['Data'])
    cv = pd.read_csv('tabla-cvm.txt', sep='\t', skiprows=1, index_col='N')
    if 'bi' in args.items:
        print('----- Item b i-----')
        chi2_m, pval = chi2_from_sample(data[:3000])
        print('chi2={}, pval={}'.format(chi2_m, pval))
        ax = plot_hist(data[:3000])
        plt.ylabel('Cuentas')
        plt.xlabel('')
        ax.legend()
        plt.savefig('fig2.jpg')
        plt.show()
    if 'bii' in args.items:
        print('--------Item b ii-------')
        print('Tk={}, pval={}'.format(*kstest(data[:3000], norm(0, 2.5).cdf)))
        print('Tc={}'.format(cramer_von_mises(data[:3000])))
        print(cv.loc[3000])
    if 'c' in args.items:
        print('--------Item b ii-------')
        tchi2 = []
        tk = []
        tc = []
        tchi2c = []
        dfs = []
        for N in cv.index[:-2]:
            T, chi2pval, df = chi2_from_sample(data[:N])
            dfs.append(pd.DataFrame({'Eventos': [N], 'T': [T], 'Test': ['Chi2'], 'Estadistico': ['Medido']}))
            Tc = chi2.isf(0.01, df)
            dfs.append(pd.DataFrame({'Eventos': [N], 'T': [Tc], 'Test': ['Chi2'], 'Estadistico': ['Critico']}))
            T = kstest(data[:N], norm(0, 2.5).cdf)[0]
            Tc = cv.loc[N, 'T_k^{critico}']
            dfs.append(pd.DataFrame({'Eventos': [N], 'T': [T], 'Test': ['Kolmolgorov'], 'Estadistico':['Medido']}))
            dfs.append(pd.DataFrame({'Eventos': [N], 'T': [Tc], 'Test': ['Kolmolgorov'], 'Estadistico': ['Critico']}))
            T = cramer_von_mises(data[:N])
            Tc = cv.loc[N, 'T_c^{critico}']
            dfs.append(pd.DataFrame({'Eventos': [N], 'T':[T], 'Test': ['Cramer Von-Mises'], 'Estadistico':['Medido']}))
            dfs.append(pd.DataFrame({'Eventos': [N], 'T': [Tc], 'Test': ['Cramer Von-Mises'], 'Estadistico': ['Critico']}))
        df = pd.concat(dfs)
        g = sns.FacetGrid(df, col='Test', hue='Estadistico', sharey=False)
        g.map(plt.scatter, 'Eventos', 'T')
        g.map(plt.plot, 'Eventos', 'T')
        g.add_legend()
        g.set(xticks=[0, 4000, 8000])
        plt.savefig('fig3.jpg')
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Resuleve el ejercicio 13 de la guia 8.')
    parser.add_argument('items', metavar='I', type=str, nargs='+',
                        help='Los items a resolver.')
    args = parser.parse_args()
    main(args)
