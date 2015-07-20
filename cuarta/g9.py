#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Resuleve el ejercicio 13 de la guia 8."""
from __future__ import division
from __future__ import unicode_literals

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


def my_hist(data, bins, err_type='poisson', **kwargs):
	"""Histogram with poissonian error bars."""
	y, bin_edges = np.histogram(data, bins=bins)
	normalization = sum(y*np.diff(bin_edges))
	if err_type == 'poisson':
		yerr = np.sqrt(y) / normalization
	elif err_type == 'binomial':
		yerr = np.sqrt(y * ( 1 - y / normalization) / normalization)
	y = y.astype(np.float) / normalization
	ax = plt.bar(bin_edges[:-1], y, yerr=yerr, width=1.0, ecolor='r', **kwargs)
	return ax


def cramer_von_mises(data):
    ecdf = ECDF(data)
    fun= lambda x: (ecdf(x) - norm.cdf(x, 0, 2.5))**2
    data[-1] = np.inf
    data[len(data)] = np.inf
    jumps = sorted(data.unique())
    cvm = 0
    cvm = integrate.quad(fun, -np.inf, min(data))[0]
    #cvm += integrate.quad(fun, min(data), max(data))[0]
    cvm += integrate.quad(fun, max(data), np.inf)[0]
    for a, b in zip(jumps[:-1], jumps[1:]):
        cvm += integrate.quad(fun, a, b)[0]
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


def plot_hist(data, nbins=56):
    ax = sns.distplot(data, nbins, kde=False, hist_kws={'range':(-7, 7),
                                                     'label':'Datos'})
    histdata = np.histogram(data, nbins, range=(-7, 7))
    x = histdata[1][:-1] + np.diff(histdata[1])/2
    yerr = np.sqrt(histdata[0] * (1- histdata[0]/sum(histdata[0])))
    ax.errorbar(x, histdata[0], yerr=yerr, fmt='.')
    N = sum(histdata[0])
    norm025 = norm(0, 2.5)
    y_fit = N*(norm025.cdf(histdata[1][1:]) - norm025.cdf(histdata[1][:-1]))
    ax.plot(x, y_fit, label='Nº de Eventos Esperados')
    return ax


def ej13c(cv, data):
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
    print(df[(df['Estadistico']=='Medido') & (df['Test']=='Cramer Von-Mises')])
    plt.show()

def ej14(data_x, data_y, s):
    a2, a1 = np.polyfit(data_x, data_y, 1)
    chi2_t_list = []
    chi2_f_list = []
    for _ in range(1000):
    	new_y = np.random.normal(a1 + a2*data_x, s)
        p = np.polyfit(data_x, new_y, 1)
        chi2_t_list.append(sum(((new_y -a1 - a2*data_x)/s)**2))
        chi2_f_list.append(sum(((new_y -p[1] - p[0]*data_x)/s)**2))
    #ax = sns.distplot(chi2_list, kde=False, norm_hist=True, label='Datos')
    plt.figure(1)
    ax = my_hist(chi2_t_list,20, label='Datos')
    x = np.linspace(0, 30,100)
    chi29 = chi2.pdf(x, 11)
    plt.plot(x, chi29, label='Chi2(11)')
    plt.legend()
    plt.figure(2)
    ax = my_hist(chi2_f_list,20, label='Datos')
    x = np.linspace(0, 30,100)
    chi211 = chi2.pdf(x, 11)
    plt.plot(x, chi211, label='Chi2(11)')
    chi29 = chi2.pdf(x, 9)
    plt.plot(x, chi29, label='Chi2(9)')

    plt.legend()


    plt.show()

def main(args):
    if '13' in ''.join(args.items):
        data = pd.read_csv('datos-G9E13.dat',squeeze=True, header=None,
                           names=['Data'])
        cv = pd.read_csv('tabla-cvm.txt', sep='\t', skiprows=1, index_col='N')
    if '13bi' in args.items:
        np.set_printoptions(precision=3)
        print('----- 13 Item b i-----')
        chi2_m, pval, df = chi2_from_sample(data[:3000])
        print('chi2={}, pval={}'.format(chi2_m, pval))
        ax = plot_hist(data[:3000])
        plt.ylabel('Cuentas')
        plt.xlabel('')
        ax.legend()
        #plt.savefig('fig2.jpg')
        plt.show()
    if '13bii' in args.items:
        print('--------13 Item b ii-------')
        print('Tk={}, pval={}'.format(*kstest(data[:3000], norm(0, 2.5).cdf)))
        print('Tc={}'.format(cramer_von_mises(data[:3000])))
        print(cv.loc[3000])
    if '13c' in args.items:
        print('--------13 Item c-------')
        ej13c(cv, data)
    if '14a' in args.items:
    	x = np.array([2.00, 2.10, 2.20, 2.30, 2.40, 2.50, 2.60, 2.70, 2.80, 2.90, 3.00])
    	y = np.array([2.78, 3.29, 3.29, 3.33, 3.23, 3.69, 3.46, 3.87, 3.62, 3.40, 3.99])
    	s = 0.3
        ej14(x, y, s)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Resuleve el ejercicio 13 de la guia 8.')
    parser.add_argument('items', metavar='I', type=str, nargs='+',
                        help='Los items a resolver.')
    args = parser.parse_args()
    main(args)
