#!/usr/bin/env python
"""Resuleve el ejercicio 15 de la guia 2."""
import random
from matplotlib import pyplot as plt
import numpy as np
from scipy.special import binom
import argparse


def lin_fit(x, y, s):
	Cov = np.zeros((2, 2))
	N = len(x)
	sqsum = sum([xi**2 for xi in x])
	xsum = sum(x)
	ysum = sum(y)
	xysum = sum([xi*yi for xi, yi in zip(x, y)])
	Delta = N * sqsum - xsum**2
	a1 = (sqsum * ysum -xsum * xysum) / Delta
	a2 = (N * xysum - xsum * ysum) / Delta
	Cov[0, 0] = sqsum
	Cov[1, 1] = N
	Cov[0, 1] = -xsum
	Cov[1, 0] = Cov[0, 1]
	Cov = (s**2 / Delta) * Cov
	return a1, a2, Cov 


def get_bounds(x, a1, a2, Cov, ignore_cov=False):
	y = a1 + a2 * x
	var_y = Cov[0, 0] + x**2 * Cov[1, 1] + 2 * x * Cov[0, 1]
	if ignore_cov:
		var_y -= 2 * x * Cov[0, 1]
	lower = y - var_y**0.5
	upper = y + var_y**0.5
	return lower, upper


def ej6d(data_x, data_y, s):
	a1, a2, Cov = lin_fit(data_x, data_y, s)
	print a1, a2, Cov
	x = np.arange(6)
	y_fit = a1 + a2*x
	plt.errorbar(data_x, data_y, s, fmt='.', label='Datos')
	plt.plot(x, y_fit, 'r-', label='Ajuste')
	lower, upper = get_bounds(x, a1, a2, Cov)
	plt.plot(x, lower, 'r--', x, upper, 'r--', label='Error del Ajuste')
	plt.legend(loc='upper left')
	plt.grid(True)
	plt.savefig('ej6d.jpg')
	plt.show()


def ej6e(data_x, data_y, s):
	a1, a2, Cov = lin_fit(data_x, data_y, s)
	print a1, a2, Cov
	x = np.arange(6)
	y_fit = a1 + a2*x
	plt.errorbar(data_x, data_y, s, fmt='.', label='Datos')
	plt.plot(x, y_fit, 'r-', label='Ajuste')
	lower, upper = get_bounds(x, a1, a2, Cov, ignore_cov=True)
	plt.plot(x, lower, 'r--', x, upper, 'r--', label='Error del Ajuste')
	plt.legend(loc='upper left')
	plt.grid(True)
	plt.savefig('ej6e.jpg')
	plt.show()


def ej7(data_x, data_y, s):
	a1, a2, cov = lin_fit(data_x, data_y, s)
	y05 = []
	for _ in range(1000):
		new_y = []
		for xi in data_x:
			new_y.append(np.random.normal(a1 + a2*xi, s))
		new_a1, new_a2, new_cov = lin_fit(data_x, new_y, s)
		y05.append(new_a1 + new_a2*0.5)
	plt.hist(y05, normed=True, bins=20, label='$y_a$')
	lower, upper = get_bounds(0.5, a1, a2, cov)
	plt.plot([lower, lower], [0, 1], 'k--')
	plt.plot([upper, upper], [0, 1], 'k--')
	mu = (lower + upper) /2
	sigma = mu - lower
	C = 1.0 /(2 * np.pi * sigma**2)**0.5
	print C
	xgauss = np.arange(-1, 5, 0.01)
	ygauss = C * np.exp(-((xgauss - mu)**2 ) / (2 * sigma**2))  
	plt.plot(xgauss, ygauss, 'r-', label=r'$Dsitribuci\'on\ Gaussiana\ Te\'orica$')
	plt.legend()
	plt.xlabel('$y_a$')
	plt.ylabel('Frecuencia')
	plt.savefig('ej7.jpg')
	plt.show()


def main(args):
	print '-----Ejercicio 6-------'
	x = [2.00, 2.10, 2.20, 2.30, 2.40, 2.50, 2.60, 2.70, 2.80, 2.90, 3.00]
	y = [2.78, 3.29, 3.29, 3.33, 3.23, 3.69, 3.46, 3.87, 3.62, 3.40, 3.99]
	s = 0.3
	if '6d' in args.items:
		print '-----6d)-------'
		ej6d(x, y, s)
	if '6e' in args.items:
		print '-----6e)-------'
		ej6e(x, y, s)
	if '7' in args.items:
		print '-----7)-------'
		ej7(x, y, s)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Resuleve el ejercicio 6d, 6e y 7 de la guia 5.')
	parser.add_argument('items', metavar='I', type=str, nargs='+',
	                    help='Los items a resolver.')
	args = parser.parse_args()
	main(args)
