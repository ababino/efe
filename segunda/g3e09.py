#!/usr/bin/env python
"""Resuleve el ejercicio 15 de la guia 2."""
import random
from matplotlib import pyplot as plt
import numpy as np
from scipy.special import binom
import inspect
import argparse
from math import atan2

def exp_random_variable(l):
	"""
	Exponencial random variable.
	"""
	x = random.random()
	y = - np.log(1-x)/l
	return y


def my_hist(data, bins, err_type='poisson', **kwargs):
	"""Histogram with poissonian error bars."""
	y, bin_edges = np.histogram(data, bins=bins)
	width = bin_edges[1] - bin_edges[0]
	normalization = width * sum(y)
	if err_type == 'poisson':
		yerr = np.sqrt(y) / normalization
	elif err_type == 'binomial':
		yerr = np.sqrt(abs(y * ( 1 - y / normalization))) / normalization
	y = y.astype(np.float) / normalization
	plt.bar(bin_edges[:-1], y, yerr=yerr, width=width, ecolor='r', **kwargs)


def cauchy(x):
	y = 1.0 / (1.0 + x**2)
	y = y / np.pi
	return y


def normal(x, mu, sigma):
	a = 1 / np.sqrt(2 * np.pi * sigma**2)
	y = a * np.exp(-(x - mu)**2 / (2 * sigma**2))
	return y


def ej4a():
	x = np.linspace(-6, 6, 1000)
	y_cauchy = cauchy(x) / cauchy(0)
	y_normal = normal(x, 0, 0.75) / normal(0, 0, 0.75)
	plt.plot(x, y_cauchy, 'b', label="Cauchy")
	plt.plot(x, y_normal, 'r', label="Normal (0, 0.75)")
	plt.legend()
	plt.grid(True)
	plt.savefig('ej4a.jpg')
	plt.show()



def ej4b():
	x = np.linspace(-10, 10, 1000)
	y_cauchy = cauchy(x)  / cauchy(0)
	y_normal = 0.5 * normal(x, 0, 0.75) + 0.5 * normal(x, 0, 3)
	y_normal = y_normal / (0.5 * normal(0, 0, 0.75) + 0.5 * normal(0, 0, 3))
	plt.plot(x, y_cauchy, 'b', label="Cauchy")
	plt.plot(x, y_normal, 'r', label="0.5 * Normal (0, 0.75) + 0.5 * Normal (0, 3)")
	plt.xticks(range(-10, 10))
	plt.legend()
	plt.grid(True)
	plt.savefig('ej4b.jpg')
	plt.show()


def ej9():
	n = 500
	l = 0.25
	exp_rand = [exp_random_variable(l) for x in range(n)]
	x = np.arange(0.5, 30.5)
	y = l*np.exp(-l*x)
	my_hist(exp_rand, bins=np.arange(0, 31), label=r'$Simulaci\'on$',
			err_type='binomial')
	plt.plot(x, y, 'k--*', label=r'$Distribuci\'on\ Te\'orica$')
	plt.ylabel('$Frecuencia$')
	plt.legend(loc='upper left')
	plt.savefig('ej9b.jpg')
	plt.show()


def densidad(r):
	if r<0 or r>1:
		rho = 0
	else:
		rho = 1.0 / (1 + r**2)
		rho = rho / (np.pi * (4 - np.pi))
	return rho


def random_densidad(n):
	vecs = []
	while True:
		(x, y, z) = 2 * np.random.rand(3) - 1
		u = np.random.rand(1) / (np.pi * (4 - np.pi))
		r = np.sqrt(x**2 + y**2 + z**2)
		if u <= densidad(r):
			vecs.append((x, y, z))
		if len(vecs) == n:
			break
	return vecs


def r_marginal(r):
	return 4* r**2 / ((4 - np.pi)*(1 + r**2))


def theta_marginal(theta):
	return 0.5 * np.cos(theta)


def phi_marginal(phi):
	return [1.0 / (2 * np.pi) for i in range(len(phi))]


def g4ej10():
	v = random_densidad(1000)
	r = [np.sqrt(x**2 + y**2 + z**2) for (x, y, z) in v]
	phi = [atan2(y, x) for (x, y, z) in v]
	theta = [atan2(z, np.sqrt(x**2 + y**2)) for (x, y, z) in v]
	r_x = np.linspace(0, 1, 1000)
	r_y = r_marginal(r_x)
	phi_x = np.linspace(-np.pi, np.pi, 1000)
	phi_y = phi_marginal(phi_x)
	theta_x = np.linspace(-np.pi/2, np.pi/2, 1000)
	theta_y = theta_marginal(theta_x)
	
	plt.figure(1)
	my_hist(r, bins=np.linspace(0, 1, 20), label=r'$Simulaci\'on$',
			err_type='binomial')
	plt.plot(r_x, r_y, '--k', label=r'$Distribuci\'on\ Te\'orica$' )
	plt.xlabel('r')
	plt.legend()
	plt.savefig('g4ej10_r.jpg')

	plt.figure(2)
	my_hist(phi, bins=np.linspace(-np.pi, np.pi, 20), label=r'$Simulaci\'on$',
			err_type='binomial')
	plt.plot(phi_x, phi_y, '--k', label=r'$Distribuci\'on\ Te\'orica$')
	plt.xlabel('phi')
	plt.legend()
	plt.savefig('g4ej10_phi.jpg')

	plt.figure(3)
	my_hist(theta, bins=np.linspace(-np.pi/2, np.pi/2, 20), label=r'$Simulaci\'on$',
			err_type='binomial')
	plt.plot(theta_x, theta_y, '--k', label=r'$Distribuci\'on\ Te\'orica$')
	plt.xlabel('theta')
	plt.legend()
	plt.savefig('g4ej10_theta.jpg')
	
	plt.show()
	return


def main():
	print '-----Ejercicio 4-------'
	print '-----b)-------'
	#print ej4a()
	#print ej4b()

	print '-----Ejercicio 9-------'
	print '-----b)-------'
	#print ej9()
	g4ej10()

if __name__ == '__main__':
	main()
