#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Resuleve el ejercicio 15 de la guia 2."""
from __future__ import unicode_literals
import random
from matplotlib import pyplot as plt
import numpy as np
from scipy.special import binom
import inspect
import argparse
import seaborn as sns


def binomial_sample(n, p):
	"""
	Takes a sample of size n from a binomial distribution with a success
	probability equal p. Returns the number of successes.
	"""
	s = 0
	for i in xrange(n):
		x = random.random()
		if x < p:
			s += 1
	return s


def binomial(x, n, p):
	"""The binomial density function. """
	return binom(n, x) * p**x * (1-p)**(n-x)


def poisson(k, l):
	"""The Poisson density function"""
	return l**k * np.exp(-l) / np.math.factorial(k)


def my_hist(data, bins, err_type='poisson', **kwargs):
	"""Histogram with poissonian error bars."""
	y, bin_edges = np.histogram(data, bins=bins)
	normalization = sum(y)
	if err_type == 'poisson':
		yerr = np.sqrt(y) / normalization
	elif err_type == 'binomial':
		yerr = np.sqrt(y * ( 1 - y / normalization) / normalization)
	y = y.astype(np.float) / normalization
	plt.bar(bin_edges[:-1], y, yerr=yerr, width=1.0, ecolor='r', **kwargs)


class Detector(object):
	"""Detector de fotones."""
	def __init__(self):
		self.eficiencia = 0.75
	def detectar(self, numero_fotones):
		return binomial_sample(numero_fotones, self.eficiencia)


class Fuente(object):
	"""Fuente emisora de fotones."""
	def __init__(self):
		self.intensidad = 15.
		self.n = 1000
		self.dt = 1. / self.n
		self.p = self.intensidad * self.dt
	def emitir(self, delta_t):
		return binomial_sample(int(self.n * delta_t), self.p)


class Fuente_Detector(object):
	"""Objecto conjunto fuente detector."""
	def __init__(self):
		self.fuente = Fuente()
		self.detector = Detector()
		self.eficiencia_conjunta = self.fuente.intensidad * self.detector.eficiencia
		self.p = self.fuente.p * self.detector.eficiencia
	def emitir_detectar(self, delta_t):
		n_samp = int(self.fuente.n * delta_t)
		return binomial_sample(n_samp, self.p)


def exp1(argshow):
	n = 15
	detector = Detector()
	hist_data = [detector.detectar(n) for x in xrange(1000)]
	theory_x = range(0, n + 1)
	theory_y = [binomial(x, n, detector.eficiencia) for x in theory_x]
	my_hist(hist_data, bins=np.arange(-0.5, 16.5), label='Simulación',
			err_type='binomial')
	plt.plot(theory_x, theory_y, 'k--*', label='Distribución Teórica')
	plt.xlabel('Número Fotones detectados')
	plt.ylabel('Frecuencia')
	plt.legend(loc='upper left')
	plt.savefig('figb.jpg')
	if argshow:
		plt.show()
	else:
		plt.close()


def exp2(argshow):
	fuente = Fuente()
	delta_t = 1.
	N = int(fuente.n * delta_t)
	theory_x = range(0, 5)
	theory_y = [poisson(k, fuente.intensidad * fuente.dt) for k in theory_x]
	print(theory_y)
	plt.plot(theory_x, theory_y, 'k--*')
	plt.xlabel('Número de Eventos')
	plt.ylabel('Frecuencia')
	plt.savefig('poisson.jpg')
	if argshow:
		plt.show()
	else:
		plt.close()
	hist_data = [fuente.emitir(delta_t) for x in xrange(1000)]
	theory_x = range(0, 40)
	theory_y = [poisson(k, fuente.intensidad * delta_t) for k in theory_x]
	my_hist(hist_data, bins=np.arange(-0.5, N + 1.5), label='Simulación')
	plt.plot(theory_x, theory_y, 'k--*', label='Distribución Teórica')
	plt.xlabel('Número de Fotones Emitidos')
	plt.ylabel('Frecuencia')
	plt.legend()
	plt.xlim([0, 35])
	plt.savefig('figc.jpg')
	if argshow:
		plt.show()
	else:
		plt.close()


def exp3(argshow):
	fuente = Fuente()
	detector = Detector()
	delta_t = 1.
	N = int(fuente.n * delta_t)
	hist_data = []
	for i in xrange(1000):
		fotones_emitidos = fuente.emitir(delta_t)
		photones_detectados = detector.detectar(fotones_emitidos)
		hist_data.append(photones_detectados)
	theory_x = range(0, 40)
	theory_y = [poisson(k, fuente.intensidad * detector.eficiencia * delta_t) for k in theory_x]
	my_hist(hist_data, bins=np.arange(-0.5, N + 1.5), label='Simulación')
	plt.plot(theory_x, theory_y, 'k--*', label='Distribución Teórica')
	plt.xlabel('Número de Fotones Detectados')
	plt.ylabel('Frecuencia')
	plt.legend()
	plt.xlim([0, 35])
	plt.savefig('figd.jpg')
	if argshow:
		plt.show()
	else:
		plt.close()


def exp4(argshow):
	fuente_detector = Fuente_Detector()
	delta_t = 1.
	N = int(fuente_detector.fuente.n * delta_t)
	hist_data = []
	for i in xrange(1000):
		photones_detectados = fuente_detector.emitir_detectar(delta_t)
		hist_data.append(photones_detectados)
	theory_x = range(0, 40)
	theory_y = [poisson(k, fuente_detector.eficiencia_conjunta * delta_t) for k in theory_x]
	my_hist(hist_data, bins=np.arange(-0.5, N + 1.5), label='Simulación')
	plt.plot(theory_x, theory_y, 'k--*', label='Distribución Teórica')
	plt.xlabel('Número de Fotones Detectados')
	plt.ylabel('Frecuencia')
	plt.legend()
	plt.xlim([0, 35])
	plt.savefig('fige.jpg')
	if argshow:
		plt.show()
	else:
		plt.close()


def main(args):
	n = 10
	p = 0.7
	s = binomial_sample(n, p)
	print '-----Ejercicio 15-------'
	if 'a' in args.items:
		print '-----a)-------'
		print inspect.getsource(binomial_sample)
		print 'n = %s, p = %s, s= %s' % (n, p, s)
	if 'b' in args.items:
		print '-----b)-------'
		exp1(args.show)
	if 'c' in args.items:
		print '-----c)-------'
		exp2(args.show)
	if 'd' in args.items:
		print '-----d)-------'
		exp3(args.show)
	if 'e' in args.items:
		print '-----e)-------'
		exp4(args.show)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Resuleve el ejercicio 15 de la guia 2.')
	parser.add_argument('-items', metavar='--I', type=str, nargs='+',
	                    help='Los items a resolver.',
	                    default='abcde')
	parser.add_argument('-show', action='store_true')
	args = parser.parse_args()
	main(args)
