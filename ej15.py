import random
from matplotlib import pyplot as plt
import numpy as np
from scipy.special import binom
import inspect

def binomial_sample(n, p):
	"""
	Take a sample of size n from a binomial distribution with
	success probability equal p.
	Returns the number of sucesses.
	"""
	s = 0
	for i in xrange(n):
		x = random.random()
		if x < p:
			s += 1
	return s


def binomial(x, n, p):
	return binom(n, x) * p**x * (1-p)**(n-x)


class Detector(object):
	def __init__(self):
		self.eficiencia = 0.75
	def detectar(self, numero_fotones):
		return binomial_sample(numero_fotones, self.eficiencia)


def exp1():
	n = 15
	p = 0.75
	s = binomial_sample(n, p)
	hist_data = [binomial_sample(n, p) for x in xrange(1000)]
	theory_x = range(0, n + 1)
	theory_y = [binomial(x, n, p) for x in theory_x]
	plt.hist(hist_data, bins=np.arange(0.5, 15.5), normed=1, label=r'$Simulaci\'on$')
	plt.plot(theory_x, theory_y, 'k--*', label=r'$Distribuci\'on\ Te\'orica$')
	plt.xlabel('Fotones detectados')
	plt.ylabel('Tasa')
	plt.legend(loc='upper left')
	plt.savefig('figb.jpg')
	plt.show()


def exp2():
	I = 15.
	delta_t = 1.
	n = 1000
	dt = delta_t / n
	p = I*dt
	hist_data = [binomial_sample(n, p) for x in xrange(1000)]
	theory_x = range(0, n + 1)
	theory_y = [binomial(x, n, p) for x in theory_x]
	plt.hist(hist_data, bins=np.arange(0.5, 1000.5), normed=1, label=r'$Simulaci\'on$')
	plt.plot(theory_x, theory_y, 'k--*', label=r'$Distribuci\'on\ Te\'orica$')
	plt.xlabel(r'$N\'umero\ de\ Fotones\ Emitidos$')
	plt.ylabel(r'$Tasa$')
	plt.legend()
	plt.xlim([0, 35])
	plt.savefig('figc.jpg')
	plt.show()


def main():
	n = 10
	p = 0.7
	s = binomial_sample(n, p)
	print '-----Ejercicio 15-------'
	print '-----a)-------'
	print inspect.getsource(binomial_sample)
	print 'n = %s, p = %s, s= %s' % (n, p, s)
	print '-----b)-------'
	exp1()
	print '-----c)-------'
	exp2()


if __name__ == '__main__':
	main()