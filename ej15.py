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


def exp1():
	n = 15
	p = 0.75
	s = binomial_sample(n, p)
	hist_data = [binomial_sample(n, p) for x in xrange(1000)]
	theory_x = range(0, n + 1)
	theory_y = [binomial(x, n, p) for x in theory_x]
	plt.hist(hist_data, normed=1, label=r'$Simulaci\'on$')
	plt.plot(theory_x, theory_y, 'k-', label=r'$Distribuci\'on\ Te\'orica$')
	plt.xlabel('Fotones detectados')
	plt.ylabel('Tasa')
	plt.legend(loc='upper left')
	plt.savefig('figb.jpg')
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



if __name__ == '__main__':
	main()