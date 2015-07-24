from pylab import *

P = array([ 18.71,   2.79,  13.61,  12.08,   1.89])
F = array([4854., 2586., 3752., 3753., 2605.])
t = array([200., 100., 150., 150., 100.])
tc = 100.
Fc = 1021.

B = F/t - Fc/tc

A = ones((len(P), 2))
A[:, 1] = log(P.T)

V = zeros((len(P), len(P)))
