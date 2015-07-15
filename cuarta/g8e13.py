#!/usr/bin/env python
"""Resuleve el ejercicio 13 de la guia 8."""
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import argparse
from scipy import optimize
#sns.set_style("dark")

def load_data():
    file_data = open('doble_exp.dat', 'r')
    file_data.readline()
    file_data.readline()
    time = []
    count = []
    for line in file_data:
        data_list = line.split()
        if len(data_list) > 1:
            time.append(float(line.split()[0]))
            count.append(float(line.split()[-1]))
    time = np.array(time)
    count = np.array(count)
    poi_err = np.sqrt(count)
    return time, count, poi_err


def fit_fun(tita, xdata):
    yfit = tita[0]
    yfit += tita[1] * np.exp(-xdata/tita[3])
    yfit += tita[2] * np.exp(-xdata/tita[4])
    return yfit


def fit_fun_grad(tita, xdata):
    yfitd = np.ones((len(tita), len(xdata)))
    yfitd[1, :] = np.exp(-xdata/tita[3])
    yfitd[2, :] = np.exp(-xdata/tita[4])
    yfitd[3, :] = tita[1] * np.exp(-xdata/tita[3]) * xdata / tita[3]**2
    yfitd[4, :] = tita[2] * np.exp(-xdata/tita[4]) * xdata / tita[4]**2
    return yfitd


def fit_fun_hess(tita, xdata):
    yfitd = fit_fun_grad(tita, xdata)
    yfith = np.zeros((len(tita), len(tita), len(xdata)))
    yfith[1, 3, :] = yfitd[1,:] * xdata/tita[3]**2
    yfith[2, 4, :] = yfitd[2,:] * xdata/tita[4]**2
    yfith[3, 3, :] = yfitd[3,:] * (xdata / tita[3] - 2) / tita[3]
    yfith[4, 4, :] = yfitd[4,:] * (xdata / tita[4] - 2) / tita[4]
    yfith[3, 1, :] = yfith[1, 3, :]
    yfith[4, 2, :] = yfith[2, 4, :]
    return yfith


def plot_fit(tita, xdata, ydata, yerr):
    f = plt.figure(1)
    ax = f.add_subplot(1,1,1)
    ax.errorbar(xdata, ydata, yerr = yerr, fmt='.', label='Datos')
    ax.plot(xdata, fit_fun(tita, xdata), label='Ajuste')
    ax.set_yscale('log')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Cuentas')
    plt.legend()
    plt.show()


def chi2(tita, xdata, ydata):
    yfit = fit_fun(tita, xdata)
    chi2 = sum((ydata - yfit)**2 / ydata)
    return chi2
def jac(tita, xdata, ydata):
    yfit = fit_fun(tita, xdata)
    yfitd = fit_fun_grad(tita, xdata)
    a = (yfit - ydata) / ydata
    j = 2 * np.dot(yfitd, a.T)
    return j
def hessian(tita, xdata, ydata):
    yfit = fit_fun(tita, xdata)
    yfitd = fit_fun_grad(tita, xdata)
    yfith = fit_fun_hess(tita, xdata)
    a = (yfit - ydata) / ydata
    a = np.tile(np.tile(a, (len(tita), 1)), (len(tita), 1, 1))
    h = 2 *  np.sum(yfith * a, axis=2)
    h += 2 *sum([np.outer(c, c) / e for c, e in zip(yfitd.T, ydata)])
    return h


def e13a(time, count, poi_err, a4, a5):
    f2 = lambda x: np.exp(-x / a4)
    f3 = lambda x: np.exp(-x / a5)
    A = np.ones((len(time), 3))
    A[:,1] = f2(time)
    A[:,2] = f3(time)
    V = np.eye(len(time))
    np.fill_diagonal(V, poi_err)
    av = np.dot(A.T, np.linalg.inv(V))
    ava = np.dot(av, A)
    ava = np.linalg.inv(ava)
    avy = np.dot(av, count)
    tita = np.dot(ava, avy)
    a23 = tita[1] / tita[2]
    jacob = np.array([1 / tita[1], -a23 / tita[2]])
    a23var = np.dot(np.dot(jacob, ava[1:, 1:]), jacob.T)
    return tita, ava, a23, a23var


def e13b(time, count):
    x0 = [10.6888, 127.9398, 960.8654, 200., 34.]
    objfun = lambda x: chi2(x, time, count)
    objfunjac = lambda x: jac(x, time, count)
    objfunhess = lambda x: hessian(x, time, count)
    #'Newton-CG' 'Nelder-Mead' 'BFGS' 'Powell' 'CG'
    res = optimize.minimize(objfun, x0, method='BFGS', jac=objfunjac,
                               hess=None,
                               options={'disp': True})
    return res, objfunhess(res.x)


def e13c(time, count):
    x0 = [10.6888, 127.9398, 960.8654, 200., 34.]
    objfun = lambda x: chi2(x, time, count)
    objfunjac = lambda x: jac(x, time, count)
    res = optimize.minimize(objfun, x0, method='BFGS', jac=objfunjac)
    row = res.x[4] + np.linspace(-2.5,2.5,50)
    col = res.x[3] + np.linspace(-30,40,50)
    X,Y = np.meshgrid(row,col)
    f = np.vectorize(lambda x, y: objfun([res.x[0], res.x[1], res.x[2], x, y]))
    Z = f(Y, X)
    plt.contour(X, Y, Z, [objfun(res.x)+1])
    @np.vectorize
    def f(a4, a5):
        tita, ava, a23, a23var = e13a(time, count, poi_err, a4, a5)
        x = np.concatenate([tita, [a4, a5]])
        return objfun(params)
    Z = f(Y, X)
    plt.contour(X, Y, Z, [objfun(res.x)+1])
    plt.show()
    return res



def main(args):
    time, count, poi_err = load_data()
    a4 =209.69
    a5 =34.244
    if '13a' in args.items:
        tita, ava, a23, a23var = e13a(time, count, poi_err, a4, a5)
        print('Guia 6, ejercicio 13, item a')
        print('tita = ' + str(tita))
        print('V(tita) = ' + str(ava))
        print('a2/a3 = ' + str(a23))
        print('a23var = ' + str(a23var))
        params = np.concatenate([tita, [a4, a5]])
        plot_fit(params, time, count, poi_err)
    if '13b' in args.items:
        res, H = e13b(time, count)
        print(H)
    if '13c' in args.items:
        e13c(time, count, poi_err)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Resuleve el ejercicio 13 de la guia 8.')
    parser.add_argument('items', metavar='I', type=str, nargs='+',
                        help='Los items a resolver.')
    args = parser.parse_args()
    main(args)
