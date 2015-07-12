#!/usr/bin/env python
"""Resuleve el ejercicio 13 de la guia 8."""
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sb
import argparse
from scipy import optimize


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
    yfitd[3, :] = - tita[1] * np.exp(-xdata/tita[3]) * xdata/tita[3]**2
    yfitd[4, :] = - tita[2] * np.exp(-xdata/tita[4]) * xdata/tita[4]**2
    return yfitd


def fit_fun_hess(tita, xdata):
    yfitd = fit_fun_grad(tita, xdata)
    yfith = np.zeros((len(tita), len(tita), len(xdata)))
    yfith[1, 3, :] = -yfitd[1,:] * xdata/tita[3]**2
    yfith[2, 4, :] = -yfitd[2,:] * xdata/tita[4]**2
    yfith[3, 3, :] = yfitd[3,:] * (xdata-2) / tita[3]
    yfith[4, 4, :] = yfitd[4,:] * (xdata-2) / tita[4]
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


def e13ia(time, count, poi_err):
    a4 =209.69
    a5 =34.244
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
    print('Guia 6, ejercicio 13, item a')
    print('tita = ' + str(tita))
    print('V(tita) = ' + str(ava))
    print('a2/a3 = ' + str(a23))
    print('a23var = ' + str(a23var))
    params = np.concatenate([tita, [a4, a5]])
    plot_fit(params, time, count, poi_err)
    return tita, ava, a23, a23var


def e13ib(time, count, poi_err):
    def objfun(tita, ydata=count, xdata=time, err=poi_err):
        yfit = fit_fun(tita, xdata)
        chi2 = sum((ydata - yfit)**2 / poi_err)
        return chi2
    def jac(tita, ydata=count, xdata=time, err=poi_err):
        yfit = fit_fun(tita, xdata)
        yfitd = fit_fun_grad(tita, xdata)
        a = -2 * (ydata - yfit) / poi_err
        j = np.dot(yfitd, a.T)
        return j
    def hessian(tita, ydata=count, xdata=time, err=poi_err):
        yfit = fit_fun(tita, xdata)
        yfitd = fit_fun_grad(tita, xdata)
        yfith = fit_fun_hess(tita, xdata)
        a = - 2 * (ydata - yfit) / poi_err
        h = np.sum(yfith * np.tile(a, (len(tita), 1, 1)), axis=2)
        h += 2 *sum([np.outer(c, c) / e for c, e in zip(yfitd.T, poi_err)])
        return h
    x0 = [10.6888, 127.9398, 960.8654, 200., 34.]
    print(x0)
    print(objfun(x0))
    optita = optimize.minimize(objfun, x0, method='Newton-CG', jac=jac,
                               hess=hessian,
                               options={'gtol': 1e-6, 'disp': True})
    print(optita)
    print(objfun(optita.x))
    print(hessian(optita.x))
    print(np.linalg.det(hessian(optita.x)))
    print(np.invert(hessian(optita.x)))
    #plot_fit(optita.x, time, count, poi_err)
    return

def main(args):
    time, count, poi_err = load_data()
    if '13a' in args.items:
        e13ia(time, count, poi_err)
    if '13b' in args.items:
        e13ib(time, count, poi_err)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Resuleve el ejercicio 13 de la guia 8.')
    parser.add_argument('items', metavar='I', type=str, nargs='+',
                        help='Los items a resolver.')
    args = parser.parse_args()
    main(args)
