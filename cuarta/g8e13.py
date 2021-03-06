#!/usr/bin/env python
"""Resuleve el ejercicio 13 de la guia 8."""
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
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
    yfit += tita[1] * np.exp(-xdata / tita[3])
    yfit += tita[2] * np.exp(-xdata / tita[4])
    return yfit


def fit_fun_grad(tita, xdata):
    yfitd = np.ones((len(tita), len(xdata)))
    yfitd[1, :] = np.exp(-xdata / tita[3])
    yfitd[2, :] = np.exp(-xdata / tita[4])
    yfitd[3, :] = tita[1] * np.exp(-xdata / tita[3]) * xdata / tita[3]**2
    yfitd[4, :] = tita[2] * np.exp(-xdata / tita[4]) * xdata / tita[4]**2
    return yfitd


def fit_fun_hess(tita, xdata):
    yfitd = fit_fun_grad(tita, xdata)
    yfith = np.zeros((len(tita), len(tita), len(xdata)))
    yfith[1, 3, :] = yfitd[1, :] * xdata / tita[3]**2
    yfith[2, 4, :] = yfitd[2, :] * xdata / tita[4]**2
    yfith[3, 3, :] = yfitd[3, :] * (xdata / tita[3] - 2) / tita[3]
    yfith[4, 4, :] = yfitd[4, :] * (xdata / tita[4] - 2) / tita[4]
    yfith[3, 1, :] = yfith[1, 3, :]
    yfith[4, 2, :] = yfith[2, 4, :]
    return yfith


def plot_fit(tita, xdata, ydata, yerr):
    f = plt.figure(1)
    ax = f.add_subplot(1, 1, 1)
    ax.errorbar(xdata, ydata, yerr=yerr, fmt='.', label='Datos')
    ax.plot(xdata, fit_fun(tita, xdata), label='Ajuste, 3 parametros')
    ax.set_yscale('log')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Cuentas')
    plt.legend()
    return f, ax


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
    h = 2 * np.sum(yfith * a, axis=2)
    h += 2 * sum([np.outer(c, c) / e for c, e in zip(yfitd.T, ydata)])
    return h


def e13a(time, count, a4, a5):
    A = np.ones((len(time), 3))
    A[:, 1] = np.exp(-time / a4)
    A[:, 2] = np.exp(-time / a5)
    V = np.eye(len(time))
    np.fill_diagonal(V, count)
    av = np.dot(A.T, np.linalg.inv(V))
    ava = np.dot(av, A)
    ava = np.linalg.inv(ava)
    avy = np.dot(av, count)
    tita = np.dot(ava, avy)
    return tita, ava


def e13b(time, count, x0=None):
    if x0 is None:
        x0 = [10.6888, 127.9398, 960.8654, 200., 34.]
    res = optimize.minimize(lambda x: chi2(x, time, count), x0, method='BFGS',
                            jac=lambda x: hessian(x, time, count), hess=None,
                            options={'disp': True})
    print(0.5 * res.hess_inv)
    return res, hessian(res.x, time, count)


def e13c(time, count, x0=None, cv=1, row_int=(-3, 3), col_int=(-35, 45),
         plot_soloid=True):
    if x0 is None:
        x0 = [10.6888, 127.9398, 960.8654, 200., 34.]
    res = optimize.minimize(lambda x: chi2(x, time, count), x0, method='BFGS',
                            jac=lambda x: jac(x, time, count))
    row = res.x[4] + np.linspace(row_int[0], row_int[1], 50)
    col = res.x[3] + np.linspace(col_int[0], col_int[1], 50)
    X, Y = np.meshgrid(row, col)
    if plot_soloid:
        a = res.x
        f = np.vectorize(lambda x: chi2([a[0], a[1], a[2], x, y], time, count))
        Z = f(Y, X)
        c1 = sns.color_palette()[0]
        plt.contour(X, Y, Z, [objfun(res.x) + cv], colors=[c1])
    x0 = res.x[:3]

    @np.vectorize
    def f(x, y):
        tita, ava = e13a(time, count, x, y)
        params = np.concatenate([tita, [x, y]])
        return objfun(params)
    Z = f(Y, X)
    ct = plt.contour(X, Y, -Z, [-objfun(res.x) - cv],
                     colors=[sns.color_palette()[1]])
    plt.plot([res.x[4]], [res.x[3]], 'o')
    plt.xlabel('$a_5$')
    plt.ylabel('$a_4$')
    p = ct.collections[0].get_paths()[0]
    v = p.vertices
    x = v[:, 0]
    y = v[:, 1]
    print(min(x), max(x), min(y), max(y))
    return res


def main(args):
    time, count, poi_err = load_data()
    a4 = 209.69
    a5 = 34.244
    if '13a' in args.items:
        tita, ava = e13a(time, count, a4, a5)
        a23 = tita[1] / tita[2]
        jacob = np.array([1 / tita[2], -a23 / tita[2]])
        a23var = np.dot(np.dot(jacob, ava[1:, 1:]), jacob.T)
        print('Guia 6, ejercicio 13, item a')
        print('tita = ' + str(tita))
        print('V(tita) = ' + str(ava))
        print('a2/a3 = ' + str(a23))
        print('a23var = ' + str(a23var))
        print('a23sigma = ' + str(np.sqrt(a23var)))
        params = np.concatenate([tita, [a4, a5]])
        print('ch2 = ' + str(chi2(params, time, count)))
        print('d.f. = ' + str(len(time) - 3))
        f, ax = plot_fit(params, time, count, poi_err)
    if '13b' in args.items:
        np.set_printoptions(precision=3)
        np.set_printoptions(suppress=True)
        res, H = e13b(time, count)
        print('tita = ' + str(res.x))
        print('ch2 = ' + str(chi2(res.x, time, count)))
        print('d.f. = ' + str(len(time) - 5))
        np.set_printoptions(precision=2)
        cov = np.linalg.inv(0.5 * H)
        print(cov)
        print(np.sqrt(cov.diagonal()))
    if '13c' in args.items:
        e13c(time, count)
        plt.savefig('fig1.jpg')
        plt.show()
    if '13d' in args.items:
        e13c(time, count, cv=5.86, row_int=(-7, 7), col_int=(-70, 140),
             plot_soloid=False)
        plt.savefig('fig11.jpg')
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Resuleve el ejercicio 13 de la guia 8.')
    parser.add_argument('items', metavar='I', type=str, nargs='+',
                        help='Los items a resolver.')
    args = parser.parse_args()
    main(args)
