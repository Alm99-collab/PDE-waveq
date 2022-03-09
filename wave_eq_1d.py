# Author: Alm99-collab
# Date: 25/10/2021 2:00
# Description: MSUT STANKIN Μyagkov Alexandr IDM - 21 - 03


"""
Программа для численного решения волнового уравнения в одномерном случае,
при разлинчых начальных и граничных условий, с помощью явной разностной
схемы типа "крест".

Общий вид волнового уравнения:
u_tt = a**2*u_xx + f(x,t) (0,L) где u=0 на диапазоне x=0,L, for t in (0,T].
Начальные условия в общем случае: u=I(x), u_t=V(x).
В случае неоднородного уравнения задается функция f(x,t).

Синтакис основной функции решателя:
u, x, t = solver(I, V, f, a, L, dt, C, T, user_func) где:
I = I(x) - функция.
V = V(x) - функция.
f = f(x,t) - функция.
U_0, U_L, - условия на границе.
C - число Куранта (=a*dt/dx), зависящее от шага dx. Является криетрием
стабильности численных расчето, если соблюдено условие (<=1)
dt - шаг по времени.
dx - шаг по координате.
T - конечное время симуляции волнового процесса.
user_func - функция (u, x, t, n) для вызова пользовательских сценариев,
таких как анимация или вывод графика, запси данных в текстовый файл,
расчет ошибки (в случае если известно точное решение) и.т.д.
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tabulate import tabulate
from mpl_toolkits.mplot3d import axes3d


# функция решатель для данного дифференциального уравнения
def solver(I, V, f, a, U_0, U_L, L, dt, C, T,
           user_func = None):
    """
    Функция решения волнового уравнения u_tt = a**2*u_xx + f(x,t) (0,L),
    где u=0 на диапазоне x=0,L, for t in (0,T].
    Начальные условия в общем случае: u=I(x), u_t=V(x).
    В случае неоднородного уравнения задается функция f(x,t).
    ------------------------------------------------------------------------
    :param I:
    :param V:
    :param f:
    :param a:
    :param L:
    :param C:
    :param T:
    :param U_0:
    :param U_L:
    :param dt:
    :param user_func:
    :return:
    """

    nt = int(round(T / dt))
    t = np.linspace(0, nt * dt, nt + 1)  # Узлы сетки по времени
    dx = dt * a / float(C)
    nx = int(round(L / dx))
    x = np.linspace(0, L, nx + 1)  # Узлы сетки по координате
    C2 = C ** 2
    dt2 = dt * dt

    # Проверка того, что  массивы являются элементами t,x
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    # Выбор и инициализация дополнительных параметров f, I, V, U_0, U_L если равны нулю или не передаются
    if f is None or f == 0:
        f = lambda x, t: 0
    if I is None or I == 0:
        I = lambda x: 0
    if V is None or V == 0:
        V = lambda x: 0
    if U_0 is not None:
        if isinstance(U_0, (float, int)) and U_0 == 0:
            U_0 = lambda t: 0
        # иначе: U_0(t) является функцией
    if U_L is not None:
        if isinstance(U_L, (float, int)) and U_L == 0:
            U_L = lambda t: 0
        # иначе: U_L(t) является функцией

    # ---  Выделяем память под значения решений  ---
    u = np.zeros(nx + 1)  # Массив решений в узлах сетки на  временном шаге u(i,n+1)
    u_n = np.zeros(nx + 1)  # Массив решений в узлах сетки на  временном шаге u(i,n)
    u_nm1 = np.zeros(nx + 1)  # Массив решений в узлах сетки на  временном шаге u(i,n-1)

    # --- Проверка индексов для соблюдения размерностей массивов ---
    Ix = range(0, nx + 1)
    It = range(0, nt + 1)

    # --- Запись начальных условий ---
    for i in Ix:
        u_n[i] = I(x[i])

    if user_func is not None:
        user_func(u_n, x, t, 0)

    # --- Разностная формулма явной схемы "типа крест" на первом шаге ---
    for i in Ix[1:-1]:
        u[i] = u_n[i] + dt * V(x[i]) + 0.5 * C2 * (u_n[i - 1] - 2 * u_n[i] + u_n[i + 1]) + 0.5 * dt2 * f(x[i], t[0])

    i = Ix[0]
    if U_0 is None:
        # Запись граничных условий (x=0: i-1 -> i+1  u[i-1]=u[i+1]
        # где du/dn = 0, on x=L: i+1 -> i-1  u[i+1]=u[i-1])
        ip1 = i + 1
        im1 = ip1  # i-1 -> i+1
        u[i] = u_n[i] + dt * V(x[i]) + 0.5 * C2 * (u_n[im1] - 2 * u_n[i] + u_n[ip1]) + 0.5 * dt2 * f(x[i], t[0])

    else:
        u[0] = U_0(dt)

    i = Ix[-1]
    if U_L is None:
        im1 = i - 1
        ip1 = im1  # i+1 -> i-1
        u[i] = u_n[i] + dt * V(x[i]) + 0.5 * C2 * (u_n[im1] - 2 * u_n[i] + u_n[ip1]) + 0.5 * dt2 * f(x[i], t[0])
    else:
        u[i] = U_L(dt)

    if user_func is not None:
        user_func(u_n, x, t, 1)

    # Обновление данных и подготовка к новому шагу
    u_nm1, u_n, u = u_n, u, u_nm1

    # --- Симуляция (цикл прохода по времени) ---
    for n in It[1:-1]:
        # Обновление значений во внутренних узлах сетки
        for i in Ix[1:-1]:
            u[i] = - u_nm1[i] + 2 * u_n[i] + C2 * (u_n[i - 1] - 2 * u_n[i] + u_n[i + 1]) + dt2 * f(x[i], t[n])

        #  --- Запись граничных условий ---
        i = Ix[0]
        if U_0 is None:
            # Установка значений граничных условий
            # x=0: i-1 -> i+1  u[i-1]=u[i+1] где du/dn=0
            # x=L: i+1 -> i-1  u[i+1]=u[i-1] где du/dn=0
            ip1 = i + 1
            im1 = ip1
            u[i] = - u_nm1[i] + 2 * u_n[i] + C2 * (u_n[im1] - 2 * u_n[i] + u_n[ip1]) + dt2 * f(x[i], t[n])
        else:
            u[0] = U_0(t[n + 1])

        i = Ix[-1]
        if U_L is None:
            im1 = i - 1
            ip1 = im1
            u[i] = - u_nm1[i] + 2 * u_n[i] + C2 * (u_n[im1] - 2 * u_n[i] + u_n[ip1]) + dt2 * f(x[i], t[n])
        else:
            u[i] = U_L(t[n + 1])

        if user_func is not None:
            if user_func(u, x, t, n + 1):
                break

        # Обновление данных и подготовка к новому шагу
        u_nm1, u_n, u = u_n, u, u_nm1

    # Присвоение значений требуемому узлу после прохода по сетке
    u = u_n

    return u, x, t


# функция симуляции и визуализации решения
def simulate(I, V, f, a, U_0, U_L, L, dt, C, T, umin, umax, plot = True):
    """
    Запуск решателя и анимации,сохранения данных в файл
    ----------------------------------------------------------------------------
    :param user_func:
    :param I:
    :param V:
    :param f:
    :param a:
    :param L:
    :param dt:
    :param C:
    :param T:
    :param umin:
    :param umax:
    :param U_0:
    :param U_L:
    :param plot:
    :return:
    """

    global x
    if callable(U_0):
        bc_left = 'u(0,t)=U_0(t)'
    elif U_0 is None:
        bc_left = 'du(0,t)/dx=0'
    else:
        bc_left = 'u(0,t)=0'
    if callable(U_L):
        bc_right = 'u(L,t)=U_L(t)'
    elif U_L is None:
        bc_right = 'du(L,t)/dx=0'
    else:
        bc_right = 'u(L,t)=0'

    class PlotMatplotlib:
        def __call__(self, u, x, t, n):

            """ user_func для визуализации """
            if n == 0:
                plt.grid()
                plt.ion()
                self.lines = plt.plot(x, u, 'b--')
                plt.xlabel('x')
                plt.ylabel('u')
                plt.axis([0, L, umin, umax])
                plt.legend(['t=%f' % t[n]], loc = 'lower left')
                plt.grid()
            else:
                plt.grid()
                self.lines[0].set_ydata(u)
                plt.legend(['t=%f' % t[n]], loc = 'lower left')
                plt.grid()
                plt.draw()
            time.sleep(0.2) if t[n] == 0 else time.sleep(0.1)
            plt.savefig('res/tmp_%0d.png' % n, dpi = 600)

    plot_u = PlotMatplotlib()
    plt.grid()

    user_func = plot_u if plot else None
    u, x, t = solver(I, V, f, a, U_0, U_L, L, dt, C, T, user_func)
    return u, x, t




# функция постановки задачи
def problem( ):
    I = lambda x: 0.2*(1-x)*np.sin(np.pi*x)  # начальная функция
    V = lambda x:  0  # начальное условие Du/Dt(t = 0)
    f = lambda x, t: 0  # однородная функция f(x,t)
    U_0 = lambda t: 0  # граничные условия
    U_L = lambda t: 0  # граничные условия

    # физические параметры симуляции
    L = 1
    a = 1
    C = 1
    nx = 15
    dt = C * (L / nx) / a
    T = 1
    umin = -0.25
    umax = abs(umin)
    # точное аналитическое решение поставленной задачи
    # u_e = lambda x, t: ((0.2 * np.sin((np.pi*x/L))*np.cos(np.pi*a*t/L))*(L-x)) \
    #                     + ((0.2 * np.sin((np.pi*2*x/L))*np.cos(np.pi*a*2*t/L))*(L-x))\
    #                     + ((0.2 * np.sin((np.pi*3*x/L))*np.cos(np.pi*a*3*t/L))*(L-x))

    # u_e = lambda x, t: 0.2 *x* (1 - x) * np.sin(np.pi * x)*np.cos(np.pi*a*t)

    # u_e = lambda x, t: 0.2 * (1 - x) * np.sin(np.pi * x*t)*np.cos(np.pi * x*t*a)


    #  вызов симуляции и построение графиков, визуализация
    simulate(I, V, f, a, U_0, U_L, L, dt, C, T, umin, umax)

    # расчет погрешности нормы на каждом шаге
    # def assert_no_error(u, x, t, n):
    #     u_err = u_e(x, t[n])
    #     diff = np.abs(u - u_err).max()
    #     print(diff)
    #
    # print()

    # решение поставленной задачи, возврат граничных значений u, а также x,t
    u, x, t = solver(I, V, f, a, U_0, U_L, L, dt, C, T, user_func = None)
    #
    # global error
    # error = 0
    #
    # def compute_error(u, x, t, nx, u_e):
    #     """ расчет глобальной ошибки, оценка по максимальному значению на всех шагах 0 <= i <= Nx """
    #     global error
    #     for i in range(nx):
    #         if nx == 0:
    #             error = 0
    #         else:
    #             error = max(error, np.abs(u - u_e(x, t[nx])).max())
    #     return error
    #
    # error = compute_error(u, x, t, nx, u_e)

    return u, x, t, nx

    # использовать только при вызове расчет нормы compute_error
    # return u, x, t, nx, error


if __name__ == '__main__':
     u, x, t, nx = problem()

     # использовать только при вызове расчет нормы compute_error
     # u, x, t, nx, error = problem()
     # print(error)

     # создание GIF-анимации из отдельных кадров

     # Список для хранения кадров.
     frames = []

     for frame_number in range(0, nx-1):
         # Открываем изображение каждого кадра.
         frame = Image.open(f'res/tmp_{frame_number}.png')
         # Добавляем кадр в список с кадрами.
         frames.append(frame)

     # добавляем к первому кадру следующий и сохраняем анимацию
     frames[0].save('sol.gif', save_all = True, append_images = frames[1:], optimize = True, duration = 300, loop = 0)

     # для вывода значений в виде таблицы (раскомментировать по необходимости)
     u_list = list(u)
     x_list = list(x)
     t_list = list(t)
     d = [u_list, x_list, t_list]
     p = list(zip(*d))
     print(tabulate(p, headers = ('u(x,t)', 'x', 't'), tablefmt = 'pipe', stralign = 'center'))
