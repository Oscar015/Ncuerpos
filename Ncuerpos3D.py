# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 19:28:43 2021

@author: oscar
"""
import numpy as np
import matplotlib.pyplot as plt

# from numba import njit

from matplotlib import animation
from scipy.constants import G
from tqdm import tqdm

import Planetas

plt.rcParams['axes.facecolor'] = 'black'
eps = np.finfo(float).eps
# =============================================================================
# =============================================================================


def LeerDict(dic, cuerpos):
    N = len(cuerpos)
    colors = np.array([dic[cuerpos[i]]['color'] for i in range(N)], str)
    m = np.array([dic[cuerpos[i]]['masa'] for i in range(N)], np.float64)
    x_0 = np.array([dic[cuerpos[i]]['x_0'] for i in range(N)], np.float64)
    v_0 = np.array([dic[cuerpos[i]]['v_0'] for i in range(N)], np.float64)
    return (colors, m, x_0, v_0)


def SaveData(data, path):
    data = data.reshape(np.prod(data.shape))
    data.tofile(path, sep=';')


def LoadData(path, shape):
    data = np.loadtxt(path, delimiter=';').reshape(shape)
    return data


# @njit
def Calc_a(m, pos):
    a = np.array([[0., 0., 0.]]*N, np.float64)
    for i in range(len(pos)):
        for j in range(len(pos)):
            r = (pos[j]-pos[i])
            dist = np.linalg.norm(r)
            a[i] += (G * m[j] * r) / (dist*dist*dist+eps)
    return a


# @njit
def CalcularSimulacion(t, m, x_0, v_0):

    X = np.zeros(shape, np.float64)
    V = np.zeros(shape, np.float64)
    A = np.zeros(shape, np.float64)
    X[0] = x_0
    V[0] = v_0
    for i in range(len(t)):
        A[i+1] = Calc_a(m, X[i])
        V[i+1] = V[i] + A[i+1] * deltaT
        X[i+1] = X[i] + V[i+1] * deltaT
    return X


def Plot(X):
    fig = plt.figure(dpi=200, figsize=(6, 6))
    ax = fig.add_subplot(projection="3d", aspect='auto')

    lim = np.max(X[:, :, :1])*1.2
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.xaxis.pane.set_edgecolor('k')
    ax.yaxis.pane.set_edgecolor('k')
    ax.zaxis.pane.set_edgecolor('k')

    Nframes = int(len(t)/speed)

    planetas = []
    lineas = []
    for i in range(N):
        temp1, = ax.plot([], [], [], 'o', color=colors[i])
        temp2, = ax.plot([], [], [], alpha=0.6, color=colors[i])
        planetas.append(temp1)
        lineas.append(temp2)

    bar = tqdm(desc='Animando...', total=Nframes)

    def animate(i, data, lines, bar):
        ax.set_zlim3d([-lim+X[i, 0, 2]*15, lim+X[i, 0, 2]*15])

        for j in range(N):
            lines[j].set_data(data[speed*i, j, 0], data[speed*i, j, 1])
            lines[j].set_3d_properties(data[speed*i, j, 2])
            lines[N+j].set_data(data[:speed*i, j, 0], data[:speed*i, j, 1])
            lines[N+j].set_3d_properties(data[:speed*i, j, 2])
            bar.update()
        return lines

    anim = animation.FuncAnimation(
        fig, animate, frames=Nframes,
        fargs=(X, tuple(planetas) + tuple(lineas), bar), interval=40)
    plt.show()

    return anim

# %% Main


if __name__ == '__main__':

    # Cuerpos que usaremos en la simulación
    cuerpos = np.array(['sol', 'mercurio', 'venus', 'tierra_S',
                        'marte', 'jupiter', 'saturno', 'urano', 'neptuno'])
    speed = 15  # Velocidad de la simulación

    t_annos = 500  # años de duración de la simulación

    Read = True  # True lee datos, falso los guarda
    Animar = True

    path = 'Solar_System.csv'
    deltaT = 86400
    T_fin = 86400 * 365 * t_annos
    t = np.arange(0, T_fin, deltaT)

    N = len(cuerpos)
    shape = (len(t)+1, N, 3)

    colors, m, x_0, v_0 = LeerDict(Planetas.DictC, cuerpos)
    if Read:
        X = LoadData(path, shape)
    else:
        X = CalcularSimulacion(t, m, x_0, v_0)
        SaveData(X, path)

    if Animar:
        anim = Plot(X)
