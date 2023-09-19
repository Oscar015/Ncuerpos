import numpy as np
import matplotlib.pyplot as plt

from numba import njit

from matplotlib import animation
from matplotlib.patches import Rectangle
from scipy.constants import G
from tqdm import tqdm

import Planetas


def LeerDict(dic, cuerpos):
    N = len(cuerpos)
    colors = np.array([dic[cuerpos[i]]['color'] for i in range(N)], str)
    m = np.array([dic[cuerpos[i]]['masa'] for i in range(N)], np.float64)
    x_0 = np.array([dic[cuerpos[i]]['x_0'] for i in range(N)], np.float64)
    v_0 = np.array([dic[cuerpos[i]]['v_0'] for i in range(N)], np.float64)
    return (colors, m, x_0, v_0)


class SimulacionNBody:
    cuerpos = np.array([],dtype=str)
    def __init__(self, cuerpos: np.ndarray[str], t_fin:float, dt:float, path = ""):
        self.cuerpos = np.array(['sol', 'mercurio', 'venus', 'tierra_S',
                        'marte', 'jupiter', 'saturno', 'urano', 'neptuno'])
        
        self.colors, self.m, self.x_0, self.v_0 = LeerDict(Planetas.DictC, cuerpos)
        self.dt = dt
        self.t = range(0, t_fin, dt)
        self.N = len(self.cuerpos)
        self.shape = (len(self.t)+1, self.N, 3)
        self.path = path or 'NBody.csv'

        self.X = np.zeros(self.shape, np.float64)
        
    
    def load_data(self, file:str):
        data = np.loadtxt(file, delimiter=';').reshape(self.shape)
        self.X = data.copy()

    def SaveData(self, file:str):
        data = self.X.copy().reshape(-1)
        data.tofile(file, sep=';')


    def Calc_a(self, pos):
        a = np.array([[0., 0., 0.]]*self.N, np.float64)
        for i in range(len(pos)):
            for j in range(len(pos)):
                if i != j:
                    r = pos[j]-pos[i]
                    dist = np.linalg.norm(r)
                    a[i] += (G * self.m[j] * r) / (dist*dist*dist)
        return a


    def CalcularSimulacion(self):

        X = np.zeros(self.shape, np.float64)
        V = np.zeros(self.shape, np.float64)
        A = np.zeros(self.shape, np.float64)
        X[0] = self.x_0
        V[0] = self.v_0
        total_it = len(self.t)
        bar = tqdm(desc='Calculando...', total=total_it)
        for i in range(total_it):
            A[i+1] = self.Calc_a(X[i])
            V[i+1] = V[i] + A[i+1] * self.dt
            X[i+1] = X[i] + V[i+1] * self.dt
            bar.update()
        self.X = X
    



    def Plot2D(self,speed=1):
        fig = plt.figure(dpi=150, figsize=(5, 5))
        lim = np.max(self.X)*1.2
        ax = plt.axes(xlim=(-lim, lim), ylim=(-lim, lim))
        ax.add_artist(Rectangle([-lim, -lim], 2*lim,
                                2*lim, color='black', zorder=-99))
        Nframes = int(len(self.t)/speed)
        planetas = []
        lineas = []
        for i in range(sim.N):
            temp1, = ax.plot([], [], 'o', color=self.colors[i])
            temp2, = ax.plot([], [], alpha=0.6, color=self.colors[i])
            planetas.append(temp1)
            lineas.append(temp2)

        def init():
            for j in range(sim.N):
                planetas[j].set_data([], [])
                lineas[j].set_data([], [])
            return tuple(planetas)+tuple(lineas)

        def animate(i):
            for j in range(sim.N):
                planetas[j].set_data(self.X[speed*i, j, 0], self.X[speed*i, j, 1])
                lineas[j].set_data(self.X[:speed*i, j, 0], self.X[:speed*i, j, 1])
            return tuple(planetas)+tuple(lineas)

        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                    frames=Nframes, interval=20, blit=True)
        plt.show()
        return anim
    
    def Plot3D(self,speed=1):
        plt.rcParams['axes.facecolor'] = 'black'
        fig = plt.figure(dpi=200, figsize=(6, 6))
        ax = fig.add_subplot(projection="3d", aspect='auto')

        lim = np.max(self.X[:, :, :1])*1.2
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

        Nframes = int(len(self.t)/speed)

        planetas = []
        lineas = []
        for i in range(self.N):
            temp1, = ax.plot([], [], [], 'o', color=self.colors[i])
            temp2, = ax.plot([], [], [], alpha=0.6, color=self.colors[i])
            planetas.append(temp1)
            lineas.append(temp2)


        def animate(i, data, lines):
            ax.set_zlim3d([-lim+self.X[i, 0, 2]*15, lim+self.X[i, 0, 2]*15])

            for j in range(self.N):
                lines[j].set_data(np.array([data[speed*i, j, 0]]),
                                np.array([data[speed*i, j, 1]]))
                lines[j].set_3d_properties(np.array([data[speed*i, j, 2]]))
                lines[self.N+j].set_data(data[:speed*i, j, 0], data[:speed*i, j, 1])
                lines[self.N+j].set_3d_properties(data[:speed*i, j, 2])
            return lines

        anim = animation.FuncAnimation(
            fig, animate, frames=Nframes,
            fargs=(self.X, tuple(planetas) + tuple(lineas)), interval=40)
        plt.show()

        return anim



if __name__=="__main__":

    years = 500
    sim = SimulacionNBody(np.array(['sol', 'mercurio', 'venus', 'tierra_S',
                        'marte', 'jupiter', 'saturno', 'urano', 'neptuno']), 86400*365*years,86400)
    sim.CalcularSimulacion()
    sim.SaveData("Solar_system"+str(years)+".csv")
    sim.Plot3D(30)
