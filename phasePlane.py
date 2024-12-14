import sympy as sym
from matplotlib import pyplot as plt
from scipy.integrate import odeint
import numpy as np


#Parameters configured for dampened pendulum

#Natural frequency
omega = 1
#Daming term
gamma = 0.1

#Graph limits
xmax =  2.5
xmin = -10
ymax =  5
ymin = -5

#Define equation
x = sym.Function('x')
y = sym.Function('y')
t = sym.symbols('t')
eq1 = sym.Eq(x(t).diff(t),y(t))
eq2 = sym.Eq(y(t).diff(t), -omega**2*sym.sin(x(t)) - gamma*y(t))

#Form matrix
FG = sym.Matrix([eq1.rhs, eq2.rhs])

#Solve equation
print(f"critical Points Are {sym.solve(FG)}")


#Define field
def vField(x, t):
    u = x[1]
    v = -omega**2*np.sin(x[0]) - gamma*x[1]
    return [u,v]

#Create meshgrid
X, Y = np.mgrid[xmin:xmax:20j,ymin:ymax:20j]
U, V = vField([X,Y],0)

"""fix arrow size and distribution with relation x/y min/max"""

M = np.hypot(U, V)

fig, ax = plt.subplots(figsize=((abs(xmax)+abs(xmin)), abs(ymax)+abs(ymin)))
ax.quiver(X, Y, U, V, M, scale=1/0.005, pivot = 'mid', cmap = plt.cm.bone)



#Ploting points and trajectories
t=np.linspace(0,100,10000)

x0, y0 = 2,-2
x = odeint(vField,[x0,y0],t)
ax.plot(x[:,0],x[:,1]);
ax.scatter([0.1], [4], color='blue', s=20)



#Graph configuration
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
plt.savefig('graphs/Pendulum.png', bbox_inches='tight', pad_inches=0.1)