import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sympy as sym

x,n = sym.symbols('x, n')

L = 10
a = 1
f = sym.Piecewise((1,(x > (L/2 -1)) & (x < (L/2 +1))),(0, (x <= (L/2 -1))),(0, (x >= (L/2 +1))))

cn = sym.Rational(2,L)*sym.integrate(f*sym.sin(n*sym.pi*x/L), (x, 0, L))

t = sym.symbols('t')
u_symbolic = sym.Sum(cn.simplify()*sym.sin(n*sym.pi*x/L)*sym.cos(n*sym.pi*a*t/L), (n,1,300))


fps = 30 # number of frames per second

fig, ax = plt.subplots()

x_vals = np.linspace(0,L,200)

u = sym.lambdify([x, t], u_symbolic, modules='numpy')

# set up the initial frame
line, = ax.plot(x_vals, u(x_vals,0), 'k-')
plt.plot(x_vals,u(x_vals,0),'r:')
plt.xlabel('x')
plt.ylabel('u')
plt.ylim(-1,1)
plt.close()

# add an annotation showing the time (this will be updated in each frame)
txt = ax.text(0, 0.9, 't=0')

def init():
    line.set_ydata(u(x_vals,0))
    return line,

def animate(i):
    line.set_ydata(u(x_vals,i/fps))  # update the data
    txt.set_text('t='+str(i/fps)) # update the annotation
    return line, txt


ani = animation.FuncAnimation(fig, animate, np.arange(0, fps*20), init_func=init,
                              interval=500, blit=True, repeat=False)

ani.save('1D_wave_Equation_Example.mp4', writer='ffmpeg', fps=fps)
