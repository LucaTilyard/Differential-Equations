import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import sympy as sym

x, n = sym.symbols('x, n')
cmap = plt.get_cmap('hot')

L = 10
a = 1
f = sym.Piecewise((1, (x > (L/2 - 1)) & (x < (L/2 + 1))), (0, (x <= (L/2 - 1))), (0, (x >= (L/2 + 1))))

cn = sym.Rational(2, L) * sym.integrate(f * sym.cos(n * sym.pi * x / L), (x, 0, L))

t = sym.symbols('t')
u_symbolic = sym.Sum(cn.simplify() * sym.cos(n * sym.pi * x / L) * sym.exp(-((n**2) * (sym.pi**2) * (a**2) * t / L**2)), (n, 1, 300))

fps = 30  # number of frames per second

fig, ax = plt.subplots()
ax.set_facecolor('lightgray')

x_vals = np.linspace(0, L, 200)

u = sym.lambdify([x, t], u_symbolic, modules='numpy')

# set up the initial frame
scat = ax.scatter(x_vals, np.zeros_like(x_vals), c=u(x_vals, 0), cmap=cmap, s=500)
plt.xlabel('x')
plt.ylabel('u')
plt.ylim(-1, 1)
plt.close()

# add an annotation showing the time (this will be updated in each frame)
txt = ax.text(0, 0.9, 't=0')

def init():
    y_vals = np.zeros_like(x_vals)
    scat.set_offsets(np.c_[x_vals, y_vals])
    scat.set_array(u(x_vals, 0))
    return scat,

def animate(i):
    y_vals =  np.zeros_like(x_vals)
    colors = u(x_vals, i / fps)
    scat.set_offsets(np.c_[x_vals, y_vals])
    scat.set_array(colors)
    txt.set_text('t=' + str(i / fps))  # update the annotation
    return scat, txt

ani = animation.FuncAnimation(fig, animate, np.arange(0, fps * 20), init_func=init,
                              interval=500, blit=True, repeat=False)

ani.save('heated_points_new_ic.mp4', writer='ffmpeg', fps=fps)