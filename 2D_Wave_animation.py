import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sympy as sym

x, n, y, m = sym.symbols('x, n, y, m')

a = 1
b = 1
c = 1
w_nm = sym.sqrt((n * sym.pi / a) ** 2 + (m * sym.pi / b) ** 2)

# initial position function
f = sym.sin(2*(sym.pi * x / a)) * sym.sin(2*(sym.pi * y / b))

# Initial velocity function
g = 0

a_nm = (4 / (a * b)) * sym.integrate(sym.integrate(f * sym.sin((n * sym.pi * x) / a) * sym.sin((m * sym.pi * y) / b), (x, 0, a)), (y, 0, b))
#b_nm = 0  # Assuming no initial velocity, assume a drum tap for next one
b_nm = (4 / (a * b * w_nm)) * sym.integrate(sym.integrate(g * sym.sin((n * sym.pi * x) / a) * sym.sin((m * sym.pi * y) / b), (x, 0, a)), (y, 0, b))

t = sym.symbols('t')
u_symbolic = sym.Sum(sym.Sum((a_nm * sym.cos(c * w_nm * t) + b_nm * sym.sin(c * w_nm * t)) * sym.sin(n * sym.pi * x / a) * sym.sin(m * sym.pi * y / b), (n, 1, 7)), (m, 1, 7))

fps = 30
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x_vals = np.linspace(0, a, 100)
y_vals = np.linspace(0, b, 100)
x_vals, y_vals = np.meshgrid(x_vals, y_vals)

u = sym.lambdify([x, y, t], u_symbolic, modules='numpy')

def init():
    ax.clear()
    ax.set_zlim(-1, 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('U')
    return ax,

def animate(i):
    ax.clear()
    z_vals = u(x_vals, y_vals, i / fps)
    if isinstance(z_vals, int):
        z_vals = np.full_like(x_vals, z_vals)
    ax.plot_surface(x_vals, y_vals, z_vals)
    ax.set_zlim(-1, 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('U')
    return ax,

ani = animation.FuncAnimation(fig, animate, frames=np.arange(0, fps * 10), init_func=init, interval=1000 / fps, blit=False)

ani.save('3D_wave_equation_2.mp4', writer='ffmpeg', fps=fps)