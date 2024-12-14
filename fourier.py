import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

x,n = sym.symbols('x, n')
fps = 60

#Define function to fourier approximate
f = sym.Piecewise((1/2*x + 1/2, (x<0)), (-1/2*x + 1/2, (x>=0)))

#calculate fourier coefficients
a0 = sym.integrate(f, (x,-1,1))
an = sym.integrate(f*sym.cos(n*sym.pi*x), (x, -1, 1))
bn = sym.integrate(f*sym.sin(n*sym.pi*x), (x, -1, 1))

def approx_fourier(f, L, num_terms):
    a0 = (1/L)*sym.integrate(f, (x,-L,L))
    an = (1/L)*sym.integrate(f*sym.cos((n*sym.pi*x)/L), (x, -L, L))
    bn = (1/L)*sym.integrate(f*sym.sin((n*sym.pi*x)/L), (x, -L, L))
    print(f"A0: {a0}, An: {an}, Bn: {bn}")
    f10 = a0/2 + sym.Sum(an*sym.cos((n*sym.pi*x)/L)+bn*sym.sin((n*sym.pi*x)/L), (n,1,num_terms))
    f10_expr = f10.doit()
    return f10_expr

f_1 = sym.lambdify(x, approx_fourier(f, 1, 4), 'numpy')

fig, ax = plt.subplots()

x_vals = np.arange(-2, 2, 0.01)
line, = ax.plot(x_vals, f_1(x_vals))
plt.xlabel('x')
plt.ylabel('f(x)')
plt.close() # this prevents the first frame being displayed as a static plot

# add an annotation showing the number of terms (this will be updated in each frame)
txt = ax.text(-2, -1, 'terms: 1')

# add a plot of the original function
x_vals_for_f = np.arange(-1, 1, 0.01)
f_num = sym.lambdify(x, f, 'numpy')
ax.plot(x_vals_for_f, f_num(x_vals_for_f))


# define the background of each frame - a "blank slate"
def init():
    line.set_ydata([np.nan] * len(x_vals))
    return line,

def animate(i):
    f_i = sym.lambdify(x, approx_fourier(f, 1, i), 'numpy')
    line.set_ydata(f_i(x_vals))  # update the data.
    txt.set_text('terms: '+str(i)) # update the annotation
    return line, txt  # return all the updated elements

ani = animation.FuncAnimation(
    fig, animate, frames=np.arange(1,50), init_func=init, interval=45, blit=True)

ani.save('fourier_aprox_x**2.mp4', writer='ffmpeg', fps=fps)