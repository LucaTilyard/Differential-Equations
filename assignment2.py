#  defining the function in the RHS of the ODE given in the question
# Import packages
import math
import numpy as np
import matplotlib.pyplot as plt


def eq1_dy_dt(y, t):
    return 4 * t - 3 * y


def timesteps(start, stop, h):
    num_steps = math.ceil((stop - start) / h)
    return np.linspace(start, start + num_steps * h, num_steps + 1)


def ode_Euler(func, times, y0):
    '''
    integrates the system of y' = func(y, t) using forward Euler method
    for the time steps in times and given initial condition y0
    ----------------------------------------------------------
    inputs:
        func: the RHS function in the system of ODE
        times: the points in time (or the span of independent variable in ODE)
        y0: initial condition (make sure the dimension of y0 and func are the same)
    output:
        y: the solution of ODE.
        Each row in the solution array y corresponds to a value returned in column vector t
    '''
    # guess why I put these two lines here?
    times = np.array(times)
    y0 = np.array(y0)
    n = y0.size  # the dimension of ODE
    nT = times.size  # the number of time steps
    y = np.zeros([nT, n])
    y[0, :] = y0
    # loop for timesteps
    for k in range(nT - 1):
        y[k + 1, :] = y[k, :] + (times[k + 1] - times[k]) * func(y[k, :], times[k])
    return y


def ode_AB2(func, initialTime, finalTime, nSteps, y0):
    y0 = np.array(y0)
    n = y0.size  # the dimension of ODE
    dt = (finalTime - initialTime) / nSteps
    times = np.linspace(initialTime, finalTime, nSteps + 1)
    y = np.zeros([nSteps + 1, n])
    y[0, :] = y0
    # First step using Euler
    y[1, :] = y[0, :] + dt * func(y[0, :], times[0])
    # Other steps
    for k in range(1, nSteps):
        y[k + 1, :] = y[k, :] + (1.5 * func(y[k, :], times[k]) - 0.5 * func(y[k - 1, :], times[k - 1])) * dt

    return y, times


def Euler_step(func, start, stop, h, ics):
    times = timesteps(start, stop, h)
    return ode_Euler(func, times, ics), times


def AB2_step(func, start, stop, h, ics):
    nSteps = math.ceil((stop - start) / h)
    return ode_AB2(func, start, stop, nSteps, ics)


def produce_df(method, vectorField, start, stop, h, ics):
    values, times = method(vectorField, start, stop, h, ics)
    return DataFrame(data=values, index=np.round(times, 3), columns=["h=" + str(h)])



from pandas import DataFrame

euler_frame = produce_df(Euler_step, eq1_dy_dt,0, 0.5, 0.05, 1)
AB2_frame = produce_df(AB2_step, eq1_dy_dt,0, 0.5, 0.05, 1)
table = euler_frame.join(AB2_frame, rsuffix='Euler_frame')
print(table)
table.plot()
# plot the results


# standard setup
import sympy as sym
sym.init_printing()
from IPython.display import display_latex
import sympy.plotting as sym_plot

t = sym.symbols('t')
y = sym.Function('y')
eq2 = sym.Eq(y(t).diff(t), 4*t - 3*y(t))
print("The equation: ")
display_latex(eq1)

eq2sol = sym.dsolve(eq2, y(t), ics={y(0):1})
print("has solutions: ")
display_latex(eq2sol)
print("or equivalently: ")
simpleEq2Sol = sym.simplify(eq2sol)
display_latex(simpleEq2Sol)


#plot equation
sym_plot.plot(simpleEq2Sol.rhs,xlim=(-5, 5), ylim=(0, 10), xlabel = 't', ylabel = 'y', legend=True)

# Differentiate the solution
first_derivative = sym.diff(simpleEq2Sol.rhs, t)
print("First Derivitive: ")
display_latex(first_derivative)

solutions = sym.solve(first_derivative,t)
solution = solutions[2]  # eliminate imaginary solutions
print(solution)
y_value = simpleEq2Sol.subs(t, solution)
print("y value: ")
display_latex(y_value)