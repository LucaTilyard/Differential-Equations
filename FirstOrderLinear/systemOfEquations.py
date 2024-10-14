import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Define the system
def system(vars, t):
    x, y = vars
    dxdt = x - x*y
    dydt = -y + x*y
    return [dxdt, dydt]

# Initialise t values
t = np.linspace(0, 5, 1000)

# Setting initial conditions for x and y
x0_values = np.linspace(-15, 15, 30)
y0_values = np.linspace(-15, 15, 30)

# Solve the system for each initial condition of y(0)
for x0 in x0_values:
    for y0 in y0_values:
        initial_conditions = [x0, y0]
        solution = odeint(system, initial_conditions, t)
        plt.plot(solution[:, 0], solution[:, 1], label=y0)

# Set Title and labels for the plot
plt.title('Solutions of our system of equations')
plt.xlabel('x values')
plt.ylabel('y values')

# Set the spines to be positioned at the center
ax = plt.gca()
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')

# Hide the top and right spines
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# Adjust the ticks
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# Set the limit of X and Y axis
ax.set_xlim([-15, 15])
ax.set_ylim([-15, 15])

plt.show()
