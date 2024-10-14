#import required packages
import sympy as sym
import sympy.plotting as sym_plot
from IPython.display import display_latex

#initalise printing
sym.init_printing()

# Define the equation
x = sym.symbols('x')
y = sym.Function('y')
eq1 = sym.Eq(y(x).diff(x), (2*y(x)-3*sym.Pow(y(x),2)) * sym.sin(x))
print("The equation: ")
display_latex(eq1)

# Solve the equation with in initial conditions
eq1sol = sym.dsolve(eq1, y(x), ics={y(0):0.5})
print("has solutions: ")
display_latex(eq1sol)
print("or equivalently: ")
simpleEq1Sol = sym.simplify(eq1sol)
display_latex(simpleEq1Sol)


#plot equation
sym_plot.plot(simpleEq1Sol.rhs,xlim=(-10, 10), ylim=(-0.75, 0.75), xlabel = 't', ylabel = 'y', legend=True)