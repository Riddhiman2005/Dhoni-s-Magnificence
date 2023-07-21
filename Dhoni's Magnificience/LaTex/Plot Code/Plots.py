
import matplotlib.pyplot as plt
import numpy as np
from math import exp, sqrt, pi
from functools import reduce, partial
from collections import deque
from itertools import accumulate

# Function to implement the Runge-Kutta method for solving ordinary differential equations
def runge_kutta(f, x0, y0, h, n):
    """
    Runge-Kutta method for solving ordinary differential equations (ODEs).

    Parameters:
        f: Callable function representing the ODEs. It takes two arguments: t (current time) and h (state vector).
        x0: Initial time.
        y0: Initial state vector.
        h: Step size for the time domain.
        n: Number of steps to be taken.

    Returns:
        xs: List of time points.
        ys: List of state vectors corresponding to each time point.
    """
    def step(xsys, v):
        xs, ys = xsys
        xv = x0 + v * h
        yv = ys[0]
        k1 = f(xv, yv)
        k2 = f(xv + h/2, yv + k1 * h/2)
        k3 = f(xv + h/2, yv + k2 * h/2)
        k4 = f(xv + h, yv + h * k3)
        kv = h/6 * (k1 + 2 * (k2 + k3) + k4)
        y = yv + kv
        xs.appendleft(xv)
        ys.appendleft(y)
        return(xs, ys)
    return reduce(step, range(1, n+1), (deque([x0]), deque([y0])))

# Alternative implementation of Runge-Kutta method using yield and accumulate
def runge_kutta_alt(f, x0, y0, h, n):
    """
    Alternative implementation of Runge-Kutta method using generator and accumulate.

    Parameters:
        f: Callable function representing the ODEs. It takes two arguments: t (current time) and h (state vector).
        x0: Initial time.
        y0: Initial state vector.
        h: Step size for the time domain.
        n: Number of steps to be taken.

    Yields:
        Tuple (t, y) at each time step.
    """
    def step(xy, v):
        _, yv = xy
        x = x0 + v * h
        k1 = f(x, yv)
        k2 = f(x + h/2, yv + k1 * h/2)
        k3 = f(x + h/2, yv + k2 * h/2)
        k4 = f(x + h, yv + h * k3)
        kv = h/6 * (k1 + 2 * (k2 + k3) + k4)
        y = yv + kv
        return (x, y)
    yield from accumulate(range(1, n+1), step, initial=(x0, y0))

# Alternative implementation of Runge-Kutta method with an end time
def runge_kutta_alt2(f, x0, y0, end, n):
    """
    Alternative implementation of Runge-Kutta method using generator and accumulate with an end time.

    Parameters:
        f: Callable function representing the ODEs. It takes two arguments: t (current time) and h (state vector).
        x0: Initial time.
        y0: Initial state vector.
        end: End time for the simulation.
        n: Number of steps to be taken.

    Yields:
        Tuple (t, y) at each time step.
    """
    h = end / n
    yield from runge_kutta_alt(f, x0, y0, h, n)

# Function defining the system of ODEs describing a round object moving through a liquid subject to gravity
def f(kappa, t, h):
    """
    System of ODEs describing a round object moving through a liquid subject to gravity.

    Parameters:
        kappa: Air resistance coefficient.
        t: Current time.
        h: State vector representing the object's position and velocity.

    Returns:
        result: Array representing the object's velocity and acceleration at the current time.
    """
    g = 9.81
    vx, vy = h[2:]
    a = -kappa*(vx**2 + vy**2)
    b = np.arctan(vy/vx)
    return np.array([vx, vy, a * np.cos(b), a * np.sin(b) - g])

# Constants and parameters for the simulation

r = 0.145  # radius in meters
c_w = 0.4  # air resistance coefficient
m = 0.420  # weight of football in kilograms
A = r ** 2 * pi

rho_air = 1.2041  # density of air at 20°C in kg/m³
kappa = (c_w * rho_air * A) / (2 * m)

# Running the simulation using Runge-Kutta method
ts, hs = zip(*runge_kutta_alt2(partial(f, kappa), 0.0, np.array([0.0, 0.0, 28.0, 18.0]), 3.0, 1000))
xs, ys, vxs, vys = np.transpose(np.array(hs))
ys[ys < 0] = 0

# Plotting the results
plt.subplot(2, 1, 1)
plt.plot(xs, ys, label="y(x)")
plt.xlabel("$x$")
plt.ylabel("$y(x)$")
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(ts, xs, label="$x(t)$")
plt.plot(ts, ys, label="$y(t)$")
plt.plot(ts, vxs, label=r"$v_x(t)$ in $\frac{m}{s}$")
plt.plot(ts, vys, label=r"$v_y(t)$ in $\frac{m}{s}$")
plt.xlabel("$t$ in $s$")
plt.legend()

plt.show()
