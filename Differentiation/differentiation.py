# differentiation.py
"""Volume 1: Differentiation.
Noah Pettinato
MTH 420
5/28/2025
"""

import time
import numpy as np
import sympy as sy
from matplotlib import pyplot as plt

from jax import numpy as jnp
from jax import grad


# Problem 1
def prob1():
    """Return the derivative of (sin(x) + 1)^sin(cos(x)) using SymPy."""
    x = sy.symbols('x')
    f = (sy.sin(x) + 1)**sy.sin(sy.cos(x))
    df = sy.diff(f, x)
    fprime = sy.lambdify(x, df, modules=['numpy'])
    return fprime

# Problem 2
def fdq1(f, x, h=1e-5):
    """Calculate the first order forward difference quotient of f at x."""
    return (f(x + h) - f(x)) / h

    
def fdq2(f, x, h=1e-5):
    """Calculate the second order forward difference quotient of f at x."""
    return (-f(x + 2*h) + 4*f(x + h) - 3*f(x)) / (2*h)

def bdq1(f, x, h=1e-5):
    """Calculate the first order backward difference quotient of f at x."""
    return (f(x) - f(x - h)) / h

def bdq2(f, x, h=1e-5):
    """Calculate the second order backward difference quotient of f at x."""
    return (3*f(x) - 4*f(x - h) + f(x - 2*h)) / (2*h)

def cdq2(f, x, h=1e-5):
    """Calculate the second order centered difference quotient of f at x."""
    return (f(x + h) - f(x - h)) / (2*h)

def cdq4(f, x, h=1e-5):
    """Calculate the fourth order centered difference quotient of f at x."""
    return (-f(x + 2*h) + 8*f(x + h) - 8*f(x - h) + f(x - 2*h)) / (12*h)


# Problem 3
def prob3(x0):
    """Let f(x) = (sin(x) + 1)^(sin(cos(x))). Use prob1() to calculate the
    exact value of f'(x0). Then use fdq1(), fdq2(), bdq1(), bdq2(), cdq1(),
    and cdq2() to approximate f'(x0) for h=10^-8, 10^-7, ..., 10^-1, 1.
    Track the absolute error for each trial, then plot the absolute error
    against h on a log-log scale.

    Parameters:
        x0 (float): The point where the derivative is being approximated.
    """
    fprime_exact = prob1()
    exact = fprime_exact(x0)
    f = lambda x: (np.sin(x) + 1)**np.sin(np.cos(x))
    hs = np.logspace(-8, 0, 9)
    errs = {
        'FDQ1': [],
        'FDQ2': [],
        'BDQ1': [],
        'BDQ2': [],
        'CDQ2': [],
        'CDQ4': []
    }
    for h in hs:
        errs['FDQ1'].append(abs(fdq1(f, x0, h) - exact))
        errs['FDQ2'].append(abs(fdq2(f, x0, h) - exact))
        errs['BDQ1'].append(abs(bdq1(f, x0, h) - exact))
        errs['BDQ2'].append(abs(bdq2(f, x0, h) - exact))
        errs['CDQ2'].append(abs(cdq2(f, x0, h) - exact))
        errs['CDQ4'].append(abs(cdq4(f, x0, h) - exact))
    for k in errs:
        errs[k] = np.array(errs[k])
    plt.figure()
    plt.loglog(hs, errs['FDQ1'], marker='o', label='Order 1 forward')
    plt.loglog(hs, errs['FDQ2'], marker='o', label='Order 2 forward')
    plt.loglog(hs, errs['BDQ1'], marker='o', label='Order 1 backward')
    plt.loglog(hs, errs['BDQ2'], marker='o', label='Order 2 backward')
    plt.loglog(hs, errs['CDQ2'], marker='o', label='Order 2 centered')
    plt.loglog(hs, errs['CDQ4'], marker='o', label='Order 4 centered')
    plt.xlabel('h')
    plt.ylabel('Absolute Error')
    plt.title(f'Error vs step size at x₀ = {x0}')
    plt.legend()
    plt.show()

# Problem 4
def prob4():
    """The radar stations A and B, separated by the distance 500m, track a
    plane C by recording the angles alpha and beta at one-second intervals.
    Your goal, back at air traffic control, is to determine the speed of the
    plane.

    Successive readings for alpha and beta at integer times t=7,8,...,14
    are stored in the file plane.npy. Each row in the array represents a
    different reading; the columns are the observation time t, the angle
    alpha (in degrees), and the angle beta (also in degrees), in that order.
    The Cartesian coordinates of the plane can be calculated from the angles
    alpha and beta as follows.

    x(alpha, beta) = a tan(beta) / (tan(beta) - tan(alpha))
    y(alpha, beta) = (a tan(beta) tan(alpha)) / (tan(beta) - tan(alpha))

    Load the data, convert alpha and beta to radians, then compute the
    coordinates x(t) and y(t) at each given t. Approximate x'(t) and y'(t)
    using a first order forward difference quotient for t=7, a first order
    backward difference quotient for t=14, and a second order centered
    difference quotient for t=8,9,...,13. Return the values of the speed at
    each t.
    """
    data = np.load('plane.npy') 
    a = 500.0
    alpha = np.deg2rad(data[:,1])
    beta  = np.deg2rad(data[:,2])
    t_alpha = np.tan(alpha)
    t_beta  = np.tan(beta)
    denom   = t_beta - t_alpha
    xs = a * t_beta / denom
    ys = a * t_beta * t_alpha / denom
    n = len(xs)   
    speeds = np.empty(n)
    for i in range(n):
        if i == 0:
            dx = xs[1] - xs[0]
            dy = ys[1] - ys[0]
        elif i == n-1:
            dx = xs[-1] - xs[-2]
            dy = ys[-1] - ys[-2]
        else:
            dx = (xs[i+1] - xs[i-1]) / 2.0
            dy = (ys[i+1] - ys[i-1]) / 2.0
        speeds[i] = np.hypot(dx, dy)
    return speeds

# Problem 5
def jacobian_cdq2(f, x, h=1e-5):
    """Approximate the Jacobian matrix of f:R^n->R^m at x using the second
    order centered difference quotient.

    Parameters:
        f (function): the multidimensional function to differentiate.
            Accepts a NumPy (n,) ndarray and returns an (m,) ndarray.
            For example, f(x,y) = [x+y, xy**2] could be implemented as follows.
            >>> f = lambda x: np.array([x[0] + x[1], x[0] * x[1]**2])
        x ((n,) ndarray): the point in R^n at which to compute the Jacobian.
        h (float): the step size in the finite difference quotient.

    Returns:
        ((m,n) ndarray) the Jacobian matrix of f at x.
    """
    raise NotImplementedError("Problem 5 Incomplete")


# Problem 6
def cheb_poly(x, n):
    """Compute the nth Chebyshev polynomial at x.

    Parameters:
        x (jax.ndarray): the points to evaluate T_n(x) at.
        n (int): The degree of the polynomial.
    """
    raise NotImplementedError("Problem 6 Incomplete")

def prob6():
    """Use JAX and cheb_poly() to create a function for the derivative
    of the Chebyshev polynomials, and use that function to plot the derivatives
    over the domain [-1,1] for n=0,1,2,3,4.
    """
    raise NotImplementedError("Problem 6 Incomplete")


# Problem 7
def prob7(N=200):
    """
    Let f(x) = (sin(x) + 1)^sin(cos(x)). Perform the following experiment N
    times:

        1. Choose a random value x0.
        2. Use prob1() to calculate the "exact" value of f′(x0). Time how long
            the entire process takes, including calling prob1() (each
            iteration).
        3. Time how long it takes to get an approximation of f'(x0) using
            cdq4(). Record the absolute error of the approximation.
        4. Time how long it takes to get an approximation of f'(x0) using
            JAX (calling grad() every time). Record the absolute error of
            the approximation.

    Plot the computation times versus the absolute errors on a log-log plot
    with different colors for SymPy, the difference quotient, and JAX.
    For SymPy, assume an absolute error of 1e-18.
    """
    raise NotImplementedError("Problem 7 Incomplete")

