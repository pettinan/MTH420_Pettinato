# cvxpy_intro.py
"""Volume 2: Intro to CVXPY.
Noah Pettinato
MTH 420
5/24/2025
"""

import numpy as np
import cvxpy as cp

def prob1():
    """Solve the following convex optimization problem:

    minimize        2x + y + 3z
    subject to      x  + 2y         <= 3
                         y   - 4z   <= 1
                    2x + 10y + 3z   >= 12
                    x               >= 0
                          y         >= 0
                                z   >= 0

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    x = cp.Variable(3, nonneg=True)
    obj = cp.Minimize(2*x[0] + x[1] + 3*x[2])
    cons = [
        x[0] + 2*x[1]       <= 3,
               x[1] - 4*x[2] <= 1,
        2*x[0] + 10*x[1] +  3*x[2] >= 12
    ]
    prob = cp.Problem(obj, cons)
    prob.solve()
    return x.value, prob.value


# Problem 2
def l1Min(A, b):
    """Calculate the solution to the optimization problem

        minimize    ||x||_1
        subject to  Ax = b

    Parameters:
        A ((m,n) ndarray)
        b ((m, ) ndarray)

    Returns:
        The optimizer x (ndarray)
        The optimal value (float)
    """
    n = A.shape[1]
    x = cp.Variable(n)
    obj = cp.Minimize(cp.norm1(x))
    cons = [A @ x == b]
    prob = cp.Problem(obj, cons)
    prob.solve()
    return x.value, prob.value


# Problem 3
def prob3():
    """Solve the transportation problem by converting the last equality constraint
    into inequality constraints.

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    x = cp.Variable(6, nonneg=True)
    c = np.array([4, 7, 6, 8, 8, 9])
    objective = cp.Minimize(c @ x)
    constraints = [
        x[0] + x[1] <= 7,
        x[2] + x[3] <= 2,
        x[4] + x[5] <= 4,
        x[0] + x[2] + x[4] >= 5,
        x[1] + x[3] + x[5] >= 8,
    ]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    return x.value, prob.value

# Problem 4
def prob4():
    """Find the minimizer and minimum of

    g(x,y,z) = (3/2)x^2 + 2xy + xz + 2y^2 + 2yz + (3/2)z^2 + 3x + z

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    x = cp.Variable(3)
    Q = np.array([
        [3, 2, 1],
        [2, 4, 2],
        [1, 2, 3]
    ])
    r = np.array([3, 0, 1])
    obj = cp.Minimize(0.5 * cp.quad_form(x, Q) + r @ x)
    prob = cp.Problem(obj, [])
    prob.solve()
    return x.value, prob.value


# Problem 5
def prob5(A, b):
    """Calculate the solution to the optimization problem
        minimize    ||Ax - b||_2
        subject to  ||x||_1 == 1
                    x >= 0
    Parameters:
        A ((m,n), ndarray)
        b ((m,), ndarray)
        
    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    n = A.shape[1]
    x = cp.Variable(n, nonneg=True)
    obj = cp.Minimize(cp.norm2(A @ x - b))
    cons = [cp.sum(x) == 1]
    prob = cp.Problem(obj, cons)
    prob.solve()
    return x.value, prob.value

# Problem 6
def prob6():
    """Solve the college student food problem. Read the data in the file 
    food.npy to create a convex optimization problem. The first column is 
    the price, second is the number of servings, and the rest contain
    nutritional information. Use cvxpy to find the minimizer and primal 
    objective.
    
    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """	 
    raise NotImplementedError("Problem 6 Incomplete")
    
