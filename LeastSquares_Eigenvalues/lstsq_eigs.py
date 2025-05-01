# lstsq_eigs.py
"""Volume 1: Least Squares and Computing Eigenvalues.
<Name>
<Class>
<Date>
"""

import numpy as np
from cmath import sqrt
from scipy import linalg as la
from matplotlib import pyplot as plt
#Problem 6
from scipy.linalg import hessenberg, qr, solve_triangular



# Problem 1
def least_squares(A, b):
    """Calculate the least squares solutions to Ax = b by using the QR
    decomposition.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n <= m.
        b ((m, ) ndarray): A vector of length m.

    Returns:
        x ((n, ) ndarray): The solution to the normal equations.
    """
    Q, R = la.qr(A, mode="economic")
    Qt_b = Q.T @ b
    x = la.solve_triangular(R, Qt_b)
    return x

# Problem 2
def line_fit():
    """Find the least squares line that relates the year to the housing price
    index for the data in housing.npy. Plot both the data points and the least
    squares line.
    """
    data = np.load("housing.npy")
    x = data[:, 0]  
    y = data[:, 1]  
    A = np.column_stack((x, np.ones(x.shape))) 
    b = y
    coeffs = least_squares(A, b)  
    plt.plot(x, y, 'o', label="Data") 
    plt.plot(x, A @ coeffs, label="Least Squares Line") 
    plt.title("Least Squares Linear Fit to Housing Price Index")
    plt.xlabel("Year (0 = 2000)")
    plt.ylabel("Price Index")
    plt.legend()
    plt.show()


# Problem 3
def polynomial_fit():
    """Find the least squares polynomials of degree 3, 6, 9, and 12 that relate
    the year to the housing price index for the data in housing.npy. Plot both
    the data points and the least squares polynomials in individual subplots.
    """
    data = np.load("housing.npy")
    x = data[:, 0]
    y = data[:, 1]
    degrees = [3, 6, 9, 12]
    x_smooth = np.linspace(x.min(), x.max(), 500)
    plt.figure(figsize=(10, 8))
    for i, deg in enumerate(degrees):
        A = np.vander(x, deg + 1)  
        coeffs = la.lstsq(A, y)[0]  
        y_fit = np.polyval(coeffs, x_smooth)  
        plt.subplot(2, 2, i + 1)
        plt.plot(x, y, 'o', label="Data")
        plt.plot(x_smooth, y_fit, label=f"Degree {deg} Fit")
        plt.title(f"Least Squares Polynomial (Degree {deg})")
        plt.xlabel("Year (0 = 2000)")
        plt.ylabel("Price Index")
        plt.legend()
    plt.tight_layout()
    plt.show()


def plot_ellipse(a, b, c, d, e):
    """Plot an ellipse of the form ax^2 + bx + cxy + dy + ey^2 = 1."""
    theta = np.linspace(0, 2*np.pi, 200)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    A = a*(cos_t**2) + c*cos_t*sin_t + e*(sin_t**2)
    B = b*cos_t + d*sin_t
    r = (-B + np.sqrt(B**2 + 4*A)) / (2*A)
    plt.plot(r*cos_t, r*sin_t)
    plt.gca().set_aspect("equal", "datalim")

# Problem 4
def ellipse_fit():
    """Calculate the parameters for the ellipse that best fits the data in
    ellipse.npy. Plot the original data points and the ellipse together, using
    plot_ellipse() to plot the ellipse.
    """
    data = np.load("ellipse.npy")
    x = data[:, 0]
    y = data[:, 1]
    A = np.column_stack((x**2, x, x*y, y, y**2))
    b = np.ones_like(x)
    a, b_, c, d, e = la.lstsq(A, b)[0]
    plt.plot(x, y, 'k*', label="Data Points")
    plot_ellipse(a, b_, c, d, e)
    plt.title("Best Fit Ellipse")
    plt.legend()
    plt.show()


# Problem 5
def power_method(A, N=20, tol=1e-12):
    """Compute the dominant eigenvalue of A and a corresponding eigenvector
    via the power method.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The maximum number of iterations.
        tol (float): The stopping tolerance.

    Returns:
        (float): The dominant eigenvalue of A.
        ((n,) ndarray): An eigenvector corresponding to the dominant
            eigenvalue of A.
    """
    n = A.shape[0]
    x = np.random.random(n)
    x = x / np.sqrt(np.sum(x**2))  
    for _ in range(N):
        x_new = A @ x
        x_new = x_new / np.sqrt(np.sum(x_new**2))  
        if np.sqrt(np.sum((x_new - x)**2)) < tol:
            break
        x = x_new
    lambda_approx = x @ (A @ x)
    return lambda_approx, x

# Problem 6
def qr_algorithm(A, N=50, tol=1e-12):
    """Compute the eigenvalues of A via the QR algorithm.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The number of iterations to run the QR algorithm.
        tol (float): The threshold value for determining if a diagonal S_i
            block is 1x1 or 2x2.

    Returns:
        ((n,) ndarray): The eigenvalues of A.
    """
    S = la.hessenberg(A)
    for _ in range(N):
        Q, R = la.qr(S)
        S = R @ Q
    eigs = []
    i = 0
    n = S.shape[0]
    while i < n:
        if i == n - 1 or abs(S[i+1, i]) < tol:
            eigs.append(S[i, i])
            i += 1
        else:
            a, b = S[i, i], S[i, i+1]
            c, d = S[i+1, i], S[i+1, i+1]
            trace = a + d
            det = a * d - b * c
            disc = sqrt(trace**2 - 4 * det)
            eig1 = (trace + disc) / 2
            eig2 = (trace - disc) / 2
            eigs.extend([eig1, eig2])
            i += 2

    return np.array(eigs)

