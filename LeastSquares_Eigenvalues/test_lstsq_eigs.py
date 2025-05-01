"""Unit testing file for Least Squares and Computing Eigenvalues problem 6"""


import lstsq_eigs
import numpy as np
# Problem 6
from scipy import linalg as la


def test_qr_algorithm():
    """
    Write at least one unit test for problem 6, the qr algorithm function.
    """
    A = np.array([[2, 1], [1, 2]])
    eigs_qr = lstsq_eigs.qr_algorithm(A)
    eigs_scipy = la.eigvals(A)
    eigs_qr_sorted = np.sort_complex(eigs_qr)
    eigs_scipy_sorted = np.sort_complex(eigs_scipy)
    assert np.allclose(eigs_qr_sorted, eigs_scipy_sorted, atol=1e-8), \
        f"Mismatch: Your eigenvalues = {eigs_qr_sorted}, SciPy = {eigs_scipy_sorted}"
    print("test_qr_algorithm passed.")

def test_power_method():
    #Sets up test cases
    A = np.array([[1, 1], [1, 1]])
    B = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    C = np.array([[2, 2], [1, 3]])
    
    Aval, Avec = lstsq_eigs.power_method(A)
    Bval, Bvec = lstsq_eigs.power_method(B)
    Cval, Cvec = lstsq_eigs.power_method(C)
    
    #Checks if it finds the appropriate eigenvalue
    assert abs(Aval - 2) < 1e-5, "Incorrect eigenvalue"
    assert abs(Bval - 3) < 1e-5, "Incorrect eigenvalue"
    assert abs(Cval - 4) < 1e-5, "Incorrect eigenvalue"
    
    #Checks if it finds an eigenvector that works
    assert np.linalg.norm(A @ Avec - Aval * Avec) < 1e-3, "Incorrect vector"
    assert np.linalg.norm(B @ Bvec - Bval * Bvec) < 1e-3, "Incorrect vector"
    assert np.linalg.norm(C @ Cvec - Cval * Cvec) < 1e-3, "Incorrect vector"
    
    print("test_power_method passed.")

    
if __name__ == "__main__":
    test_power_method()
    test_qr_algorithm()
