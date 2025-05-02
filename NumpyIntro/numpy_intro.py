# numpy_intro.py
"""Python Essentials: Intro to NumPy.
Noah Pettinato
MTH 420
4/13/2025
"""

import numpy as np


def prob1():
    """ Define the matrices A and B as arrays. Return the matrix product AB. """
    A = np.array([[3, -1, 4], [1, 5, -9]])
    B = np.array([[2, 6, -5, 3], [5, -8, 9, 7], [9, -3, -2, -3]])
    return A @ B


def prob2():
    """ Define the matrix A as an array. Return the matrix -A^3 + 9A^2 - 15A. """
    A = np.array([[3, 1, 4], [1, 5, 9], [-5, 3, 1]])
    A2 = A @ A
    A3 = A @ A2
    return -A3 + (9 * A2) - (15 * A) 


def prob3():
    """ Define the matrices A and B as arrays using the functions presented in
    this section of the manual (not np.array()). Calculate the matrix product ABA,
    change its data type to np.int64, and return it.
    """
    A = np.triu(np.ones((7, 7)))
    B = -1 * np.ones((7, 7))
    B[np.triu_indices(7, k=1)] = 5
    result = A @ B @ A
    return result.astype(np.int64)
   

def prob4(A):
    """ Make a copy of 'A' and use fancy indexing to set all negative entries of
    the copy to 0. Return the resulting array.

    Example:
        >>> A = np.array([-3,-1,3])
        >>> prob4(A)
        array([0, 0, 3])
    """
    A_copy = A.copy()
    mask = A_copy < 0
    A_copy[mask] = 0
    return A_copy


def prob5():
    """ Define the matrices A, B, and C as arrays. Use NumPy's stacking functions
    to create and return the block matrix:
                                | 0 A^T I |
                                | A  0  0 |
                                | B  0  C |
    where I is the 3x3 identity matrix and each 0 is a matrix of all zeros
    of the appropriate size.
    """
    A = np.array([[0, 2, 4], [1, 3, 5]])
    B = np.array([[3, 0, 0], [3, 3, 0], [3, 3, 3]])
    C = np.array([[-2, 0, 0], [0, -2, 0], [0, 0, -2]])
    I = np.eye(3)
    Z3x3 = np.zeros((3, 3))
    Z2x2 = np.zeros((2, 2))
    Z2x3 = np.zeros((2, 3))
    Z3x2 = np.zeros((3, 2))
    top_row = np.hstack((Z3x3, A.T, I))    
    mid_row = np.hstack((A, Z2x2, Z2x3))
    bot_row = np.hstack((B, Z3x2, C))
    block_matrix = np.vstack((top_row, mid_row, bot_row))
    return block_matrix
    


def prob6(A):
    """ Divide each row of 'A' by the row sum and return the resulting array.
    Use array broadcasting and the axis argument instead of a loop.

    Example:
        >>> A = np.array([[1,1,0],[0,1,0],[1,1,1]])
        >>> prob6(A)
        array([[ 0.5       ,  0.5       ,  0.        ],
               [ 0.        ,  1.        ,  0.        ],
               [ 0.33333333,  0.33333333,  0.33333333]])
    """
    row_sums = A.sum(axis=1).reshape((-1, 1))
    return A / row_sums

def prob7():
    """ Given the array stored in grid.npy, return the greatest product of four
    adjacent numbers in the same direction (up, down, left, right, or
    diagonally) in the grid. Use slicing, as specified in the manual.
    """
    grid = np.load("grid.npy")
    horiz = grid[:, :-3] * grid[:, 1:-2] * grid[:, 2:-1] * grid[:, 3:]
    vert = grid[:-3, :] * grid[1:-2, :] * grid[2:-1, :] * grid[3:, :]
    diag_r = grid[:-3, :-3] * grid[1:-2, 1:-2] * grid[2:-1, 2:-1] * grid[3:, 3:]
    diag_l = grid[:-3, 3:] * grid[1:-2, 2:-1] * grid[2:-1, 1:-2] * grid[3:, :-3]
    return np.max([horiz.max(), vert.max(), diag_r.max(), diag_l.max()]) 


if __name__ == "__main__":
    print("Testing Problem 1: Matrix Product AB")
    print(prob1())

    print("\nTesting Problem 2: Expression -A^3 + 9A^2 - 15A")
    print(prob2())

    print("\nTesting Problem 3: Matrix Product ABA with Modified B")
    print(prob3())

    print("\nTesting Problem 4: Zeroing Negative Entries")
    test_array = np.array([-3, -1, 3])
    print("Original:", test_array)
    print("Modified:", prob4(test_array))

    print("\nTesting Problem 5: Block Matrix Construction")
    block_matrix = prob5()
    print(block_matrix)

    print("\nTesting Problem 6: Row Normalization")
    test_matrix = np.array([[1, 1, 0], [0, 1, 0], [1, 1, 1]])
    print("Original:\n", test_matrix)
    print("Normalized:\n", prob6(test_matrix))

    print("\nTesting Problem 7: Max Product of 4 Adjacent Numbers in Grid")
    print("Max Product:", prob7())
    
    
    bm = prob5()
    print("Shape:", bm.shape)
    print(bm)
