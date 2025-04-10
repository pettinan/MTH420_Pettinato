# standard_library.py
"""Python Essentials: The Standard Library.
Noah Pettinato
MTH 420
4/10/2025
"""

import calculator as calc
from itertools import combinations 


# Problem 1
def prob1(L):
    """Return the minimum, maximum, and average of the entries of L
    (in that order, separated by a comma).
    """
    return min(L), max(L), sum(L) / len(L)


# Problem 2
def prob2():
    """Determine which Python objects are mutable and which are immutable.
    Test integers, strings, lists, tuples, and sets. Print your results.
    """
    print("Testing int:")
    int_1 = 5
    int_2 = int_1
    int_2 += 1
    is_equal = int_1 == int_2
    print("int_1 == int_2:", is_equal)
    if is_equal:
        print("Conclusion: int is mutable\n")
    else:
        print("Conclusion: int is immutable\n")

    print("Testing str:")
    str_1 = "hello"
    str_2 = str_1
    str_2 += "!"
    is_equal = str_1 == str_2
    print("str_1 == str_2:", is_equal)
    if is_equal:
        print("Conclusion: str is mutable\n")
    else:
        print("Conclusion: str is immutable\n")

    print("Testing list:")
    list_1 = [1, 2, 3]
    list_2 = list_1
    list_2.append(4)
    is_equal = list_1 == list_2
    print("list_1 == list_2:", is_equal)
    if is_equal:
        print("Conclusion: list is mutable\n")
    else:
        print("Conclusion: list is immutable\n")

    print("Testing tuple:")
    tuple_1 = (1, 2, 3)
    tuple_2 = tuple_1
    tuple_2 += (4,)
    is_equal = tuple_1 == tuple_2
    print("tuple_1 == tuple_2:", is_equal)
    if is_equal:
        print("Conclusion: tuple is mutable\n")
    else:
        print("Conclusion: tuple is immutable\n")

    print("Testing set:")
    set_1 = {1, 2, 3}
    set_2 = set_1
    set_2.add(4)
    is_equal = set_1 == set_2
    print("set_1 == set_2:", is_equal)
    if is_equal:
        print("Conclusion: set is mutable\n")
    else:
        print("Conclusion: set is immutable\n")



# Problem 3
def hypot(a, b):
    """Calculate and return the length of the hypotenuse of a right triangle.
    Do not use any functions other than sum(), product() and sqrt() that are
    imported from your 'calculator' module.

    Parameters:
        a: the length one of the sides of the triangle.
        b: the length the other non-hypotenuse side of the triangle.
    Returns:
        The length of the triangle's hypotenuse.
    """
    return calc.sqrt(
        calc.sum(
            calc.product(a, a), calc.product(b, b)
        )
    )

# Problem 4
def power_set(A):
    """Use itertools to compute the power set of A.

    Parameters:
        A (iterable): a str, list, set, tuple, or other iterable collection.

    Returns:
        (list(sets)): The power set of A as a list of sets.
    Question: Why not 'set of sets'
    Answer: Sets must contain immutable items, and sets themselves are mutable, so you can't put a set in a set.
    """
    A = list(A)
    powerset = []
    for r in range(len(A)+1):
        for subset in combinations(A, r):
            powerset.append(set(subset))
    return powerset

    


# Problem 5: Implement shut the box.
def shut_the_box(player, timelimit):
    """Play a single game of shut the box."""
    raise NotImplementedError("Problem 5 Incomplete")
 
    
if __name__=="__main__":

# Test Problem 1 (min, max, average"
    print("\nTesting Problem 1: min, max, and average of a list")
    test_list = [2, 4, 6, 8, 10]
    result = prob1(test_list)
    print("Input list:", test_list)
    print("Minimum:", result[0], "Maximum:", result[1], "Average:", result[2])
    
# Test Problem 2 (mutability experiment)
    print("\nTesting Problem 2:")
    prob2()
    
# Test Problem 3 (hypotenuse calculator using calculator.py
    print("\nTesting problem 3: Hypotentuse Calculator")
    a = 6
    b = 8
    h = hypot(a,b)
    print("Sides:", a, b, "-> Hypotenuse:", h)
    
# Test Problem 4 (power set using itertools.combinations)
    print("\nTesting Problem 4: Power Set")
    A = ['a', 'b', 'c']
    result = power_set(A)
    print("Input:, A")
    print("Power set:")
    for subset in result:
        print(subset)