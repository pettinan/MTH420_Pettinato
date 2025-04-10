# python_intro.py
"""Python Essentials: Introduction to Python.
Noah Pettinato
MTH 420
4/3/2025
"""



# Problem 1 (write code below)


# Problem 2
def sphere_volume(r):
    """ Return the volume of the sphere of radius 'r'.
    Use 3.14159 for pi in your computation.
    """
    return (4/3) * 3.14159 * r**3
    #raise NotImplementedError("Problem 2 Incomplete")


# Problem 3
def isolate(a, b, c, d, e):
    """ Print the arguments separated by spaces, but print 5 spaces on either
    side of b.
    """
    print(a, b, c, sep="     ", end=" ")
    print(d, e)
    #raise NotImplementedError("Problem 3 Incomplete")


# Problem 4
def first_half(my_string):
    """ Return the first half of the string 'my_string'. Exclude the
    middle character if there are an odd number of characters.

    Examples:
        >>> first_half("python")
        'pyt'
        >>> first_half("ipython")
        'ipy'
    """
    mid_index = len(my_string) // 2
    return my_string[:mid_index]

    #raise NotImplementedError("Problem 4 Incomplete")

def backward(my_string):
    """ Return the reverse of the string 'my_string'.

    Examples:
        >>> backward("python")
        'nohtyp'
        >>> backward("ipython")
        'nohtypi'
    """
    return my_string[::-1]
    #raise NotImplementedError("Problem 4 Incomplete")


# Problem 5
def list_ops():
    """ Define a list with the entries "bear", "ant", "cat", and "dog".
    Perform the following operations on the list:
        - Append "eagle".
        - Replace the entry at index 2 with "fox".
        - Remove (or pop) the entry at index 1.
        - Sort the list in reverse alphabetical order.
        - Replace "eagle" with "hawk".
        - Add the string "hunter" to the last entry in the list.
    Return the resulting list.

    Examples:
        >>> list_ops()
        ['fox', 'hawk', 'dog', 'bearhunter']
    """
    my_list = ["bear", "ant", "cat", "dog"]
    my_list.append("eagle")
    my_list[2] = "fox"
    my_list.pop(1)
    my_list.sort(reverse = True)
    eagle_index = my_list.index("eagle")
    my_list[eagle_index] = "hawk"
    my_list[-1] += "hunter"
    return my_list
    
    
    #raise NotImplementedError("Problem 5 Incomplete")


# Problem 6
def pig_latin(word):
    """ Translate the string 'word' into Pig Latin, and return the new word.

    Examples:
        >>> pig_latin("apple")
        'applehay'
        >>> pig_latin("banana")
        'ananabay'
    """
    vowels = {'a', 'e', 'i', 'o', 'u'}
    if word[0] in vowels:
        return word + "hay"
    else:
        return word[1:] + word[0] + "ay"
    #raise NotImplementedError("Problem 6 Incomplete")


# Problem 7
def palindrome():
    """ Find and retun the largest palindromic number made from the product
    of two 3-digit numbers.
    """
    max_palindrome = 0
    for i in range(100, 1000):
        for j in range(100, 1000):
            product = i * j
            if str(product) == str(product)[::-1]:
                if product > max_palindrome:
                    max_palindrome = product
    return max_palindrome
    #raise NotImplementedError("Problem 7 Incomplete")

# Problem 8
def alt_harmonic(n):
    """ Return the partial sum of the first n terms of the alternating
    harmonic series, which approximates ln(2).
    """
    return sum([(-1)**(k+1) / k for k in range(1, n+1)])
    #raise NotImplementedError("Problem 8 Incomplete")
    
    
if __name__=="__main__":
    print("Hello, world!")

# Test Problem 2 (sphere volume)
    print("\nTesting Problem 2: Sphere Volume")
    r = 3
    volume = sphere_volume(r)
    print("Volume of sphere with radius", r, ":", volume)
    
# Test Problem 3 (isolate function)
    print("\nTesting Problem 3: Isolate Function:")
    isolate(1, 2, 3, 4, 5)

# Test Problem 4 (first_half)
    print("\nTesting Problem 4: First Half Function")
    my_string = "Python"
    print("first_half('python'):", first_half(my_string))
    
# Test Problem 4 (backward)
    print("\nTesting Problem 4: Backward Function")
    print("backward('python'):", backward(my_string))

# Test Problem 5 (list_ops)
    print("\nTesting Problem 5: List Operations")
    print("\nFinal list:", list_ops())
    
# Test Problem 6 (pig_latin)
    print("\nTesting Problem 6: Pig Latin Conversion")
    print('apple ->', pig_latin("apple"))
    print('banana ->', pig_latin("banana"))
    
# Test Problem 7 (palindrome)
    print("\nTesting Problem 7: Largest Palindrom Product")
    print("Largest palindrome from a product of two 3-digit numbers:", palindrome())
    
# Test Problem 8 (alt_harmonic)
    print("\nTesting Problem 8: Alternating Harmonic Series")
    print("Sum of the first 500,000 terms:", alt_harmonic(500000))