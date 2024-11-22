"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(a: float, b: float) -> float:
    """Multiplies two numbers

    Args:
    ----
        a (float): float 1 for multiplication operation
        b (float): float 2 for multiplication operation

    Returns:
    -------
        float: multiplied result of a and b

    """
    return a * b


def id(a: float) -> float:
    """Returns the input unchanged

    Args:
    ----
        a (float): input to be returned

    Returns:
    -------
        float: input unchanged

    """
    return a


def add(a: float, b: float) -> float:
    """Adds two numbers

    Args:
    ----
        a (float): float 1 for addition operation
        b (float): float 2 for addition operation

    Returns:
    -------
        float: sum result of a and b

    """
    return a + b


def neg(a: float) -> float:
    """Negates a number

    Args:
    ----
        a (float): number to be negated

    Returns:
    -------
        float: negated input

    """
    return -a


def lt(a: float, b: float) -> float:
    """Checks if one number is less than another

    Args:
    ----
        a (float): float to check if less than float b
        b (float): float to check against for if a is less than

    Returns:
    -------
        float: 1.0 if a is less than b otherwise 0.0

    """
    return 1.0 if a < b else 0.0


def eq(a: float, b: float) -> float:
    """Checks if two numbers are equal

    Args:
    ----
        a (float): float to check if equal to float b
        b (float): float to check is equal to float a

    Returns:
    -------
        float: 1.0 if a equals b otherwise 0.0

    """
    return 1.0 if a == b else 0.0


def max(a: float, b: float) -> float:
    """Returns the larger of two numbers

    Args:
    ----
        a (float): float to check for max between float b
        b (float): float to check for max between float a

    Returns:
    -------
        float: returns a or b depending on which is larger

    """
    return a if a > b else b


def is_close(a: float, b: float) -> float:
    """Checks if two numbers are close in value (within a small value - 1e-2 to handle float precision)

    Args:
    ----
        a (float): float to check for if close to float b
        b (float): float to check for if close to float a

    Returns:
    -------
        float: 1.0 if a is close to b, 0.0 otherwise

    """
    return abs(a - b) < 1e-2


def sigmoid(a: float) -> float:
    """Calculates the sigmoid function for the given input

    Args:
    ----
        a (float): float to calculate the sigmoid function for

    Returns:
    -------
        float: result of sigmoid function with value a

    """
    if a >= 0:
        return 1.0 / (1.0 + math.exp(-a))
    else:
        return math.exp(a) / (1.0 + math.exp(a))


def relu(a: float) -> float:
    """Applies the ReLU activation function on input float

    Args:
    ----
        a (float): float to apply ReLU function to

    Returns:
    -------
        float: result ReLU function on input a

    """
    return a if a > 0 else 0.0


EPS = 1e-6


def log(a: float) -> float:
    """Calculates the natural logarithm for input

    Args:
    ----
        a (float): calculates natural logarithm on input float

    Returns:
    -------
        float: result of log function on input a

    """
    return math.log(a)


def exp(a: float) -> float:
    """Calculates the e^input_float

    Args:
    ----
        a (float): float to use for exponential function

    Returns:
    -------
        float: result of exponential function with value a

    """
    return math.exp(a)


def inv(a: float) -> float:
    """Calculates the reciprocal of an input float

    Args:
    ----
        a (float): input float to calculate reciprocal for

    Returns:
    -------
        float: reciprocal of input a

    """
    return 1.0 / a


def log_back(a: float, b: float) -> float:
    """Computes the derivative of log times a second arg, which is d/dx log(a) * b = 1/a * b

    Args:
    ----
        a (float): input float to calculate derivative of log
        b (float): second float to multiply times result of log derivative of input a

    Returns:
    -------
        float: (derivative of log(a)) * b

    """
    return b / (a + EPS)


def inv_back(a: float, b: float) -> float:
    """Computes the derivative of reciprocal times a second arg, which is d/dx(1/x) * b = -1 / x^2 * b

    Args:
    ----
        a (float): input float to calculate derivative of reciprocal
        b (float): second float to multiply times result of derivatie of reciprocal of input a

    Returns:
    -------
        float: (derivative of (1 / a)) * b

    """
    return -(1.0 / a**2) * b


def relu_back(a: float, b: float) -> float:
    """Computes the derivative of ReLU function times a second arg, which is d/dx(max(0,a)) * b => b when a > 0, 0 otherwise

    Args:
    ----
        a (float): input float to check if greater than 0, and determine output of derative of ReLU function
        b (float): second float to return when a > 0, for the result of derivative of ReLU(a) * b

    Returns:
    -------
        float: derivative of ReLU(a) * b

    """
    return b if a > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(fn: Callable[[float], float], a: Iterable[float]) -> Iterable[float]:
    """Higher-order function that applies a given function to each element of an iterable

    Args:
    ----
        fn (Callable[[float], float]): function to be applied to values in iterable
        a (Iterable[float]): list of values to apply fn to

    Returns:
    -------
        Iterable[float]: result of applying fn to iterable list

    """
    return [fn(x) for x in a]


def zipWith(
    fn: Callable[[float, float], float], a: Iterable[float], b: Iterable[float]
) -> Iterable[float]:
    """Higher-order function that combines elements from two iterables using a given function

    Args:
    ----
        fn (Callable[[float, float], float]): function to combine elements in lists a,b
        a (Iterable[float]): list of floats to combine with elements in other list
        b (Iterable[float]): list of floats to combine with elements in other list

    Returns:
    -------
        Iterable[float]: combined result of applying function to lists a and b

    """
    return [fn(x, y) for x, y in zip(a, b)]


def reduce(
    fn: Callable[[float, float], float], a: Iterable[float], initial: float
) -> float:
    """Higher-order function that reduces an iterable to a single value using a given function

    Args:
    ----
        fn (Callable[[float, float], float]): function used to reduce values in iterable to a single value
        a (Iterable[float]): list of values to reduce using fn function
        initial (float): initialization value for the reduction result, and return value if a is empty

    Returns:
    -------
        float: result of reducing values in a using function fn. initial is returned if a is empty

    """
    it = iter(a)

    result = initial
    for x in it:
        result = fn(result, x)

    return result


def negList(a: Iterable[float]) -> Iterable[float]:
    """Negate all elements in a list

    Args:
    ----
        a (Iterable[float]): list to negate

    Returns:
    -------
        Iterable[float]: list with all elements of a negated

    """
    return map(neg, a)


def addLists(a: Iterable[float], b: Iterable[float]) -> Iterable[float]:
    """Given two lists, add corresponding elements together

    Args:
    ----
        a (Iterable[float]): first list to add to list b
        b (Iterable[float]): second list to add to list a

    Returns:
    -------
        Iterable[float]: result of adding corresponding elements between list a and b

    """
    return zipWith(add, a, b)


def sum(a: Iterable[float]) -> float:
    """Sum up all elements in a list

    Args:
    ----
        a (Iterable[float]): list of floats to sum up

    Returns:
    -------
        float: sum result of all elements in list a

    """
    return reduce(add, a, 0)


def prod(a: Iterable[float]) -> float:
    """Product of all elements in a list

    Args:
    ----
        a (Iterable[float]): list of floats to multiply

    Returns:
    -------
        float: product result of multiplying elements in list a

    """
    return reduce(mul, a, 1)
