from numba import njit


@njit(cache=True)
def gcd_recursive(a: int, b: int) -> int:
    """
    Computes GCD of a and b
    """
    while b:
        a, b = b, a % b
    return a


@njit(cache=True)
def get_gcd_of_array(arr) -> int:
    """
    Calculates GCD of a vector.
    Returns 1 immediately if any pair gives 1.
    """
    d = len(arr)
    if d == 0:
        return 0
    result = abs(arr[0])
    for i in range(1, d):
        result = gcd_recursive(result, abs(arr[i]))
        if result == 1:
            return 1
    return result
