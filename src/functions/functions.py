import numpy as np

def sphere(x):
    return np.sum(x**2)


def cigar(x):
    return x[0] ** 2 + 1e6 * np.sum(x[1:] ** 2)


def discus(x):
    return 1e6 * x[0] ** 2 + np.sum(x[1:] ** 2)

def elipsoid(x):
    n = len(x)
    exponents = 6 * (np.arange(n) / (n - 1))
    return np.sum(10 ** (exponents * x**2))

def rosenbrock(x):
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)

def ackley(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = len(x)
    sum_sq = np.sum(x**2)
    cos_sum = np.sum(np.cos(c * x))
    return -a * np.exp(-b * np.sqrt(sum_sq / d)) - np.exp(cos_sum / d) + a + np.exp(1)
