import numpy as np
import opfunu
from opfunu.cec_based.cec2014 import F232014
import matplotlib.pyplot as plt

def Ufun(x, a, b, c):
    # Element-wise operations to match input shape
    return a * np.sin(x) + b * np.cos(x) + c

# def ackley(x):
#     a = 20
#     b = 0.2
#     c = 2 * np.pi
#     sum1 = np.sum(x**2)
#     sum2 = np.sum(np.cos(c * x))
#     return -a * np.exp(-b * np.sqrt(sum1 / len(x))) - np.exp(sum2 / len(x)) + a + np.exp(1)
#
# def rastrigin(x):
#     A = 10
#     return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))
#
# def weierstrass(x, a=0.5, b=3):
#     sum1 = 0
#     sum2 = 0
#     for k in range(20):
#         sum1 += a**k * np.cos(b**k * (x + 0.5))
#         sum2 += a**k * np.cos(b**k * 0.5)
#     return sum1 - sum2
#
# def griewank(x):
#     return 1 + (np.sum(x**2) / 4000) - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
#
# def sphere(x):
#     return np.sum(x**2)
def ackley(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))
    return -a * np.exp(-b * np.sqrt(sum1 / len(x))) - np.exp(sum2 / len(x)) + a + np.exp(1)

def rastrigin(x):
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def weierstrass(x, a=0.5, b=3):
    sum1 = np.sum([a**k * np.cos(b**k * (x_elem + 0.5)) for k in range(20) for x_elem in x])
    sum2 = np.sum([a**k * np.cos(b**k * 0.5) for k in range(20)])
    return sum1 - sum2

def griewank(x):
    return 1 + (np.sum(x**2) / 4000) - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))

def sphere(x):
    return np.sum(x**2)



# Example test functions (add your F1, F2, etc. implementations)
def F1(x):
    return np.sum(x ** 2)

def F2(x):
    return np.sum(np.abs(x)) + np.prod(np.abs(x))


def F3(x):
    return np.sum(np.abs(x) ** (np.arange(1, len(x) + 1)))


def F4(x):
    return np.sum(np.cumsum(x) ** 2)


def F5(x):
    return np.max(np.abs(x))


def F6(x):
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2)


def F7(x):
    return np.sum(np.abs(x + 0.5) ** 2)


def F8(x):
    return np.sum(np.arange(1, len(x) + 1) * x ** 4) + np.random.rand()


def F9(x):
    return np.sum(x**2) + (np.sum(0.5 * np.arange(1, len(x) + 1) * x))**2 + (np.sum(0.5 * np.arange(1, len(x) + 1) * x))**4

def F10(x):
    return np.sum(-x * np.sin(np.sqrt(np.abs(x))))

def F11(x):
    return 1 + np.sum(np.sin(x) ** 2) - np.exp(-np.sum(x ** 2))

def F12(x):
    return 0.5 * np.sum(x ** 4 - 16 * x ** 2 + 5 * x)

def F13(x):
    return np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)) + 10 * len(x)

def F14(x):
    return -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / len(x))) - np.exp(
        np.sum(np.cos(2 * np.pi * x)) / len(x)) + 20 + np.exp(1)

def F15(x):
    return np.sum(x ** 2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1)))) + 1

def F16(x):
    return (np.sum(np.sin(x) ** 2) - np.exp(-np.sum(x ** 2))) * np.exp(-np.sum(np.sin(np.sqrt(np.abs(x))) ** 2))

def F17(x):
    dim = len(x)
    return (np.pi / dim) * (10 * ((np.sin(np.pi * (1 + (x[0] + 1) / 4)))**2) +
           np.sum(((x[:-1] + 1) / 4)**2 * (1 + 10 * (np.sin(np.pi * (1 + (x[1:] + 1) / 4))**2))) +
           ((x[-1] + 1) / 4)**2) + np.sum(Ufun(x, 10, 100, 4))

def F18(x):
    dim = len(x)
    return 0.1 * ((np.sin(3 * np.pi * x[0])) ** 2 +
                  np.sum((x[:-1] - 1) ** 2 * (1 + (np.sin(3 * np.pi * x[1:])) ** 2)) +
                  (x[-1] - 1) ** 2 * (1 + (np.sin(2 * np.pi * x[-1])) ** 2)) + np.sum(Ufun(x, 5, 100, 4))

def F19(x):
    aS = np.array([[-32, -16, 0, 16, 32],
                   [-32, -32, -32, -32, -32],
                   [-16, -16, -16, -16, -16],
                   [0, 0, 0, 0, 0],
                   [16, 16, 16, 16, 16],
                   [32, 32, 32, 32, 32]])
    bS = np.sum((x[:, np.newaxis] - aS) ** 6, axis=0)
    return (1 / 500 + np.sum(1 / (np.arange(1, 26) + bS))) ** (-1)

def F20(x):
    aK = np.array([0.1957, 0.1947, 0.1735, 0.16, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246])
    bK = 1 / np.array([0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16])
    return np.sum((aK - ((x[0] * (bK ** 2 + x[1] * bK)) / (bK ** 2 + x[2] * bK + x[3]))) ** 2)

def F21(x):
    return 4 * (x[0] ** 2) - 2.1 * (x[0] ** 4) + (x[0] ** 6) / 3 + x[0] * x[1] - 4 * (x[1] ** 2) + 4 * (
                x[1] ** 4)

def F22(x):
    aSH = np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6],
                    [3, 7, 3, 7], [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1],
                    [6, 2, 6, 2], [7, 3.6, 7, 3.6]])
    cSH = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])

    o = 0
    for i in range(5):
        o -= ((x - aSH[i]) @ (x - aSH[i]) + cSH[i]) ** (-1)
    return o

def F23(x):
    aSH = np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6],
                    [3, 7, 3, 7], [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1],
                    [6, 2, 6, 2], [7, 3.6, 7, 3.6]])
    cSH = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])

    o = 0
    for i in range(7):
        o -= ((x - aSH[i]) @ (x - aSH[i]) + cSH[i]) ** (-1)
    return o

def F24(x):
    aSH = np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6],
                    [3, 7, 3, 7], [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1],
                    [6, 2, 6, 2], [7, 3.6, 7, 3.6]])
    cSH = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])

    o = 0
    for i in range(10):
        o -= ((x - aSH[i]) @ (x - aSH[i]) + cSH[i]) ** (-1)
    return o

def F25(x):
    # Check that the input x is within the specified bounds
    if np.any(x < -5) or np.any(x > 5):
        raise ValueError("Input values must be within the range [-5, 5].")

    # Sigma and Lambda values
    sigma = np.ones(10)
    lambda_vals = np.full(10, 5 / 100)

    # Calculate the objective value using the Sphere function
    return sum(F1(x) * l for l in lambda_vals)

def F26(x):
    # Check that the input x is within the specified bounds
    if np.any(x < -5) or np.any(x > 5):
        raise ValueError("Input values must be within the range [-5, 5].")

    # Sigma and Lambda values
    sigma = np.ones(10)
    lambda_vals = np.full(10, 5 / 100)

    # Calculate the objective value using Griewank’s function
    return sum(F15(x) * l for l in lambda_vals)

def F27(x):
    # Check that the input x is within the specified bounds
    if np.any(x < -5) or np.any(x > 5):
        raise ValueError("Input values must be within the range [-5, 5].")

    # Sigma and Lambda values
    sigma = np.ones(10)
    lambda_vals = np.ones(10)  # All lambda values set to 1

    # Calculate the objective value using Griewank’s function
    return sum(F15(x) * l for l in lambda_vals)

def F28(x):

    return F232014(ndim=30)

def F29(x):
    # Check that the input x is within the specified bounds
    if np.any(x < -5) or np.any(x > 5):
        raise ValueError("Input values must be within the range [-5, 5].")

    # Sigma and Lambda values
    sigma = np.ones(10)
    lambda_vals = np.array([1 / 5, 1 / 5, 5 / 0.5, 5 / 0.5, 5 / 100, 5 / 100, 5 / 32, 5 / 32, 5 / 100, 5 / 100])

    # Calculate the objective value using the respective functions
    return (lambda_vals[0] * rastrigin(x[:2]) +
            lambda_vals[1] * rastrigin(x[:2]) +
            lambda_vals[2] * weierstrass(x[2:4]) +
            lambda_vals[3] * weierstrass(x[2:4]) +
            lambda_vals[4] * griewank(x[4:6]) +
            lambda_vals[5] * griewank(x[4:6]) +
            lambda_vals[6] * ackley(x[6:8]) +
            lambda_vals[7] * ackley(x[6:8]) +
            lambda_vals[8] * sphere(x[8:10]) +
            lambda_vals[9] * sphere(x[8:10]))

def F30(x):
    # Check that the input x is within the specified bounds
    if np.any(x < -5) or np.any(x > 5):
        raise ValueError("Input values must be within the range [-5, 5].")

    # Sigma values
    sigma = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

    # Lambda values calculated based on the given specifications
    lambda_vals = np.array([
        0.1 * 1 / 5,
        0.2 * 1 / 5,
        0.3 * 5 / 0.5,
        0.4 * 5 / 0.5,
        0.5 * 5 / 100,
        0.6 * 5 / 100,
        0.7 * 5 / 32,
        0.8 * 5 / 32,
        0.9 * 5 / 100,
        1 * 5 / 100
    ])

    # Calculate the objective value using the respective functions
    return (lambda_vals[0] * rastrigin(x[:2]) +
            lambda_vals[1] * rastrigin(x[:2]) +
            lambda_vals[2] * weierstrass(x[2:4]) +
            lambda_vals[3] * weierstrass(x[2:4]) +
            lambda_vals[4] * griewank(x[4:6]) +
            lambda_vals[5] * griewank(x[4:6]) +
            lambda_vals[6] * ackley(x[6:8]) +
            lambda_vals[7] * ackley(x[6:8]) +
            lambda_vals[8] * sphere(x[8:10]) +
            lambda_vals[9] * sphere(x[8:10]))


def get_function(key):
    function_mapping = {
        'Sphere': 'F1',
        "Schwefel's 2.22": 'F2',
        'Powell Sum': 'F3',
        "Schwefel's 1.2": 'F4',
        "Schwefel's 2.21": 'F5',
        'Rosenbrock': 'F6',
        'Step': 'F7',
        'Quartic': 'F8',
        'Zakharov': 'F9',
        'Schwefel': 'F10',
        'Periodic': 'F11',
        'Styblinski-Tang': 'F12',
        'Rastrigin': 'F13',
        'Ackley 1': 'F14',
        'Griewank': 'F15',
        'Xin-She Yang N.4': 'F16',
        'Penalized': 'F17',
        'Penalized2': 'F18',
        'Foxholes': 'F19',
        'Kowalik': 'F20',
        'Six Hump Camel': 'F21',
        'Shekel 5': 'F22',
        'Shekel 7': 'F23',
        'Shekel 10': 'F24',
        'composite_function_1': 'F25',
        'composite_function_2': 'F26',
        'composite_function_3': 'F27',
        'composite_function_4': 'F28',
        'composite_function_5': 'F29',
        'composite_function_6': 'F30'
    }
    return function_mapping[key]
def get_function_details(F):
    if F == 'F1':
        return -100, 100, 10, F1
    elif F == 'F2':
        return -10, 10, 10, F2
    elif F == 'F3':
        return -1, 1, 10, F3
    elif F == 'F4':
        return -100, 100, 10, F4
    elif F == 'F5':
        return -100, 100, 10, F5
    elif F == 'F6':
        return -30, 30, 10, F6
    elif F == 'F7':
        return -100, 100, 10, F7
    elif F == 'F8':
        return -1.28, 1.28, 10, F8
    elif F == 'F9':
        return -5, 10, 10, F9
    elif F == 'F10':
        return -500, 500, 30, F10
    elif F == 'F11':
        return -10, 10, 30, F11
    elif F == 'F12':
        return -5, 5, 30, F12
    elif F == 'F13':
        return -5.12, 5.12, 30, F13
    elif F == 'F14':
        return -32, 32, 30, F14
    elif F == 'F15':
        return -600, 600, 30, F15
    elif F == 'F16':
        return -10, 10, 30, F16
    elif F == 'F17':
        return -50, 50, 30, F17
    elif F == 'F18':
        return -50, 50, 30, F18
    elif F == 'F19':
        return -65, 65, 2, F19
    elif F == 'F20':
        return -5, 5, 4, F20
    elif F == 'F21':
        return -5, 5, 2, F21
    elif F == 'F22':
        return 0, 10, 4, F22
    elif F == 'F23':
        return 0, 10, 4, F23
    elif F == 'F24':
        return 0, 10, 4, F24
    elif F == 'F25':
        return -5, 5, 30, F25
    elif F == 'F26':
        return -5, 5, 30, F26
    elif F == 'F27':
        return -5, 5, 30, F27
    elif F == 'F28':
        return -5, 5, 30, F28
    elif F == 'F29':
        return -5, 5, 30, F29
    elif F == 'F30':
        return -5, 5, 30, F30
    else:
        raise ValueError("Invalid function name.")

# def plot_function(func, bounds, resolution=100):
#     x = np.linspace(bounds[0], bounds[1], resolution)
#     if func.__code__.co_argcount == 1:  # Check if the function takes one argument
#         y = func(x)
#     else:
#         raise ValueError("Function must take a single input array.")
#
#     plt.figure(figsize=(10, 5))
#     plt.plot(x, y, label=func.__name__)
#     plt.title(f"Plot of {func.__name__}")
#     plt.xlabel('x')
#     plt.ylabel('f(x)')
#     plt.grid()
#     plt.legend()
#     plt.show()
#
#
# def plot_all_functions():
#     function_names = [
#         "Sphere", "Schwefel's 2.22", 'Powell Sum', "Schwefel's 1.2",
#         "Schwefel's 2.21", 'Rosenbrock', 'Step', 'Quartic', 'Zakharov',
#         'Schwefel', 'Periodic', 'Styblinski-Tang', 'Rastrigin', 'Ackley 1',
#         'Griewank', 'Xin-She Yang N.4', 'Penalized', 'Penalized2',
#         'Foxholes', 'Kowalik', 'Six Hump Camel', 'Shekel 5', 'Shekel 7',
#         'Shekel 10'
#     ]
#
#     for name in function_names:
#         bounds = get_function_details(get_function(name))[0:2]  # Get the bounds for the function
#         func = get_function_details(get_function(name))[-1]  # Get the function to plot
#         plot_function(func, bounds)