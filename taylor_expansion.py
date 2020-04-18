import math
import numpy as np
from tools import *

SEPARATION_RATIO = 0.7
r = 10


def n(radius, l):
    if l == 0:
        return 1
    return ((l / math.e) * ((2 * math.pi * r) ** (1 / (2 * r))) * (1 / radius)) ** l


def phi(radius, l, x):
    return n(radius, l) * ((x ** l) / math.factorial(l))

'''
def c(k, l, a, b, radius_a, radius_b):
    if l > k:
        return 0
    return -math.factorial(k)*((b - a) ** (-k - 1)) * (1 / (n(radius_a, l))) * (1 / n(radius_b, k - l)) * ((-1) ** (k - l))
'''

def form_U(X):
    #matrix = np.array([[0] * r] * len(X))
    _, radius_x = get_metadata(X)

    #U = np.array([[phi(radius_x, l, x[i] - x_center) for l in range(r)] for i in X])

    for i in range(len(X)):
        x = X[i]
        for l in range(r):
            matrix[i][l] = phi(radius_x, l, x - radius_x)
    return matrix


def form_V(Y):
    matrix = np.array([[0] * r] * len(Y))
    _, radius_y = get_metadata(Y)
    for i in range(len(Y)):
        y = Y[i]
        for l in range(r):
            matrix[i][l] = phi(radius_y, l, y - radius_y)
    return matrix

'''
def form_B(a, b, radius_a, radius_b):
    matrix = np.array([[0] * r] * r)
    for k in range(r):
        for l in range(r):
            matrix[k][l] = c(k, l, a, b, radius_a, radius_b)
    return matrix
'''


'''
def get_data_points(points):
    min_val, max_val = min(points), max(points)

    center = (max_val + min_val) / 2

    radius = max_val - center

    return center, radius


def build_well_separated(X_i, Y_j): #sets of indices
    r = 2

    x_center, x_radius = get_data_points(X_i)
    y_center, y_radius = get_data_points(Y_j)

    U = np.array([[phi(x_radius, l, x[i] - x_center) for l in range(r)] for i in X_i])
    B = np.array([[c(k, l, x_center, y_center, x_radius, y_radius) for l in range(r)] for k in range(r)])
    V = np.array([[phi(y_radius, l, y[j] - y_center) for l in range(r)] for j in Y_j])

    return U, B, V
'''