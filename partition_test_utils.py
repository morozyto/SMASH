import numpy as np
from math import sqrt


def batman_points():
    def f1(x):
        return 2 * sqrt(-abs(abs(x) - 1) * abs(3 - abs(x)) / ((abs(x) - 1) * (3 - abs(x)))) * \
        (1 + abs(abs(x) - 3) / (abs(x) - 3)) * sqrt(1 - (x / 7) ** 2) + (5 + 0.97 * \
            (abs(x - .5) + abs(x + .5)) - 3 * (abs(x - .75) + abs(x + .75))) * \
               (1 + abs(1. - abs(x)) / (1. - abs(x)))

    def f2(x):
        return -3 * sqrt(1 - (x / 7) ** 2) * sqrt(abs(abs(x) - 4) / (abs(x) - 4))

    def f3(x):
        return abs(x / 2) - 0.0913722 * (x ** 2) - 3 + sqrt(1 - (abs(abs(x) - 2) - 1) ** 2)

    def f4(x):
        return (2.71052 + (1.5 - .5 * abs(x)) - 1.35526 * sqrt(4 - (abs(x) - 1) ** 2)) * \
            sqrt(abs(abs(x) - 1) / (abs(x) - 1)) + 0.9

    step = 0.02
    x1 = list(np.arange(-7, -3 + step, step)) + list(np.arange(-1 + step, 1, step)) + list(np.arange(3 + step, 7, step))
    x2 = list(np.arange(-7, -4 + step, step)) + list(np.arange(4, 7 + step, step))
    x3 = list(np.arange(-4, 4, step))
    x4 = list(np.arange(-3, -1, step)) + list(np.arange(1, 3, step))

    y1 = [f1(x) for x in x1]
    y2 = [f2(x) for x in x2]
    y3 = [f3(x) for x in x3]
    y4 = [f4(x) for x in x4]

    points = list(zip(x1, y1)) + list(zip(x2, y2)) + list(zip(x3, y3)) + list(zip(x4, y4))
    points = [point for point in points if not np.isnan(point[1])]
    return points
