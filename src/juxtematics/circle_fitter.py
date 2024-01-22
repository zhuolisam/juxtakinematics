import numpy as np


class CircleFitter:
    @staticmethod
    def fit(traj):
        if traj.length < 3:
            return Circle.empty()

        # Least-squares circle fitting.
        rows = traj.length
        m = np.zeros((rows, 3))
        v = np.zeros((rows, 1))

        for i in range(rows):
            point = traj.coordinates(i)
            m[i, 0] = point[0]
            m[i, 1] = point[1]
            m[i, 2] = 1.0
            v[i, 0] = point[0] ** 2 + point[1] ** 2

        try:
            mt = m.transpose()
            b = np.dot(mt, m)
            c = np.dot(mt, v)
            z = np.linalg.solve(b, c)

            center = (z[0, 0] * 0.5, z[1, 0] * 0.5)
            radius = np.sqrt(z[2, 0] + (center[0] ** 2) + (center[1] ** 2))

            return Circle(center, radius)
        except np.linalg.LinAlgError:
            return Circle.empty()

class Circle:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    @staticmethod
    def empty():
        return Circle((0, 0), 0)
