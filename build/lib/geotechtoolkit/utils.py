import numpy as np
import math


def get_length_and_width_rectangle(points):
    """
    Get the length and width of a rectangle defined by four points

    """

    if len(points) != 4:
        return False
    dist = [
        math.hypot(
            abs(points[i][0] - points[i + 1][0]), abs(points[i][1] - points[i + 1][1])
        )
        for i in [0, 1, 2]
    ]
    return min(dist), max(dist)


def contour_mesh(mesh, step=5):
    z = mesh.points[:, 2]
    print(z)
    mesh["Elevation"] = z
    mi, ma = round(min(z), ndigits=-2), round(max(z), ndigits=-2)
    step = 10
    cntrs = np.arange(mi, ma + step, step)
    return mesh.contour(cntrs, scalars="Elevation")


def get_point_on_vector(initial_pt, terminal_pt, distance):
    v = np.array(initial_pt, dtype=float)
    u = np.array(terminal_pt, dtype=float)
    n = v - u
    n /= np.linalg.norm(n, 2)
    point = v - distance * n

    return tuple(point)


def get_length(initial_pt, terminal_pt):
    v = np.array(initial_pt, dtype=float)
    u = np.array(terminal_pt, dtype=float)
    n = v - u
    n = n**2
    return sum(n) ** 0.5
