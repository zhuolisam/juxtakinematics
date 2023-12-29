from typing import Literal

import numpy as np


def calculate_angle(a, o, b):
    # Calculate the angles using numpy
    angle1 = np.arctan2(b.y - o.y, b.x - o.x)
    angle2 = np.arctan2(a.y - o.y, a.x - o.x)

    # Compute the angular difference in degrees
    ang = np.degrees(angle1 - angle2)

    # Adjust the result to be within [0, 360)
    if (ang < 0).any():
        ang[ang < 0] += 360
    return ang

# def calculate_angle(point1, point2, point3):
#     if len(point1.x) != len(point2.x) or len(point2.x) != len(point3.x):
#         raise ValueError("Input points must have the same length")

#     angle = np.arctan2(point3.y - point1.y, point3.x - point1.x) - np.arctan2(point2.y - point1.y, point2.x - point1.x)

#     return angle

# def calculate_angle(point1, point2, point3):
#     result = atan2(P3.y - P1.y, P3.x - P1.x) -
#                 atan2(P2.y - P1.y, P2.x - P1.x);
#     return result


def buildComponent(num_frames, enum):
    # return {each: np.full((num_frames), np.nan) for each in enum}
    return {
            "Raw X": np.full(num_frames, np.nan),
            "Raw Y": np.full(num_frames, np.nan),
            "X": np.full(num_frames, np.nan),
            "Y": np.full(num_frames, np.nan),
            "X Displacement": np.full(num_frames, np.nan),
            "Y Displacement": np.full(num_frames, np.nan),
            "Displacement": np.full(num_frames, np.nan),
            "X Speed No Smooth": np.full(num_frames, np.nan),
            "Y Speed No Smooth": np.full(num_frames, np.nan),
            "Speed No Smooth": np.full(num_frames, np.nan),
            "X Speed": np.full(num_frames, np.nan),
            "Y Speed": np.full(num_frames, np.nan),
            "Speed": np.full(num_frames, np.nan),
            "X Acceleration No Smooth": np.full(num_frames, np.nan),
            "Y Acceleration No Smooth": np.full(num_frames, np.nan),
            "Acceleration No Smooth": np.full(num_frames, np.nan),
            "X Acceleration": np.full(num_frames, np.nan),
            "Y Acceleration": np.full(num_frames, np.nan),
            "Acceleration": np.full(num_frames, np.nan),
        }


if __name__ == '__main__':
    class Point:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    # Create Point objects with numpy arrays
    point1 = Point(np.array([1, 2, 3]), np.array([4, 5, 6]))
    point2 = Point(np.array([2, 3, 4]), np.array([5, 6, 7]))
    point3 = Point(np.array([3, 4, 5]), np.array([6, 7, 8]))

    # Calculate the angle between the points
    angle = calculate_angle(point1, point2, point3)
    print(angle)
