from datetime import datetime

import numpy as np


def get_time():
    # current date and time
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def calculate_angle(a, o, b):
    # Reconstruct o,a,b from Point
    a = {
        "X": a.data["X"],
        "Y": a.data["Y"],
    }
    b = {
        "X": b.data["X"],
        "Y": b.data["Y"],
    }
    o = {
        "X": o.data["X"],
        "Y": o.data["Y"],
    }

    angle1 = np.arctan2(b["Y"] - o["Y"], b["X"] - o["X"])
    angle2 = np.arctan2(a["Y"] - o["Y"], a["X"] - o["X"])

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
        "X Speed": np.full(num_frames, np.nan),
        "Y Speed": np.full(num_frames, np.nan),
        "Speed": np.full(num_frames, np.nan),
        "X Acceleration": np.full(num_frames, np.nan),
        "Y Acceleration": np.full(num_frames, np.nan),
        "Acceleration": np.full(num_frames, np.nan),
    }
