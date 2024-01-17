import json

import numpy as np

from src.kinematics.point import Point

mockingData = np.array([[1, 2], [4, 5]])

# Two ways to initialize a Point object, either with data or without data but must provide num_frames
pointA = Point(data=mockingData, point_name="Left Shoulder")

pointA.preprocess(
    preprocess_interpolate_on=True,
    preprocess_filter_on=True,
    preprocess_smoothing_on=True,
)  #

pointA.compute(
    postcalculate_filter_on=False, postcalculate_smoothing_on=False
)  # displacements, speed, acceleration

# Print and Export
metrics = pointA.get_metrics()
metrics_json = json.dumps(metrics, indent=2)

# Export chart
pointA.export_metrics("./test/point")
