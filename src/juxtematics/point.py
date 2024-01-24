import json
import uuid
from pathlib import Path
from typing import Any, Literal

import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate

from .constants import BODY_JOINTS_MAP
from .filtering import FilteredTrajectory
from .smoothing import MovingAverage
from .utils import get_time

PointMetrics = [
    "X",
    "Y",
    "X Displacement",
    "Y Displacement",
    "Displacement",
    "X Speed",
    "Y Speed",
    "Speed",
    "X Acceleration",
    "Y Acceleration",
    "Acceleration",
]


# So at first I let people initEmptyFrames so can easily update by using index. This is to ease if we know the length of data.
# But then we may not know the length of data, so I considering remove this thing. Either you create empty data and append, or you initialize with data. You have to add the empty frames if you initialize with data.
# In the end, if you use initWithData, I assume we already have the length. If you use appendPoints, you must update num_frames.


class Point:
    @property
    def get_frames(self):
        return self.calibrationHelper["num_frames"]

    def __init__(
        self,
        data=None,
        point_idx=None,
        point_name=None,
        num_frames=0,
        fps=30,
        x_meter_per_pixel=None,
        y_meter_per_pixel=None,
    ) -> None:
        # raw_ori_x and raw_ori_y are the original data
        # raw_x and raw_y are the interpolated data
        # x and y are the filtered data
        self.data = {}
        self.calibrationHelper = {}

        # TODO: The point can be dependent on other points, user must provide the function to calculate "Raw Ori X" and "Raw Ori Y" from the dependent points
        self.dependencies = []

        self.calibrationHelper = {
            "unit": "pixel",  # can be "pixel" or "meter
            "x_meter_per_pixel": 1.0
            if x_meter_per_pixel is None
            else float(x_meter_per_pixel),
            "y_meter_per_pixel": 1.0
            if y_meter_per_pixel is None
            else float(y_meter_per_pixel),
            "fps": 30 if fps is None else float(fps),
        }

        if data is not None:
            self.initWithData(data=data, fps=fps)
        else:
            self.initWithEmptyFrames(num_frames=num_frames, fps=fps)

        self.point_idx = point_idx
        self.point_name = (
            point_name
            if point_name
            else BODY_JOINTS_MAP[point_idx]
            if point_idx is not None
            else "point_" + uuid.uuid4().hex
        )
        self.filter = FilteredTrajectory()

    # SETUP
    def initWithEmptyFrames(self, num_frames, fps):
        self.data["Raw Ori X"] = np.full(num_frames, np.nan)
        self.data["Raw Ori Y"] = np.full(num_frames, np.nan)
        self.calibrationHelper["num_frames"] = num_frames

    def initWithData(self, data, fps=None):
        assert (
            len(data[0]) == 2
        ), (
            f"Expected 2 coordinates, got {len(data[0])}"
        )  # Assert data has two coordinates
        self.data["Raw Ori X"] = np.array(data)[:, 0]
        self.data["Raw Ori Y"] = np.array(data)[:, 1]
        self.calibrationHelper["num_frames"] = len(data)

    def __getitem__(self, __name: str) -> Any:
        return self.__dict__[__name]

    def preprocess_dependent_points(self, dependent_points):
        # TODO: Plot Raw Ori X and Raw Ori Y from the dependent points
        pass

    def preprocess(
        self,
        filter_threshold=None,
        preprocess_interpolate_on=True,
        preprocess_filter_on=True,
        preprocess_smoothing_on=False,
    ):
        # Find the first non-null data points from the beginning and end
        start_index, end_index = self.find_non_null_indices(self.data["Raw Ori X"])

        self.start_index = start_index if start_index is not None else 0
        self.end_index = (
            end_index
            if end_index is not None
            else self.calibrationHelper["num_frames"] - 1
        )

        # update X and Y
        self.data["X"] = self.data["Raw Ori X"]
        self.data["Y"] = self.data["Raw Ori Y"]

        if preprocess_interpolate_on:
            # Interpolate the data within the range defined by the first and last non-null data points
            self.data["Raw Ori X Interpolate"] = self.data["X"].copy()
            self.data["Raw Ori Y Interpolate"] = self.data["Y"].copy()
            self.data["Raw Ori X Interpolate"][
                self.start_index : self.end_index + 1
            ] = self.interpolate_missing(
                self.data["Raw Ori X Interpolate"][
                    self.start_index : self.end_index + 1
                ]
            )
            self.data["Raw Ori Y Interpolate"][
                self.start_index : self.end_index + 1
            ] = self.interpolate_missing(
                self.data["Raw Ori Y Interpolate"][
                    self.start_index : self.end_index + 1
                ]
            )

            # Update X and Y
            self.data["X"] = self.data["Raw Ori X Interpolate"]
            self.data["Y"] = self.data["Raw Ori Y Interpolate"]

        # Proceed with filtering if the number of frames is greater than 10
        can_filter = preprocess_filter_on and self.calibrationHelper["num_frames"] > 10

        if can_filter:
            self.data["Raw Ori X Filter"] = self.data["X"].copy()
            self.data["Raw Ori Y Filter"] = self.data["Y"].copy()
            # Apply filtering
            (
                self.data["Raw Ori X Filter"][self.start_index : self.end_index + 1],
                self.data["Raw Ori Y Filter"][self.start_index : self.end_index + 1],
            ) = self.filter.initialize(
                self.data["Raw Ori X Filter"][self.start_index : self.end_index + 1],
                self.data["Raw Ori Y Filter"][self.start_index : self.end_index + 1],
            )

            # Update X and Y
            self.data["X"] = self.data["Raw Ori X Filter"]
            self.data["Y"] = self.data["Raw Ori Y Filter"]

        if preprocess_smoothing_on:
            self.data["Raw Ori X Smooth"] = self.data["X"].copy()
            self.data["Raw Ori Y Smooth"] = self.data["Y"].copy()
            smoothing = MovingAverage()
            self.data["Raw Ori X Smooth"][
                self.start_index : self.end_index + 1
            ] = smoothing.filter_samples(
                samples=self.data["Raw Ori X Smooth"][
                    self.start_index : self.end_index + 1
                ],
                fs=self.calibrationHelper["fps"],
                span=40,
                sentinels=1,
            )
            self.data["Raw Ori Y Smooth"][
                self.start_index : self.end_index + 1
            ] = smoothing.filter_samples(
                samples=self.data["Raw Ori Y Smooth"][
                    self.start_index : self.end_index + 1
                ],
                fs=self.calibrationHelper["fps"],
                span=40,
                sentinels=1,
            )
            # Update X and Y
            self.data["X"] = self.data["Raw Ori X Smooth"]
            self.data["Y"] = self.data["Raw Ori Y Smooth"]

    def find_non_null_indices(self, data):
        # Find the first non-null data point from the beginning
        start_index = np.where(~np.isnan(data))[0]
        start_index = start_index[0] if start_index.size > 0 else None

        # Find the first non-null data point from the end
        end_index = np.where(~np.isnan(data))[0]
        end_index = end_index[-1] if end_index.size > 0 else None
        return int(start_index), int(end_index)

    def interpolate_missing(self, data):
        # Create an array of indices where values are not missing
        valid_indices = np.where(~np.isnan(data))[0]

        # Perform linear interpolation
        f = interpolate.interp1d(
            valid_indices, data[valid_indices], kind="linear", fill_value="extrapolate"
        )
        interpolated_data = f(np.arange(len(data)))

        return interpolated_data

    def compute(self, postcalculate_filter_on=False, postcalculate_smoothing_on=False):
        self.computeDisplacements()
        self.computeSpeeds(
            filter_on=postcalculate_filter_on, smoothing_on=postcalculate_smoothing_on
        )
        self.computeAccelerations(
            filter_on=postcalculate_filter_on, smoothing_on=postcalculate_smoothing_on
        )

        # Remove keys with all NaN values and replace with empty list
        for key, value in self.data.items():
            if np.isnan(value).all():
                self.data[key] = np.array([])

    def computeDisplacements(self):
        # Calculate displacement relative to the first frame
        x_displacement = (
            self.data["X"][self.start_index : self.end_index + 1]
            - self.data["X"][self.start_index]
        ) * self.calibrationHelper["x_meter_per_pixel"]

        y_displacement = (
            self.data["Y"][self.start_index : self.end_index + 1]
            - self.data["Y"][self.start_index]
        ) * self.calibrationHelper["y_meter_per_pixel"]

        self.data["X Displacement"] = self.data["X"].copy()
        self.data["Y Displacement"] = self.data["Y"].copy()
        self.data["Displacement"] = self.data["X"].copy()

        self.data["X Displacement"][
            self.start_index : self.end_index + 1
        ] = x_displacement
        self.data["Y Displacement"][
            self.start_index : self.end_index + 1
        ] = y_displacement
        # Calculate Displacement relative to the first frame
        self.data["Displacement"][self.start_index : self.end_index + 1] = np.sqrt(
            x_displacement**2 + y_displacement**2
        )

    def computeSpeeds(self, filter_on=False, smoothing_on=False):
        # X Speed Raw -> X Speed Filter -> X Speed Smooth -> X Speed
        time_diff = 1 / self.calibrationHelper["fps"]
        x_speed = (
            np.diff(
                self.data["X Displacement"][self.start_index : self.end_index + 1],
                prepend=self.data["X Displacement"][self.start_index],
            )
            / time_diff
        )
        y_speed = (
            np.diff(
                self.data["Y Displacement"][self.start_index : self.end_index + 1],
                prepend=self.data["Y Displacement"][self.start_index],
            )
            / time_diff
        )
        self.data["X Speed"] = self.data["X Displacement"].copy()
        self.data["Y Speed"] = self.data["Y Displacement"].copy()
        self.data["Speed"] = self.data["Displacement"].copy()

        self.data["X Speed"][self.start_index : self.end_index + 1] = x_speed
        self.data["Y Speed"][self.start_index : self.end_index + 1] = y_speed
        self.data["Speed"][self.start_index : self.end_index + 1] = np.sqrt(
            x_speed**2 + y_speed**2
        )
        self.data["X Speed Raw"] = self.data["X Speed"]
        self.data["Y Speed Raw"] = self.data["Y Speed"]
        self.data["Speed Raw"] = self.data["Speed"]

        if filter_on:
            self.data["X Speed Filter"] = self.data["X Speed"]
            self.data["Y Speed Filter"] = self.data["Y Speed"]
            # Apply filtering
            (
                self.data["X Speed Filter"][self.start_index : self.end_index + 1],
                self.data["Y Speed Filter"][self.start_index : self.end_index + 1],
            ) = self.filter.initialize(
                self.data["X Speed Filter"][self.start_index : self.end_index + 1],
                self.data["Y Speed Filter"][self.start_index : self.end_index + 1],
            )
            self.data["Speed Filter"] = np.sqrt(
                self.data["X Speed Filter"] ** 2 + self.data["Y Speed Filter"] ** 2
            )
            # Replace the speed with the filtered speed
            self.data["X Speed"] = self.data["X Speed Filter"]
            self.data["Y Speed"] = self.data["Y Speed Filter"]
            self.data["Speed"] = self.data["Speed Filter"]

        if smoothing_on:
            self.data["X Speed Smooth"] = self.data["X Speed"]
            self.data["Y Speed Smooth"] = self.data["Y Speed"]
            # Apply smoothing
            self.data["X Speed Smooth"][
                self.start_index : self.end_index + 1
            ] = MovingAverage.filter_samples(
                self.data["X Speed Smooth"][self.start_index : self.end_index + 1],
                self.calibrationHelper["fps"],
                40,
                1,
            )
            self.data["Y Speed Smooth"][
                self.start_index : self.end_index + 1
            ] = MovingAverage.filter_samples(
                self.data["Y Speed Smooth"][self.start_index : self.end_index + 1],
                self.calibrationHelper["fps"],
                40,
                1,
            )
            self.data["Speed Smooth"] = np.sqrt(
                self.data["X Speed Smooth"] ** 2 + self.data["Y Speed Smooth"] ** 2
            )

            # Replace the speed with the filtered speed
            self.data["X Speed"] = self.data["X Speed Smooth"]
            self.data["Y Speed"] = self.data["Y Speed Smooth"]
            self.data["Speed"] = self.data["Speed Smooth"]

    def computeAccelerations(self, filter_on=False, smoothing_on=True):
        # X Acceleration Raw -> X Acceleration Filter -> X Acceleration Smooth -> X Acceleration
        time_diff = 1 / self.calibrationHelper["fps"]
        x_accel = (
            np.diff(
                self.data["X Speed"][self.start_index : self.end_index + 1],
                prepend=self.data["X Speed"][self.start_index],
            )
            / time_diff
        )
        y_accel = (
            np.diff(
                self.data["Y Speed"][self.start_index : self.end_index + 1],
                prepend=self.data["Y Speed"][self.start_index],
            )
            / time_diff
        )

        self.data["X Acceleration"] = self.data["X Speed"].copy()
        self.data["Y Acceleration"] = self.data["Y Speed"].copy()
        self.data["Acceleration"] = self.data["Speed"].copy()

        self.data["X Acceleration"][self.start_index : self.end_index + 1] = x_accel
        self.data["Y Acceleration"][self.start_index : self.end_index + 1] = y_accel
        self.data["Acceleration"][self.start_index : self.end_index + 1] = np.sqrt(
            x_accel**2 + y_accel**2
        )
        self.data["X Acceleration Raw"] = self.data["X Acceleration"]
        self.data["Y Acceleration Raw"] = self.data["Y Acceleration"]
        self.data["Acceleration Raw"] = self.data["Acceleration"]

        if filter_on:
            self.data["X Acceleration Filter"] = self.data["X Acceleration"]
            self.data["Y Acceleration Filter"] = self.data["Y Acceleration"]

            # Apply filtering
            (
                self.data["X Acceleration Filter"][
                    self.start_index : self.end_index + 1
                ],
                self.data["Y Acceleration Filter"][
                    self.start_index : self.end_index + 1
                ],
            ) = self.filter.initialize(
                self.data["X Acceleration Filter"][
                    self.start_index : self.end_index + 1
                ],
                self.data["Y Acceleration Filter"][
                    self.start_index : self.end_index + 1
                ],
            )
            self.data["Acceleration Filter"] = np.sqrt(
                self.data["X Acceleration Filter"] ** 2
                + self.data["Y Acceleration Filter"] ** 2
            )

            # Replace the acceleration with the filtered acceleration
            self.data["X Acceleration"] = self.data["X Acceleration Filter"]
            self.data["Y Acceleration"] = self.data["Y Acceleration Filter"]
            self.data["Acceleration"] = self.data["Acceleration Filter"]

        if smoothing_on:
            self.data["X Acceleration Smooth"] = self.data["X Acceleration"]
            self.data["Y Acceleration Smooth"] = self.data["Y Acceleration"]

            # Apply smoothing
            self.data["X Acceleration"][
                self.start_index : self.end_index + 1
            ] = MovingAverage.filter_samples(
                self.data["X Acceleration"][self.start_index : self.end_index + 1],
                self.calibrationHelper["fps"],
                50,
                1,
            )
            self.data["Y Acceleration"][
                self.start_index : self.end_index + 1
            ] = MovingAverage.filter_samples(
                self.data["Y Acceleration"][self.start_index : self.end_index + 1],
                self.calibrationHelper["fps"],
                50,
                1,
            )
            self.data["Acceleration Smooth"] = np.sqrt(
                self.data["X Acceleration Smooth"] ** 2
                + self.data["Y Acceleration Smooth"] ** 2
            )

            # Replace the acceleration with the filtered acceleration
            self.data["X Acceleration"] = self.data["X Acceleration Smooth"]
            self.data["Y Acceleration"] = self.data["Y Acceleration Smooth"]
            self.data["Acceleration"] = self.data["Acceleration Smooth"]

    # SETTERS
    def append_empty_frames(self, frames=1):
        self.data["Raw Ori X"] = np.append(
            self.data["Raw Ori X"], np.full(frames, np.nan)
        )
        self.data["Raw Ori Y"] = np.append(
            self.data["Raw Ori Y"], np.full(frames, np.nan)
        )
        # update num_frames
        self.calibrationHelper["num_frames"] = len(self.data["Raw Ori X"])

    def append_points(self, data):
        assert len(data) == 2, f"Expected 2 coordinates, got {len(data)}"
        data = np.array(data)
        self.data["Raw Ori X"] = np.append(self.data["Raw Ori X"], data[0])
        self.data["Raw Ori Y"] = np.append(self.data["Raw Ori Y"], data[1])
        # update num_frames
        self.calibrationHelper["num_frames"] = len(self.data["Raw Ori X"])

    def set_frame_point(self, index, x, y):
        self.data["Raw Ori X"][index] = x
        self.data["Raw Ori Y"][index] = y
        return

    # GETTERS
    def get_metrics(self):
        data = {
            "metadata": self.get_metadata(),
            "metrics": self.get_metrics_data(),
            # { key: value.tolist() for key, value in self.data.items()},
            "metrics_info": self.get_metrics_info(),
            "time": [
                i / self.calibrationHelper["fps"]
                for i in range(self.calibrationHelper["num_frames"])
            ],
        }
        if None in data["metrics"]:
            del data["metrics"][None]
        if None in data["metrics_info"]:
            del data["metrics_info"][None]
        return data

    def get_frame_point(self, index):
        return self.data["Raw Ori X"][index], self.data["Raw Ori Y"][index]

    def get_metadata(self):
        return {
            "x_meter_per_pixel": self.calibrationHelper["x_meter_per_pixel"],
            "y_meter_per_pixel": self.calibrationHelper["y_meter_per_pixel"],
            "fps": self.calibrationHelper["fps"],
            "point_idx": self.point_idx,
            "num_frames": self.calibrationHelper["num_frames"],
            "name": self.point_name
            if self.point_name
            else BODY_JOINTS_MAP[self.point_idx]
            if self.point_idx is not None
            else None,
            "start_index": self.start_index,
            "end_index": self.end_index,
        }

    def get_metrics_data(self):
        data = {
            # coordinate
            "Raw Ori X": self.data["Raw Ori X"].tolist(),
            "Raw Ori Y": self.data["Raw Ori Y"].tolist(),
            "Raw Ori X Interpolate"
            if "Raw Ori X Interpolate" in self.data
            else None: self.data["Raw Ori X Interpolate"].tolist()
            if "Raw Ori X Interpolate" in self.data
            else None,
            "Raw Ori Y Interpolate"
            if "Raw Ori Y Interpolate" in self.data
            else None: self.data["Raw Ori Y Interpolate"].tolist()
            if "Raw Ori Y Interpolate" in self.data
            else None,
            "Raw Ori X Filter" if "Raw Ori X Filter" in self.data else None: self.data[
                "Raw Ori X Filter"
            ].tolist()
            if "Raw Ori X Filter" in self.data
            else None,
            "Raw Ori Y Filter" if "Raw Ori Y Filter" in self.data else None: self.data[
                "Raw Ori Y Filter"
            ].tolist()
            if "Raw Ori Y Filter" in self.data
            else None,
            "Raw Ori X Smooth" if "Raw Ori X Smooth" in self.data else None: self.data[
                "Raw Ori X Smooth"
            ].tolist()
            if "Raw Ori X Smooth" in self.data
            else None,
            "Raw Ori Y Smooth" if "Raw Ori Y Smooth" in self.data else None: self.data[
                "Raw Ori Y Smooth"
            ].tolist()
            if "Raw Ori Y Smooth" in self.data
            else None,
            "X": self.data["X"].tolist(),
            "Y": self.data["Y"].tolist(),
            # displacement
            "X Displacement": self.data["X Displacement"].tolist(),
            "Y Displacement": self.data["Y Displacement"].tolist(),
            "Displacement": self.data["Displacement"].tolist(),
            # speed
            "X Speed Raw": self.data["X Speed Raw"].tolist(),
            "Y Speed Raw": self.data["Y Speed Raw"].tolist(),
            "Speed Raw": self.data["Speed Raw"].tolist(),
            "X Speed Filter" if "X Speed Filter" in self.data else None: self.data[
                "X Speed Filter"
            ].tolist()
            if "X Speed Filter" in self.data
            else None,
            "Y Speed Filter" if "Y Speed Filter" in self.data else None: self.data[
                "Y Speed Filter"
            ].tolist()
            if "Y Speed Filter" in self.data
            else None,
            "Speed Filter" if "Speed Filter" in self.data else None: self.data[
                "Speed Filter"
            ].tolist()
            if "Speed Filter" in self.data
            else None,
            "Y Speed Smooth" if "Y Speed Smooth" in self.data else None: self.data[
                "Y Speed Smooth"
            ].tolist()
            if "Y Speed Smooth" in self.data
            else None,
            "X Speed Smooth" if "X Speed Smooth" in self.data else None: self.data[
                "X Speed Smooth"
            ].tolist()
            if "X Speed Smooth" in self.data
            else None,
            "Speed Smooth" if "Speed Smooth" in self.data else None: self.data[
                "Speed Smooth"
            ].tolist()
            if "Speed Smooth" in self.data
            else None,
            "X Speed": self.data["X Speed"].tolist(),
            "Y Speed": self.data["Y Speed"].tolist(),
            "Speed": self.data["Speed"].tolist(),
            # acceleration
            "X Acceleration Raw": self.data["X Acceleration Raw"].tolist(),
            "Y Acceleration Raw": self.data["Y Acceleration Raw"].tolist(),
            "Acceleration Raw": self.data["Acceleration Raw"].tolist(),
            "X Acceleration Filter"
            if "X Acceleration Filter" in self.data
            else None: self.data["X Acceleration Filter"].tolist()
            if "X Acceleration Filter" in self.data
            else None,
            "Y Acceleration Filter"
            if "Y Acceleration Filter" in self.data
            else None: self.data["Y Acceleration Filter"].tolist()
            if "Y Acceleration Filter" in self.data
            else None,
            "Acceleration Filter"
            if "Acceleration Filter" in self.data
            else None: self.data["Acceleration Filter"].tolist()
            if "Acceleration Filter" in self.data
            else None,
            "X Acceleration Smooth"
            if "X Acceleration Smooth" in self.data
            else None: self.data["X Acceleration Smooth"].tolist()
            if "X Acceleration Smooth" in self.data
            else None,
            "Y Acceleration Smooth"
            if "Y Acceleration Smooth" in self.data
            else None: self.data["Y Acceleration Smooth"].tolist()
            if "Y Acceleration Smooth" in self.data
            else None,
            "Acceleration Smooth"
            if "Acceleration Smooth" in self.data
            else None: self.data["Acceleration Smooth"].tolist()
            if "Acceleration Smooth" in self.data
            else None,
            "X Acceleration": self.data["X Acceleration"].tolist(),
            "Y Acceleration": self.data["Y Acceleration"].tolist(),
            "Acceleration": self.data["Acceleration"].tolist(),
        }
        if None in data:
            del data[None]
        return data

    def get_metrics_info(self):
        data = {
            # coordinate
            "Raw Ori X": {"Unit": "Pixel"},
            "Raw Ori Y": {"Unit": "Pixel"},
            "Raw Ori X Interpolate" if "Raw Ori X Interpolate" in self.data else None: {
                "Unit": "Pixel"
            },
            "Raw Ori Y Interpolate" if "Raw Ori Y Interpolate" in self.data else None: {
                "Unit": "Pixel"
            },
            "Raw Ori X Filter" if "Raw Ori X Filter" in self.data else None: {
                "Unit": "Pixel"
            },
            "Raw Ori Y Filter" if "Raw Ori Y Filter" in self.data else None: {
                "Unit": "Pixel"
            },
            "Raw Ori X Smooth" if "Raw Ori X Smooth" in self.data else None: {
                "Unit": "Pixel"
            },
            "Raw Ori Y Smooth" if "Raw Ori Y Smooth" in self.data else None: {
                "Unit": "Pixel"
            },
            "X": {"Unit": "Pixel"},
            "Y": {"Unit": "Pixel"},
            # displacement
            "X Displacement": {
                "Unit": "Meter"
                if self.calibrationHelper["x_meter_per_pixel"] is not None
                else "Pixel"
            },
            "Y Displacement": {
                "Unit": "Meter"
                if self.calibrationHelper["y_meter_per_pixel"] is not None
                else "Pixel"
            },
            "Displacement": {"Unit": "Meter"}
            if self.calibrationHelper["x_meter_per_pixel"] is not None
            and self.calibrationHelper["y_meter_per_pixel"] is not None
            else {"Unit": "Pixel"},
            # speed
            "X Speed": {"Unit": "Pixel/s"}
            if self.calibrationHelper["x_meter_per_pixel"] is None
            else {"Unit": "Meter/s"},
            "Y Speed": {
                "Unit": "Pixel/s"
                if self.calibrationHelper["y_meter_per_pixel"] is None
                else "Meter/s"
            },
            "Speed": {
                "Unit": "Pixel/s"
                if self.calibrationHelper["x_meter_per_pixel"] is None
                and self.calibrationHelper["y_meter_per_pixel"] is not None
                else "Meter/s"
            },
            # add in speed filter and speed smooth
            "X Speed Filter" if "X Speed Filter" in self.data else None: {
                "Unit": "Pixel/s"
            }
            if self.calibrationHelper["x_meter_per_pixel"] is None
            else {"Unit": "Meter/s"},
            "Y Speed Filter" if "Y Speed Filter" in self.data else None: {
                "Unit": "Pixel/s"
                if self.calibrationHelper["y_meter_per_pixel"] is None
                else "Meter/s"
            },
            "Speed Filter" if "Speed Filter" in self.data else None: {
                "Unit": "Pixel/s"
                if self.calibrationHelper["x_meter_per_pixel"] is None
                and self.calibrationHelper["y_meter_per_pixel"] is None
                else "Meter/s"
            },
            "X Speed Smooth" if "X Speed Smooth" in self.data else None: {
                "Unit": "Pixel/s"
            }
            if self.calibrationHelper["x_meter_per_pixel"] is None
            else {"Unit": "Meter/s"},
            "Y Speed Smooth" if "Y Speed Smooth" in self.data else None: {
                "Unit": "Pixel/s"
                if self.calibrationHelper["y_meter_per_pixel"] is None
                else "Meter/s"
            },
            "Speed Smooth" if "Speed Smooth" in self.data else None: {
                "Unit": "Pixel/s"
                if self.calibrationHelper["x_meter_per_pixel"] is None
                and self.calibrationHelper["y_meter_per_pixel"] is None
                else "Meter/s"
            },
            # acceleration
            "X Acceleration": {
                "Unit": "Pixel/s^2"
                if self.calibrationHelper["x_meter_per_pixel"] is None
                else "Meter/s^2"
            },
            "Y Acceleration": {
                "Unit": "Pixel/s^2"
                if self.calibrationHelper["y_meter_per_pixel"] is None
                else "Meter/s^2"
            },
            "Acceleration No Smooth"
            if "Acceleration No Smooth" in self.data
            else None: {
                "Unit": "Pixel/s^2"
                if self.calibrationHelper["x_meter_per_pixel"] is None
                and self.calibrationHelper["y_meter_per_pixel"] is None
                else "Meter/s^2"
            },
            "Acceleration": {
                "Unit": "Pixel/s^2"
                if self.calibrationHelper["x_meter_per_pixel"] is None
                and self.calibrationHelper["y_meter_per_pixel"] is None
                else "Meter/s^2"
            },
            # add in acceleration filter and acceleration smooth
            "X Acceleration Filter" if "X Acceleration Filter" in self.data else None: {
                "Unit": "Pixel/s^2"
                if self.calibrationHelper["x_meter_per_pixel"] is None
                else "Meter/s^2"
            },
            "Y Acceleration Filter" if "Y Acceleration Filter" in self.data else None: {
                "Unit": "Pixel/s^2"
                if self.calibrationHelper["y_meter_per_pixel"] is None
                else "Meter/s^2"
            },
            "Acceleration Filter" if "Acceleration Filter" in self.data else None: {
                "Unit": "Pixel/s^2"
                if self.calibrationHelper["x_meter_per_pixel"] is None
                and self.calibrationHelper["y_meter_per_pixel"] is None
                else "Meter/s^2"
            },
        }
        if None in data:
            del data[None]

    # EXPORTS
    def export_metrics(self, dir):
        if self.point_idx is not None:
            name = f"{self.point_idx}_{BODY_JOINTS_MAP[self.point_idx]}"
        else:
            name = self.point_name
        # export json
        json_output = self.get_metrics()
        json_file = Path(dir) / get_time()
        json_file = json_file / f"{name}.json"
        json_file.parent.mkdir(parents=True, exist_ok=True)
        with open(json_file, "w") as f:
            json.dump(json_output, f, indent=2)

    def export_diagrams(
        self,
        human_dir,
        mode: Literal['compact", "complete'] = "compact",
        frequency=30,
        meter_per_pixel=None,
        metrics_to_export=PointMetrics,
    ):
        # THINGS TO CONSIDER TO EXPORT JSON DATA
        # 1. frequency

        # THINGS TO CONSIDER TO EXPORT CHARTS
        # 1. what is the name of the diagram
        # 2. what is the x-ais unit? time or frame
        # 3. what is the y-axis unit? pixel or meter
        # 4. what is the dir to export? what is the png name?

        data = {
            "X Displacement": self.data["X Displacement"].tolist()
            if meter_per_pixel is None
            else (self.data["X Displacement"] * meter_per_pixel).tolist(),
            "Y Displacement": self.data["Y Displacement"].tolist()
            if meter_per_pixel is None
            else (self.data["Y Displacement"] * meter_per_pixel).tolist(),
            "X Coordinate": self.data["X"].tolist()
            if meter_per_pixel is None
            else (self.data["X"] * meter_per_pixel).tolist(),
            "Y Coordinate": self.data["Y"].tolist()
            if meter_per_pixel is None
            else (self.data["Y"] * meter_per_pixel).tolist(),
            "Displacement": self.data["Displacement"].tolist()
            if meter_per_pixel is None
            else (self.data["Displacement"] * meter_per_pixel).tolist(),
            "X Speed": self.data["X Speed"].tolist()
            if meter_per_pixel is None
            else (self.data["X Speed"] * meter_per_pixel).tolist(),
            "Y Speed": self.data["Y Speed"].tolist()
            if meter_per_pixel is None
            else (self.data["Y Speed"] * meter_per_pixel).tolist(),
            "Speed": self.data["Speed"].tolist()
            if meter_per_pixel is None
            else (self.data["Speed"] * meter_per_pixel).tolist(),
            "X Acceleration": self.data["X Acceleration"].tolist()
            if meter_per_pixel is None
            else (self.data["X Acceleration"] * meter_per_pixel).tolist(),
            "Y Acceleration": self.data["Y Acceleration"].tolist()
            if meter_per_pixel is None
            else (self.data["Y Acceleration"] * meter_per_pixel).tolist(),
            "Acceleration": self.data["Acceleration"].tolist()
            if meter_per_pixel is None
            else (self.data["Acceleration"] * meter_per_pixel).tolist(),
        }

        if self.point_idx is not None:
            name = f"{self.point_idx}_{BODY_JOINTS_MAP[self.point_idx]}"
        else:
            name = self.point_name
        if mode == "compact":
            num_metrics = len(metrics_to_export)
            num_cols = 3
            # unlimited rows
            num_rows = (
                num_metrics // num_cols + 1
                if num_metrics % num_cols != 0
                else num_metrics // num_cols
            )

            # Adjust the figsize as needed
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(8, 6))

            for i, metric in enumerate(metrics_to_export):
                row = i // num_cols
                col = i % num_cols
                ax = axes[row, col]
                ax.plot(range(self.calibrationHelper["num_frames"]), data[metric])
                ax.set_xlabel("Frame") if frequency is not None else ax.set_xlabel(
                    "Time (s)"
                )
                ax.set_ylabel(
                    "Meter"
                ) if meter_per_pixel is not None else ax.set_ylabel("Pixel")
                ax.set_title(f"{metric}")  # Add chart title

            chart_filename = Path(human_dir)
            chart_filename.mkdir(parents=True, exist_ok=True)

            chart_filename = chart_filename / f"{name}_compact.png"
            plt.tight_layout()
            plt.savefig(chart_filename)
            plt.close()

        elif mode == "complete":
            for each in metrics_to_export:
                plt.figure(figsize=(8, 6))
                plt.plot(range(self.calibrationHelper["num_frames"]), data[each])
                plt.xlabel("Frame") if frequency is not None else plt.xlabel("Time (s)")
                plt.ylabel("Meter") if meter_per_pixel is not None else plt.ylabel(
                    "Pixel"
                )
                plt.title(f"{each}")
                chart_filename = (
                    Path(human_dir)
                    / f"{self.point_idx}_{BODY_JOINTS_MAP[self.point_idx]}"
                )
                chart_filename.mkdir(parents=True, exist_ok=True)
                chart_filename = (
                    chart_filename / f"{each.lower().replace(' ', '_')}_chart.png"
                )
                plt.savefig(chart_filename)
                plt.close()

        # export json
        json_output = self.get_metrics()
        json_file = Path(human_dir) / f"{name}.json"
        with open(json_file, "w") as f:
            json.dump(json_output, f, indent=4)
