import json
import uuid
from pathlib import Path
from typing import Any, Generator, List, Literal, Optional, Union

import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate

from .constants import BODY_JOINTS_MAP
from .filtering import FilteredTrajectory
from .smoothing import MovingAverage

PointMetrics = ["X Coordinate", "Y Coordinate", "X Displacement", "Y Displacement", "Distance",
                "X Speed", "Y Speed", "Speed", "X Acceleration", "Y Acceleration", "Acceleration"]

PointMetricsMapping = {
    "raw_ori_x": "Raw X Coordinate",
    "raw_ori_y": "Raw Y Coordinate",
    "raw_x": "Raw X Coordinate",
    "raw_y": "Raw Y Coordinate",
    "x": "X Coordinate",
    "y": "Y Coordinate",

    "x_displacement": "X Displacement",
    "y_displacement": "Y Displacement",
    "distance": "Distance",

    "x_speed_no_smooth": "X Speed",
    "y_speed_no_smooth": "Y Speed",
    "x_speed": "X Speed",
    "y_speed": "Y Speed",
    "speed_no_smooth": "Speed",
    "speed": "Speed",

    "x_acceleration_no_smooth": "X Acceleration",
    "y_acceleration_no_smooth": "Y Acceleration",
    "x_acceleration": "X Acceleration",
    "y_acceleration": "Y Acceleration",
    "acceleration_no_smooth": "Acceleration",
    "acceleration": "Acceleration"
}


class Point:
    @property
    def get_frames(self):
        return self.num_frames

    def __init__(self, data=None, joint_idx=None, joint_name=None, num_frames=0, fps=30, calibrationHelper=None, x_meter_per_pixel=None, y_meter_per_pixel=None) -> None:
        self.component = {}
        if data:
            self.initWithData(data=data, fps=fps)
        else:
            self.initWithEmptyFrames(num_frames=num_frames, fps=fps)

        # to break down do individual arguments
        if calibrationHelper:
            self.calibrationHelper = calibrationHelper
        else:  # default
            self.calibrationHelper = {
                "unit": "pixel",  # can be "pixel" or "meter
                'x_meter_per_pixel': 1.0 if x_meter_per_pixel is None else float(x_meter_per_pixel),
                'y_meter_per_pixel': 1.0 if y_meter_per_pixel is None else float(y_meter_per_pixel),
                'fps': 30 if fps is None else float(fps)
            }

        self.joint_idx = joint_idx
        self.joint_name = joint_name if joint_name else BODY_JOINTS_MAP[joint_idx] if joint_idx is not None else uuid.uuid4(
        ).hex

        # raw_ori_x and raw_ori_y are the original data
        # raw_x and raw_y are the interpolated data
        # x and y are the filtered data
        self.raw_x = np.full(self.num_frames, np.nan)
        self.raw_y = np.full(self.num_frames, np.nan)
        self.x = np.full(self.num_frames, np.nan)
        self.y = np.full(self.num_frames, np.nan)
        self.x_displacement = np.full(self.num_frames, np.nan)
        self.y_displacement = np.full(self.num_frames, np.nan)
        self.distance = np.full(self.num_frames, np.nan)
        self.x_speed = np.full(self.num_frames, np.nan)
        self.y_speed = np.full(self.num_frames, np.nan)
        self.speed = np.full(self.num_frames, np.nan)
        self.x_acceleration = np.full(self.num_frames, np.nan)
        self.y_acceleration = np.full(self.num_frames, np.nan)
        self.acceleration = np.full(self.num_frames, np.nan)
        # self.component = buildComponent(num_frames, PointMetrics)

    def initWithEmptyFrames(self, num_frames, fps):
        self.component['Raw Ori X'] = np.full(num_frames, np.nan)
        self.component['Raw Ori Y'] = np.full(num_frames, np.nan)

        self.raw_ori_x = np.full(num_frames, np.nan)
        self.raw_ori_y = np.full(num_frames, np.nan)
        self.num_frames = num_frames
        self.fps = fps

    def initWithData(self, data, fps=None):
        self.component['Raw Ori X'] = np.array(data)[:, 0]
        self.component['Raw Ori Y'] = np.array(data)[:, 1]

        self.raw_ori_x = np.array(data)[:, 0]
        self.raw_ori_y = np.array(data)[:, 1]
        self.num_frames = len(data)
        self.fps = fps

    # def __getattr__(self, __name: str) -> Any:
    #     return self.__name

    def __getitem__(self, __name: str) -> Any:
        return self.__dict__[__name]

    def preprocess(self, filter_threshold=None, filter_on=True, interpolate_on=True, preprocess_smoothing_on=None):
        # Find the first non-null data points from the beginning and end
        start_index, end_index = self.find_non_null_indices(self.raw_ori_x)

        self.start_index = start_index if start_index is not None else 0
        self.end_index = end_index if end_index is not None else self.num_frames - 1

        if interpolate_on:
            # Interpolate the data within the range defined by the first and last non-null data points
            self.raw_x[self.start_index:self.end_index + 1] = self.interpolate_missing(
                self.raw_ori_x[self.start_index:self.end_index + 1])
            self.raw_y[self.start_index:self.end_index + 1] = self.interpolate_missing(
                self.raw_ori_y[self.start_index:self.end_index + 1])
        else:
            self.raw_x = self.raw_ori_x
            self.raw_y = self.raw_ori_y

        # Proceed with filtering if the number of frames is greater than 10
        can_filter = filter_on and self.num_frames > 10

        if not can_filter:
            self.x = self.raw_x
            self.y = self.raw_y
        else:
            filter = FilteredTrajectory()

            # Apply filtering
            self.x[self.start_index:self.end_index + 1], self.y[self.start_index:self.end_index + 1] = filter.initialize(
                self.raw_x[self.start_index:self.end_index + 1], self.raw_y[self.start_index:self.end_index + 1])

        if preprocess_smoothing_on:
            self.x[self.start_index:self.end_index+1] = MovingAverage.filter_samples(
                self.x[self.start_index:self.end_index+1], self.calibrationHelper['fps'], 40, 1)
            self.y[self.start_index:self.end_index+1] = MovingAverage.filter_samples(
                self.y[self.start_index:self.end_index+1], self.calibrationHelper['fps'], 40, 1)

    def find_non_null_indices(self, data):
        # Find the first non-null data point from the beginning
        start_index = np.where(~np.isnan(data))[0]
        start_index = start_index[0] if start_index.size > 0 else None

        # Find the first non-null data point from the end
        end_index = np.where(~np.isnan(data))[0]
        end_index = end_index[-1] if end_index.size > 0 else None
        return start_index, end_index

    def interpolate_missing(self, data):
        # Create an array of indices where values are not missing
        valid_indices = np.where(~np.isnan(data))[0]

        # Perform linear interpolation
        f = interpolate.interp1d(
            valid_indices, data[valid_indices], kind='linear', fill_value="extrapolate")
        interpolated_data = f(np.arange(len(data)))

        return interpolated_data

    def compute(self, smoothing_on=True, smoothing_on_post=True, ):
        self.computeDistances()
        self.computeSpeeds(smoothing_on=smoothing_on)
        self.computeAccelerations(smoothing_on=smoothing_on)

    def computeDistances(self):
        # Calculate displacement relative to the first frame
        x_displacement = (self.x[self.start_index:self.end_index +
                                 1] - self.x[self.start_index]) * self.calibrationHelper['x_meter_per_pixel']

        y_displacement = (self.y[self.start_index:self.end_index +
                                 1] - self.y[self.start_index]) * self.calibrationHelper['y_meter_per_pixel']
        self.x_displacement[self.start_index:self.end_index +
                            1] = x_displacement
        self.y_displacement[self.start_index:self.end_index +
                            1] = y_displacement
        # Calculate distance relative to the first frame
        self.distance[self.start_index:self.end_index + 1] = np.sqrt(
            x_displacement**2 + y_displacement**2
        )

    def computeSpeeds(self, smoothing_on=True):
        time_diff = 1 / self.calibrationHelper['fps']
        x_speed = np.diff(
            self.x[self.start_index:self.end_index + 1], prepend=self.x[self.start_index]) / time_diff
        y_speed = np.diff(
            self.y[self.start_index:self.end_index + 1], prepend=self.y[self.start_index]) / time_diff
        self.x_speed[self.start_index:self.end_index+1] = x_speed
        self.y_speed[self.start_index:self.end_index+1] = y_speed

        if smoothing_on:
            self.x_speed_no_smooth = self.x_speed
            self.y_speed_no_smooth = self.y_speed
            self.speed_no_smooth = np.sqrt(
                self.x_speed_no_smooth**2 + self.y_speed_no_smooth**2)

            self.x_speed[self.start_index:self.end_index+1] = MovingAverage.filter_samples(
                self.x_speed[self.start_index:self.end_index+1], self.calibrationHelper['fps'], 40, 1)
            self.y_speed[self.start_index:self.end_index+1] = MovingAverage.filter_samples(
                self.y_speed[self.start_index:self.end_index+1], self.calibrationHelper['fps'], 40, 1)

        self.speed[self.start_index:self.end_index+1] = np.sqrt(
            x_speed**2 + y_speed**2)

    def computeAccelerations(self, smoothing_on=True):
        time_diff = 1 / self.calibrationHelper['fps']
        x_accel = np.diff(
            self.x_speed[self.start_index:self.end_index+1], prepend=self.x_speed[self.start_index]) / time_diff
        y_accel = np.diff(
            self.y_speed[self.start_index:self.end_index+1], prepend=self.y_speed[self.start_index]) / time_diff
        self.x_acceleration[self.start_index:self.end_index+1] = x_accel
        self.y_acceleration[self.start_index:self.end_index+1] = y_accel

        if smoothing_on:
            self.x_acceleration_no_smooth = self.x_acceleration
            self.y_acceleration_no_smooth = self.y_acceleration
            self.acceleration_no_smooth = np.sqrt(
                self.x_acceleration_no_smooth**2 + self.y_acceleration_no_smooth**2)

            self.x_acceleration[self.start_index:self.end_index+1] = MovingAverage.filter_samples(
                self.x_acceleration[self.start_index:self.end_index+1], self.calibrationHelper['fps'], 50, 1)
            self.y_acceleration[self.start_index:self.end_index+1] = MovingAverage.filter_samples(
                self.y_acceleration[self.start_index:self.end_index+1], self.calibrationHelper['fps'], 50, 1)
        self.acceleration[self.start_index:self.end_index+1] = np.sqrt(
            x_accel**2 + y_accel**2)

    def get_metrics(self):
        data = {
            "metadata":
            {
                "x_meter_per_pixel": self.calibrationHelper['x_meter_per_pixel'],
                "y_meter_per_pixel": self.calibrationHelper['y_meter_per_pixel'],
                "fps": self.calibrationHelper['fps'],
                "joint_idx": self.joint_idx,
                "name": self.joint_name if self.joint_name else BODY_JOINTS_MAP[self.joint_idx] if self.joint_idx is not None else None,
            },
            "metrics": {
                # coordinate
                "raw_ori_x": self.raw_ori_x.tolist(),
                "raw_ori_y": self.raw_ori_y.tolist(),
                "raw_x": self.raw_x.tolist(),
                "raw_y": self.raw_y.tolist(),
                "x": self.x.tolist(),
                "y": self.y.tolist(),

                # displacement
                "x_displacement": self.x_displacement.tolist(),
                "y_displacement": self.y_displacement.tolist(),
                "distance": self.distance.tolist(),

                # speed
                "x_speed_no_smooth" if hasattr(self, 'x_speed_no_smooth') else None: self.x_speed_no_smooth.tolist() if hasattr(self, 'x_speed_no_smooth') else None,
                "y_speed_no_smooth" if hasattr(self, 'y_speed_no_smooth') else None: self.y_speed_no_smooth.tolist() if hasattr(self, 'y_speed_no_smooth') else None,
                "x_speed": self.x_speed.tolist(),
                "y_speed": self.y_speed.tolist(),
                "speed_no_smooth" if hasattr(self, 'speed_no_smooth') else None: self.speed_no_smooth.tolist() if hasattr(self, 'speed_no_smooth') else None,
                "speed": self.speed.tolist(),

                # acceleration
                "x_acceleration_no_smooth" if hasattr(self, "x_acceleration_no_smooth") else None: self.x_acceleration_no_smooth.tolist() if hasattr(self, 'x_acceleration_no_smooth') else None,
                "y_acceleration_no_smooth" if hasattr(self, "y_acceleration_no_smooth") else None: self.y_acceleration_no_smooth.tolist() if hasattr(self, 'y_acceleration_no_smooth') else None,
                "y_acceleration": self.y_acceleration.tolist(),
                "x_acceleration": self.x_acceleration.tolist(),
                "y_acceleration": self.y_acceleration.tolist(),
                "acceleration_no_smooth" if hasattr(self, "acceleration_no_smooth") else None: self.acceleration_no_smooth.tolist() if hasattr(self, 'acceleration_no_smooth') else None,
                "acceleration": self.acceleration.tolist(),
            },
            "metrics_info":
            {
                # coordinate
                "raw_ori_x": {"Unit": "Pixel"},
                "raw_ori_y": {"Unit": "Pixel"},
                "raw_x": {"Unit": "Pixel"},
                "raw_y": {"Unit": "Pixel"},
                "x": {"Unit": "Pixel"},
                "y": {"Unit": "Pixel"},

                # displacement
                "x_displacement": {"Unit": "Meter" if self.calibrationHelper['x_meter_per_pixel'] is not None else "Pixel"},
                "y_displacement": {"Unit": "Meter" if self.calibrationHelper['y_meter_per_pixel'] is not None else "Pixel"},
                "distance": {"Unit": "Meter"} if self.calibrationHelper['x_meter_per_pixel'] is not None and self.calibrationHelper['y_meter_per_pixel'] is not None else {"Unit": "Pixel"},

                # speed
                "x_speed_no_smooth" if hasattr(self, 'x_speed_no_smooth') else None: {"Unit": "Pixel/s" if self.calibrationHelper['x_meter_per_pixel'] is None else "Meter/s"},
                "y_speed_no_smooth" if hasattr(self, 'y_speed_no_smooth') else None: {"Unit": "Pixel/s" if self.calibrationHelper['y_meter_per_pixel'] is None else "Meter/s"},
                "x_speed": {"Unit": "Pixel/s"} if self.calibrationHelper['x_meter_per_pixel'] is None else {"Unit": "Meter/s"},
                "y_speed": {"Unit": "Pixel/s" if self.calibrationHelper['y_meter_per_pixel'] is None else "Meter/s"},
                "speed_no_smooth" if hasattr(self, 'speed_no_smooth') else None: {"Unit": "Pixel/s" if self.calibrationHelper['x_meter_per_pixel'] is None and self.calibrationHelper['y_meter_per_pixel'] is None else "Meter/s"},
                "speed": {"Unit": "Pixel/s" if self.calibrationHelper['x_meter_per_pixel'] is None and self.calibrationHelper['y_meter_per_pixel'] is None else "Meter/s"},

                # acceleration
                "x_acceleration_no_smooth" if hasattr(self, 'x_acceleration_no_smooth') else None: {"Unit": "Pixel/s^2" if self.calibrationHelper['x_meter_per_pixel'] is None else "Meter/s^2"},
                "y_acceleration_no_smooth" if hasattr(self, 'y_acceleration_no_smooth') else None: {"Unit": "Pixel/s^2" if self.calibrationHelper['y_meter_per_pixel'] is None else "Meter/s^2"},
                "x_acceleration": {"Unit": "Pixel/s^2" if self.calibrationHelper['x_meter_per_pixel'] is None else "Meter/s^2"},
                "y_acceleration": {"Unit": "Pixel/s^2" if self.calibrationHelper['y_meter_per_pixel'] is None else "Meter/s^2"},
                "acceleration_no_smooth" if hasattr(self, 'acceleration_no_smooth') else None: {"Unit": "Pixel/s^2" if self.calibrationHelper['x_meter_per_pixel'] is None and self.calibrationHelper['y_meter_per_pixel'] is None else "Meter/s^2"},
                "acceleration": {"Unit": "Pixel/s^2" if self.calibrationHelper['x_meter_per_pixel'] is None and self.calibrationHelper['y_meter_per_pixel'] is None else "Meter/s^2"},
            },
            "time": [i / self.calibrationHelper['fps'] for i in range(self.num_frames)]
        }
        if None in data['metrics']:
            del data['metrics'][None]
            del data['metrics_info'][None]
        return data

    def export_metrics(self, human_dir):
        if self.joint_idx is not None:
            name = f"{self.joint_idx}_{BODY_JOINTS_MAP[self.joint_idx]}"
        else:
            name = self.joint_name
        # export json
        json_output = self.get_metrics()
        json_file = Path(human_dir) / f"{name}.json"
        with open(json_file, "w") as f:
            json.dump(json_output, f, indent=4)

    def export(self, human_dir, mode: Literal["compact", "complete"] = "compact", frequency=30, meter_per_pixel=None, metrics_to_export=list(PointMetricsMapping.keys())):
        # THINGS TO CONSIDER TO EXPORT JSON DATA
        # 1. frequency

        # THINGS TO CONSIDER TO EXPORT CHARTS
        # 1. what is the name of the diagram
        # 2. what is the x-ais unit? time or frame
        # 3. what is the y-axis unit? pixel or meter
        # 4. what is the dir to export? what is the png name?

        data = {
            "X Displacement": self.x_displacement.tolist() if meter_per_pixel is None else (self.x_displacement * meter_per_pixel).tolist(),
            "Y Displacement": self.y_displacement.tolist() if meter_per_pixel is None else (self.y_displacement * meter_per_pixel).tolist(),
            "X Coordinate": self.x.tolist() if meter_per_pixel is None else (self.x * meter_per_pixel).tolist(),
            "Y Coordinate": self.y.tolist() if meter_per_pixel is None else (self.y * meter_per_pixel).tolist(),
            "Distance": self.distance.tolist() if meter_per_pixel is None else (self.distance * meter_per_pixel).tolist(),
            "X Speed": self.x_speed.tolist() if meter_per_pixel is None else (self.x_speed * meter_per_pixel).tolist(),
            "Y Speed": self.y_speed.tolist() if meter_per_pixel is None else (self.y_speed * meter_per_pixel).tolist(),
            "Speed": self.speed.tolist() if meter_per_pixel is None else (self.speed * meter_per_pixel).tolist(),
            "X Acceleration": self.x_acceleration.tolist() if meter_per_pixel is None else (self.x_acceleration * meter_per_pixel).tolist(),
            "Y Acceleration": self.y_acceleration.tolist() if meter_per_pixel is None else (self.y_acceleration * meter_per_pixel).tolist(),
            "Acceleration": self.acceleration.tolist() if meter_per_pixel is None else (self.acceleration * meter_per_pixel).tolist(),
        }

        if self.joint_idx is not None:
            name = f"{self.joint_idx}_{BODY_JOINTS_MAP[self.joint_idx]}"
        else:
            name = self.joint_name
        if mode == 'compact':
            num_metrics = len(metrics_to_export)
            num_cols = 3
            # unlimited rows
            num_rows = num_metrics // num_cols + \
                1 if num_metrics % num_cols != 0 else num_metrics // num_cols

            # Adjust the figsize as needed
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(8, 6))

            for i, metric in enumerate(metrics_to_export):
                row = i // num_cols
                col = i % num_cols
                ax = axes[row, col]
                ax.plot(range(self.num_frames), data[metric])
                ax.set_xlabel("Frame") if frequency is not None else ax.set_xlabel(
                    "Time (s)")
                ax.set_ylabel(
                    "Meter") if meter_per_pixel is not None else ax.set_ylabel("Pixel")
                ax.set_title(f"{metric}")  # Add chart title

            chart_filename = Path(human_dir)
            chart_filename.mkdir(parents=True, exist_ok=True)

            chart_filename = chart_filename / \
                f"{name}_compact.png"
            plt.tight_layout()
            plt.savefig(chart_filename)
            plt.close()

        elif mode == 'complete':
            for each in metrics_to_export:
                plt.figure(figsize=(8, 6))
                plt.plot(range(self.num_frames), data[each])
                plt.xlabel("Frame") if frequency is not None else plt.xlabel(
                    "Time (s)")
                plt.ylabel("Meter") if meter_per_pixel is not None else plt.ylabel(
                    "Pixel")
                plt.title(f"{each}")
                chart_filename = Path(
                    human_dir) / f"{self.joint_idx}_{BODY_JOINTS_MAP[self.joint_idx]}"
                chart_filename.mkdir(parents=True, exist_ok=True)
                chart_filename = chart_filename / \
                    f"{each.lower().replace(' ', '_')}_chart.png"
                plt.savefig(chart_filename)
                plt.close()

        # export json
        json_output = self.get_metrics()
        json_file = Path(human_dir) / f"{name}.json"
        with open(json_file, "w") as f:
            json.dump(json_output, f, indent=4)

    def updateXY():
        # only happens if passed n frames of xy keypoints
        pass

    def loadFromJson(self, data, calibrationHelper=None, x_meter_per_pixel=None, y_meter_per_pixel=None, fps=None):
        # validate data shape has 2 coordinate
        # (n, 2)
        assert len(data[0]) == 2
        data = np.array(data)
        self.raw_ori_x = data[:, 0]
        self.raw_ori_y = data[:, 1]
        self.num_frames = len(data)
        if calibrationHelper:
            self.calibrationHelper = calibrationHelper
        else:
            self.calibrationHelper = {
                'x_meter_per_pixel': 1.0 if x_meter_per_pixel is None else float(x_meter_per_pixel),
                'y_meter_per_pixel': 1.0 if y_meter_per_pixel is None else float(y_meter_per_pixel),
                'fps': 30 if fps is None else float(fps)
            }
