import json
import uuid
from pathlib import Path
from typing import Literal

import numpy as np
from matplotlib import pyplot as plt

from .point import Point
from .smoothing import MovingAverage
from .utils import calculate_angle, get_time


class Angle:
    def __init__(
        self,
        angle_name=None,
        dependencies=None,
        num_frames=0,
        fps=30,
        x_meter_per_pixel=None,
        y_meter_per_pixel=None,
    ):
        self.calibrationHelper = {
            "unit": "pixel",  # can be "pixel" or "meter
            "x_meter_per_pixel": 1.0
            if x_meter_per_pixel is None
            else float(x_meter_per_pixel),
            "y_meter_per_pixel": 1.0
            if y_meter_per_pixel is None
            else float(y_meter_per_pixel),
            "fps": 30 if fps is None else float(fps),
            "num_frames": 0 if num_frames is None else num_frames,
        }

        if dependencies:
            self.init_dependencies(dependencies)

        self.angle_name = f"{angle_name}" if angle_name else f"angle_{uuid.uuid4().hex}"
        self.data = {}
        self.smoothing_helper = None

    def init_dependencies(self, dependencies):
        self.dependencies = dependencies
        self.calibrationHelper = self.dependencies[0].calibrationHelper

    def preprocess_dependent_points(
        self,
    ):
        if self.dependencies:
            assert len(self.dependencies) == 3  # make sure got three points
            if self.dependencies:
                # check if all dependencies have the same num_frames, and have x and y
                for each in self.dependencies:
                    if np.isnan(each.data["X"]).all() or np.isnan(each.data["Y"]).all():
                        each.preprocess()
                self.calibrationHelper = self.dependencies[0].calibrationHelper
                assert all(
                    [
                        each.calibrationHelper["num_frames"]
                        == self.calibrationHelper["num_frames"]
                        for each in self.dependencies
                    ]
                )

    def preprocess(
        self,
        preprocess_interpolate_on=True,
        preprocess_filter_on=False,
        preprocess_smoothing_on=False,
    ):
        pass
        self.preprocess_dependent_points()

    def compute(self, postcalculate_filter_on=False, postcalculate_smoothing_on=False):
        self.computeAngle(
            self.dependencies[0], self.dependencies[1], self.dependencies[2]
        )
        self.computeAngleVelocity(smoothing_on=postcalculate_smoothing_on)
        self.computeAngleAcceleration(smoothing_on=postcalculate_smoothing_on)

    def computeAngle(self, point1, point2, point3):
        # compute angle
        self.data["Angle"] = calculate_angle(
            self.dependencies[0], self.dependencies[1], self.dependencies[2]
        )

    def computeAngleVelocity(self, smoothing_on=False):
        # compute angle velocity
        time = 1 / self.dependencies[0].calibrationHelper["fps"]
        self.data["Angle Velocity"] = (
            np.diff(self.data["Angle"], prepend=self.data["Angle"][0]) / time
        )

        if smoothing_on:
            self.data["Angle Velocity Smoothing"] = self.data["Angle Velocity"].copy()
            smoothing = self.smoothing_helper or MovingAverage()
            self.data["Angle Velocity Smoothing"] = smoothing.smooth(
                self.data["Angle Velocity Smoothing"]
            )
            # Update Angle Velocity
            self.data["Angle Velocity"] = self.data["Angle Velocity Smoothing"]

    def computeAngleAcceleration(self, smoothing_on=False):
        # compute angle acceleration
        time = 1 / self.dependencies[0].calibrationHelper["fps"]
        self.data["Angle Acceleration"] = (
            np.diff(self.data["Angle Velocity"], prepend=self.data["Angle Velocity"][0])
            / time
        )

        if smoothing_on:
            self.data["Angle Acceleration Smoothing"] = self.data[
                "Angle Acceleration"
            ].copy()
            smoothing = self.smoothing_helper or MovingAverage()
            self.data["Angle Acceleration Smoothing"] = smoothing.smooth(
                self.data["Angle Acceleration Smoothing"]
            )
            # Update Angle Acceleration
            self.data["Angle Acceleration"] = self.data["Angle Acceleration Smoothing"]

    def get_metrics(self):
        return {
            "metadata": {
                "x_meter_per_pixel": self.dependencies[0].calibrationHelper[
                    "x_meter_per_pixel"
                ],
                "y_meter_per_pixel": self.dependencies[0].calibrationHelper[
                    "y_meter_per_pixel"
                ],
                "fps": self.dependencies[0].calibrationHelper["fps"],
                "unit": "degrees",
                "name": self.angle_name,
            },
            "metrics": {
                "O Point X": self.dependencies[0].data["X"].tolist(),
                "O Point Y": self.dependencies[0].data["Y"].tolist(),
                "A Point X": self.dependencies[1].data["X"].tolist(),
                "A Point Y": self.dependencies[1].data["Y"].tolist(),
                "B Point X": self.dependencies[2].data["X"].tolist(),
                "B Point Y": self.dependencies[2].data["Y"].tolist(),
                "Angle": self.data["Angle"].tolist(),
                "Angle Velocity": self.data["Angle Velocity"].tolist(),
                "Angle Velocity Smoothing"
                if "Angle Velocity Smoothing" in self.data
                else None: self.data["Angle Velocity Smoothing"].tolist()
                if "Angle Velocity Smoothing" in self.data
                else None,
                "Angle Acceleration": self.data["Angle Acceleration"].tolist(),
                "Angle Acceleration Smoothing"
                if "Angle Acceleration Smoothing" in self.data
                else None: self.data["Angle Acceleration Smoothing"].tolist()
                if "Angle Acceleration Smoothing" in self.data
                else None,
            },
            "metrics_info": {
                "O Point X": {
                    "unit": "pixel",
                },
                "O Point Y": {
                    "unit": "pixel",
                },
                "A Point X": {
                    "unit": "pixel",
                },
                "A Point Y": {
                    "unit": "pixel",
                },
                "B Point X": {
                    "unit": "pixel",
                },
                "B Point Y": {
                    "unit": "pixel",
                },
                "Angle": {
                    "unit": "degrees",
                },
                "Angle Velocity": {
                    "unit": "degrees/s",
                },
                "Angle Velocity Smoothing"
                if "Angle Velocity Smoothing" in self.data
                else None: {
                    "unit": "degrees/s",
                }
                if "Angle Velocity Smoothing" in self.data
                else None,
                "Angle Acceleration": {
                    "unit": "degrees/s^2",
                },
                "Angle Acceleration Smoothing"
                if "Angle Acceleration Smoothing" in self.data
                else None: {
                    "unit": "degrees/s^2",
                }
                if "Angle Acceleration Smoothing" in self.data
                else None,
            },
        }

    def export_metrics(self, dir):
        name = f"{self.angle_name}"

        # export json
        json_output = self.get_metrics()
        json_file = Path(dir) / get_time()
        json_file = json_file / f"{name}.json"
        json_file.parent.mkdir(parents=True, exist_ok=True)
        with open(json_file, "w") as f:
            json.dump(json_output, f, indent=2)

    def export(
        self,
        human_dir,
        mode: Literal["compact", "complete"] = "compact",
        frequency=30,
        meter_per_pixel=None,
        metrics_to_export=None,
    ):
        if self.dependencies is not None:
            data = {
                "Angle": self.data["Angle"].tolist(),
                "Angle Velocity": self.data["Angle Velocity"].tolist(),
                "Angle Acceleration": self.self.data["Angle Acceleration"].tolist(),
            }
            if mode == "compact":
                num_rows = 3  # Number of rows for the 3x3 grid
                # Adjust the figsize as needed
                fig, axes = plt.subplots(num_rows, figsize=(8, 6))

                for i, metric in enumerate(data):
                    ax = axes[i]
                    ax.plot(range(self.num_frames), data[metric])
                    ax.set_xlabel("Frame") if frequency is not None else ax.set_xlabel(
                        "Time (s)"
                    )
                    # You can adjust the unit accordingly
                    ax.set_ylabel("Degrees")
                    ax.set_title(metric)
                chart_file = Path(human_dir) / f"{self.angle_name}_compact.png"
                plt.tight_layout()
                plt.savefig(chart_file)
                plt.close()

            elif mode == "complete":
                for metric in data:
                    plt.figure(figsize=(8, 6))
                    plt.plot(range(self.num_frames), data[metric])
                    plt.xlabel("Frame") if frequency is not None else plt.xlabel(
                        "Time (s)"
                    )
                    # You can adjust the unit accordingly
                    plt.ylabel("Degrees")
                    plt.title(metric)
                    chart_file = Path(human_dir) / f"{self.angle_name}"
                    chart_file.mkdir(parents=True, exist_ok=True)
                    chart_file = (
                        chart_file / f"{metric.lower().replace(' ', '_')}_chart.png"
                    )
                    plt.savefig(chart_file)
                    plt.close()

            # export json
            json_output = self.get_metrics()
            json_file = Path(human_dir) / f"{self.angle_name}.json"
            with open(json_file, "w") as f:
                json.dump(json_output, f, indent=4)

    def loadFromJson(
        self,
        data,
        calibrationHelper=None,
        x_meter_per_pixel=None,
        y_meter_per_pixel=None,
        fps=None,
    ):
        # assert data shape
        assert len(data) == 3
        assert len(data["o"][0]) == 2
        self.dependencies = [
            Point(data=data["a"], num_frames=len(data["a"])),
            Point(data=data["o"], num_frames=len(data["o"])),
            Point(data=data["b"], num_frames=len(data["b"])),
        ]
        self.num_frames = self.dependencies[0].get_frames
