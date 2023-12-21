import csv
import json
import os
import uuid
from collections import OrderedDict
from io import StringIO
from pathlib import Path
from typing import Any, Generator, List, Literal, Optional, Union

import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate

from .constants import BODY_JOINTS_MAP
from .filtering import FilteredTrajectory
from .point import Point, PointMetrics, PointMetricsMapping
from .smoothing import MovingAverage
from .utils import buildComponent, calculate_angle

AngleMetrics = ["Angular Degree", "Angular Velocity"]


class Angle:
    def __init__(self, angle_name=None, dependencies=None):
        self.angle_name = angle_name if angle_name is not None else uuid.uuid4().hex
        if dependencies:
            self.dependencies = dependencies
            self.num_frames = self.dependencies[0].get_frames

    def compute(self):
        assert len(self.dependencies) == 3  # make sure got three points
        time = 1 / self.dependencies[0].calibrationHelper['fps']
        if self.dependencies is not None:
            # check if all dependencies have the same num_frames, and have x and y
            for each in self.dependencies:
                if np.isnan(each.x).all() or np.isnan(each.y).all():
                    each.preprocess()
                    each.compute()
            assert all(
                [each.num_frames == self.num_frames for each in self.dependencies])
            self.angle = calculate_angle(
                self.dependencies[0], self.dependencies[1], self.dependencies[2])
            self.angle_velocity = np.diff(
                self.angle, prepend=self.angle[0]) / time
            self.angle_acceleration = np.diff(
                self.angle_velocity, prepend=self.angle_velocity[0]) / time

    def get_metrics(self):
        return {
            "metadata": {
                "x_meter_per_pixel": self.dependencies[0].calibrationHelper['x_meter_per_pixel'],
                "y_meter_per_pixel": self.dependencies[0].calibrationHelper['y_meter_per_pixel'],
                "fps": self.dependencies[0].calibrationHelper['fps'],
                "unit": "degrees",
                "name": self.angle_name
            },
            "metrics": {
                "angle": self.angle.tolist(),
                "angle_velocity": self.angle_velocity.tolist(),
                "angle_acceleration": self.angle_acceleration.tolist(),
            }
        }

    def export(self, human_dir, mode: Literal["compact", "complete"] = 'compact', frequency=30, meter_per_pixel=None, metrics_to_export=PointMetrics):
        if self.dependencies is not None:
            data = {
                "Angle": self.angle.tolist(),
                "Angle Velocity": self.angle_velocity.tolist(),
                "Angle Acceleration": self.angle_acceleration.tolist(),
            }
            if mode == 'compact':
                num_metrics = len(data)
                num_rows = 3  # Number of rows for the 3x3 grid
                # Adjust the figsize as needed
                fig, axes = plt.subplots(num_rows, figsize=(8, 6))

                for i, metric in enumerate(data):
                    ax = axes[i]
                    ax.plot(range(self.num_frames), data[metric])
                    ax.set_xlabel("Frame") if frequency is not None else ax.set_xlabel(
                        "Time (s)")
                    # You can adjust the unit accordingly
                    ax.set_ylabel("Degrees")
                    ax.set_title(metric)
                chart_file = Path(human_dir) / f"{self.angle_name}_compact.png"
                plt.tight_layout()
                plt.savefig(chart_file)
                plt.close()

            elif mode == 'complete':
                for metric in data:
                    plt.figure(figsize=(8, 6))
                    plt.plot(range(self.num_frames), data[metric])
                    plt.xlabel("Frame") if frequency is not None else plt.xlabel(
                        "Time (s)")
                    # You can adjust the unit accordingly
                    plt.ylabel("Degrees")
                    plt.title(metric)
                    chart_file = Path(human_dir) / f"{self.angle_name}"
                    chart_file.mkdir(parents=True, exist_ok=True)
                    chart_file = chart_file / \
                        f"{metric.lower().replace(' ', '_')}_chart.png"
                    plt.savefig(chart_file)
                    plt.close()

            # export json
            json_output = self.get_metrics()
            json_file = Path(human_dir) / f"{self.angle_name}.json"
            with open(json_file, "w") as f:
                json.dump(json_output, f, indent=4)

    def loadFromJson(self, data, calibrationHelper=None, x_meter_per_pixel=None, y_meter_per_pixel=None, fps=None):
        # assert data shape
        assert len(data) == 3
        assert len(data['o'][0]) == 2
        self.dependencies = [Point(data=data['a'], num_frames=len(data['a'])),
                             Point(data=data['o'], num_frames=len(data['o'])),
                             Point(data=data['b'], num_frames=len(data['b']))]
        self.num_frames = self.dependencies[0].get_frames


class Record:
    def __init__(self) -> Any:
        self.frequency = None
        pass

    def compute(self) -> Any:

        pass

    def export(self):
        pass

    def update(self):
        pass


class HumanProfile(Record):
    def __init__(self, human_idx=None, num_frames=None, x_meter_per_pixel=None, y_meter_per_pixel=None, fps=None) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.human_idx = human_idx
        self.calibrationHelper = {
            'x_meter_per_pixel': 1.0 if x_meter_per_pixel is None else float(x_meter_per_pixel),
            'y_meter_per_pixel': 1.0 if y_meter_per_pixel is None else float(y_meter_per_pixel),
            'fps': 30 if fps is None else float(fps)
        }
        self.body_joints = {
            i: Point(num_frames=num_frames, joint_idx=i,
                     calibrationHelper=self.calibrationHelper)
            for i in range(17)
        }

    def appendFrame(self, points: list[list[float]], frame_no) -> None:
        """
        Args:
            points (list[list[float]]): shape -> (17, 2)
        """
        assert len(points) == 17, f"Expected 17 points, got {len(points)}"
        assert len(
            points[0]) == 2, f"Expected 2 coordinates, got {len(points[0])}"

        for index, point in enumerate(points):
            self.body_joints[index].raw_ori_x[
                frame_no
            ], self.body_joints[index].raw_ori_y[
                frame_no
            ] = point[0], point[1]
            # append the coordinates of each joint as [x, y], but still will have empty points

    def setXY(self, points: list[list[list[float]]]):
        pass

    def compute(self, filter_threshold=None, filter_on=None, interpolate_on=None, smoothing_on=None, preprocess_smoothing_on=None) -> Any:
        for each, body_joint in self.body_joints.items():
            body_joint.preprocess(filter_threshold=filter_threshold,
                                  filter_on=filter_on, interpolate_on=interpolate_on, preprocess_smoothing_on=True)
            body_joint.compute(smoothing_on=smoothing_on)

        # cater for custom metrics
        self.customRecord = {
            'left_angle': Angle(angle_name='Left Angle', dependencies=[self.body_joints[11], self.body_joints[13], self.body_joints[15]]),
            'right_angle': Angle(angle_name='Right Angle', dependencies=[self.body_joints[12], self.body_joints[14], self.body_joints[16]]),
            # TODO
            # 'center_of_mass' : Point()
        }

        for each, customRecord in self.customRecord.items():
            customRecord.compute()

    def associate_human_profile(self, associate_human: List['HumanProfile']):
        assert self == associate_human[0]
        for frame in range(self.num_frames):
            if self.body_joints[0].raw_ori_x[frame] is not None:
                pass
            else:
                for each in associate_human[1:]:
                    if each[frame] is not None:
                        for each_joint in self.body_joints:
                            self.body_joints[each_joint].raw_ori_x[frame] = each.body_joints[each_joint].raw_ori_x[frame]
                            self.body_joints[each_joint].raw_ori_y[frame] = each.body_joints[each_joint].raw_ori_y[frame]
                        break
                    else:
                        continue
        return self

    def get_metrics(self):
        data = {
            "metadata": {
                "x_meter_per_pixel": self.calibrationHelper['x_meter_per_pixel'],
                "y_meter_per_pixel": self.calibrationHelper['y_meter_per_pixel'],
                "fps": self.calibrationHelper['fps'],
                "human_idx": self.human_idx,
            },
            "body_joints": [each.get_metrics() for each in self.body_joints.values()],
            "custom_records": [each.get_metrics() for each in self.customRecord.values()],
            "time": [i / self.calibrationHelper['fps'] for i in range(self.num_frames)]
        }
        return data

    def export(self, save_dirs=None, mode=None, frequency=30, metrics_to_export=list(PointMetrics), meter_per_pixel=None):
        human_dir = Path(save_dirs) / f"human_{self.human_idx}"
        for idx, (each, body_joint) in enumerate(self.body_joints.items()):
            body_joint.export(human_dir=human_dir, mode=mode, frequency=frequency,
                              meter_per_pixel=meter_per_pixel, metrics_to_export=metrics_to_export)

        for each, customRecord in self.customRecord.items():
            customRecord.export(human_dir=human_dir, mode=mode, frequency=frequency,
                                meter_per_pixel=meter_per_pixel, metrics_to_export=metrics_to_export)

    def export_metrics(self, save_dirs=None):
        data = self.get_metrics()
        # save the json
        human_dir = Path(save_dirs) / f"human_{self.human_idx}"
        json_file = Path(human_dir) / "metrics.json"
        with open(json_file, "w") as f:
            json.dump(data, f, indent=4)

class Kinematics(Record):
    def __init__(self, num_frames=None, save_dirs=None, results=None, run_path=None, ) -> None:
        super().__init__()
        self.frequency = None
        self.profileSet: dict[str, Record] = {}
        self.source_path = None
        self.save_dirs = None
        self.num_frames = None
        self.calibrationHelper = None

        if results is not None:
            self.setup(results)
        elif run_path is not None:
            self.setup_from_path(run_path)

    def setup():
        pass

    def setup_from_path(self, run_csv) -> None:
        # Get the file name without extension using pathlib library
        if isinstance(run_csv, str):
            self.source_path = Path(run_csv)
            with open(run_csv, "r") as csv_file:
                csv_reader = csv.reader(csv_file)

                csv_list = list(csv_reader)
                self.num_frames = len(csv_list)
                for row in csv_list:
                    frame_number, keypoints_data = row
                    frame_number = int(frame_number)
                    # Parse the JSON string
                    keypoints_data = json.loads(keypoints_data)
                    self.process_frame(frame_number - 1, keypoints_data)
        elif isinstance(run_csv, StringIO):
            self.source_path = Path(__file__).resolve() / "temp.csv"
            csv_reader = csv.reader(run_csv)
            csv_list = list(csv_reader)
            self.num_frames = len(csv_list)
            for row in csv_list:
                frame_number, keypoints_data = row
                frame_number = int(frame_number)
                # Parse the JSON string
                keypoints_data = json.loads(keypoints_data)
                self.process_frame(frame_number - 1, keypoints_data)

    def setup_from_json(self, json):
        pass

    def process_frame(self, frame_number, keypoints_data):
        # make sure correct dimension
        for profile_id, profile_points in keypoints_data.items():
            if profile_id == "":  # cuz there's empty datapoints
                continue
            profile_id = int(profile_id)  # Convert profile_id to float
            if profile_id not in self.profileSet:
                self.profileSet[profile_id] = HumanProfile(
                    num_frames=self.num_frames, human_idx=profile_id)
            self.profileSet[profile_id].appendFrame(
                profile_points, frame_no=frame_number
            )

    def associate_human_profile(self, associate_human=[[]]):
        # rmb delete profile
        if len(associate_human) == 0 and len(associate_human[0]) == 0:
            return
        for each in associate_human:
            toRetain = each[0]
            self.profileSet[toRetain].associate_human_profile(
                [self.profileSet[i] for i in each])
            for toRemove in each[1:]:
                self.profileSet.pop(toRemove)

    def compute(self, angle=None, filter_on=None, interpolate_on=None, smoothing_on=None, preprocess_smoothing_on=None):
        for each, profile in self.profileSet.items():
            profile.compute(
                filter_on=filter_on, interpolate_on=interpolate_on, smoothing_on=smoothing_on, preprocess_smoothing_on=preprocess_smoothing_on)

    def export(self, mode: Literal['compact, complete'], save_dirs=None, overwrite=False, meter_per_pixel=None) -> None:
        save_dirs = Path(
            self.source_path).parent if save_dirs is None else save_dirs
        for each, profile in self.profileSet.items():
            profile.export(save_dirs=save_dirs, mode=mode,
                           meter_per_pixel=meter_per_pixel)

    def export_metrics(self, save_dirs=None) -> None:
        data = {
            "metadata": {
                "num_frames": self.num_frames,
                "calibrationHelper": self.calibrationHelper
            },
            "human_profiles": [each.get_metrics() for each in self.profileSet.values()]
        }
        # save the json
        save_dirs = Path(
            self.source_path).parent if save_dirs is None else save_dirs
        json_file = Path(save_dirs) / "metrics.json"
        with open(json_file, "w") as f:
            json.dump(data, f, indent=4)
        return data

    def __call__(self, save=True, save_dirs=None, overwrite=True, mode='compact', meter_per_pixel=None, associate_human_profile=[], onlyHumanProfile=[], filter_on=None, interpolate_on=None, smoothing_on=None, preprocess_smoothing_on=None) -> Any:
        # associate human profile
        if len(associate_human_profile) > 0 and len(associate_human_profile[0]) > 0:
            self.associate_human_profile(associate_human_profile)

        if len(onlyHumanProfile) > 0:
            self.profileSet = {
                each: self.profileSet[each] for each in onlyHumanProfile}

        # compute
        self.compute(filter_on=filter_on,
                     interpolate_on=interpolate_on, smoothing_on=smoothing_on, preprocess_smoothing_on=preprocess_smoothing_on)

        if save:
            # self.export(save_dirs=save_dirs, overwrite=overwrite,
            #             mode=mode, meter_per_pixel=meter_per_pixel)
            self.export_metrics(save_dirs=save_dirs)
