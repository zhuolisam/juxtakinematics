import csv
import json
import uuid
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd
from .angle import Angle
from .point import Point, PointMetrics
from .utils import get_time

# Two methods to init Human
# 1. We already know the len, in Point we can init empty frames and replace index
# 2. We don't know the len, in Point we can use append function


class HumanProfile:
    def __init__(
        self,
        human_idx=None,
        clean_data=None,
        frame_data=None,
        run_path=None,
        x_meter_per_pixel=None,
        y_meter_per_pixel=None,
        fps=None,
        body_format="coco",
    ) -> None:
        self.human_idx = human_idx
        self.name = f"{human_idx}" if human_idx else f"human_{uuid.uuid4().hex}"
        self.calibrationHelper = {
            "x_meter_per_pixel": 1.0
            if x_meter_per_pixel is None
            else float(x_meter_per_pixel),
            "y_meter_per_pixel": 1.0
            if y_meter_per_pixel is None
            else float(y_meter_per_pixel),
            "fps": 30 if fps is None else float(fps),
        }
        if body_format is None or body_format == "coco":
            self.body_joints = {
                i: Point(
                    point_idx=i,
                    x_meter_per_pixel=self.calibrationHelper["x_meter_per_pixel"],
                    y_meter_per_pixel=self.calibrationHelper["y_meter_per_pixel"],
                    fps=self.calibrationHelper["fps"],
                    num_frames=0,
                )
                for i in range(17)
            }
        if clean_data is not None:
            self.init_with_data(clean_data)
        elif frame_data is not None:
            self.process_frame_and_append(frame_data)
        elif run_path is not None:
            self.setup_from_csv(run_path)

        self.customRecord = {}

    def compute(
        self,
        filter_threshold=None,
        preprocess_interpolate_on=False,
        preprocess_filter_on=False,
        preprocess_smoothing_on=False,
        postcalculate_filter_on=False,
        postcalculate_smoothing_on=False,
    ) -> Any:
        for each, body_joint in self.body_joints.items():
            body_joint.preprocess(
                filter_threshold=filter_threshold,
                preprocess_filter_on=preprocess_filter_on,
                preprocess_interpolate_on=preprocess_interpolate_on,
                preprocess_smoothing_on=preprocess_smoothing_on,
            )
            body_joint.compute(
                postcalculate_filter_on=postcalculate_filter_on,
                postcalculate_smoothing_on=postcalculate_smoothing_on,
            )

        self.customRecord["Left Knee"] = Angle(
            angle_name="Left Knee",
            dependencies=[
                self.body_joints[11],
                self.body_joints[13],
                self.body_joints[15],
            ],
        )

        self.customRecord["Right Knee"] = Angle(
            angle_name="Right Knee",
            dependencies=[
                self.body_joints[12],
                self.body_joints[14],
                self.body_joints[16],
            ],
        )

        self.customRecord["Left Hip"] = Angle(
            angle_name="Left Hip",
            dependencies=[
                self.body_joints[5],
                self.body_joints[11],
                self.body_joints[13],
            ],
        )

        self.customRecord["Right Hip"] = Angle(
            angle_name="Right Hip",
            dependencies=[
                self.body_joints[6],
                self.body_joints[12],
                self.body_joints[14],
            ],
        )

        self.customRecord["Left Shoulder"] = Angle(
            angle_name="Left Shoulder",
            dependencies=[
                self.body_joints[7],
                self.body_joints[5],
                self.body_joints[11],
            ],
        )

        self.customRecord["Right Shoulder"] = Angle(
            angle_name="Right Shoulder",
            dependencies=[
                self.body_joints[8],
                self.body_joints[6],
                self.body_joints[12],
            ],
        )

        self.customRecord["Left Elbow"] = Angle(
            angle_name="Left Elbow",
            dependencies=[
                self.body_joints[5],
                self.body_joints[7],
                self.body_joints[9],
            ],
        )

        self.customRecord["Right Elbow"] = Angle(
            angle_name="Right Elbow",
            dependencies=[
                self.body_joints[6],
                self.body_joints[8],
                self.body_joints[10],
            ],
        )

        # TODO
        # 'center_of_mass' : Point()

        for each, customRecord in self.customRecord.items():
            customRecord.preprocess()
            customRecord.compute()

    def associate_human_profile(self, associate_human: List["HumanProfile"]):
        assert self == associate_human[0]
        for frame in range(self.get_num_frames):
            x, y = self.body_joints[0].get_frame_point(frame)
            if x and y:
                pass
            else:
                for each in associate_human[1:]:
                    x, y = each.body_joints[0].get_frame_point(frame)
                    if x and y:
                        for joint_idx, each_joint in self.body_joints.items():
                            each_joint.set_frame_point(frame, x, y)
                        break

    # GETTERS
    @property
    def get_num_frames(self):
        n = len(self.body_joints[0].data["Raw Ori X"])
        assert n == len(
            self.body_joints[0].data["Raw Ori Y"]
        ), "Inconsistent data length"
        return n

    def get_joint_frame_point(self, frame):
        return self.body_joints[0].get_frame_point(frame)

    def get_body_joints_data(self):
        data = {}
        for each, body_joint in self.body_joints.items():
            data[each] = {key: item.tolist() for key, item in body_joint.data}

    def get_metadata(self):
        return (
            {
                "x_meter_per_pixel": self.calibrationHelper["x_meter_per_pixel"],
                "y_meter_per_pixel": self.calibrationHelper["y_meter_per_pixel"],
                "fps": self.calibrationHelper["fps"],
                "human_idx": self.human_idx,
                "name": self.name,
                "start_index": self.body_joints[0].get_metadata()["start_index"],
                "end_index": self.body_joints[0].get_metadata()["end_index"],
            },
        )

    def get_metrics(self):
        data = {
            "metadata": self.get_metadata(),
            "body_joints_metrics": {
                # key: {
                #     "metrics": joint.get_metrics()["metrics"],
                #     "metadata": {
                #         "name": joint.get_metrics()["metadata"]["name"],
                #         "start_index": joint.get_metrics()["metadata"]["start_index"],
                #         "end_index": joint.get_metrics()["metadata"]["start_index"],
                #     },
                # }
                key: joint.get_metrics_data()
                for key, joint in self.body_joints.items()
            },
            "body_joints_metrics_info": self.body_joints[0].get_metrics_info(),
            "custom_metrics": {
                key: metric.get_metrics()["metrics"]
                for key, metric in self.customRecord.items()
            },
            "custom_metrics_info": self.customRecord["Left Knee"].get_metrics()[
                "metrics_info"
            ],
            "time": [
                i / self.calibrationHelper["fps"] for i in range(self.get_num_frames)
            ],
        }

        return data

    # SETTERS AND IMPORT
    def setup_from_csv(self, run_csv, human_idx=1):
        if isinstance(run_csv, str):
            self.source_path = Path(run_csv)
            with open(run_csv, "r") as csv_file:
                csv_reader = csv.reader(csv_file)

                csv_list = list(csv_reader)

                accumulated_data = []  # Concat data of only one human from all frames
                for row in csv_list:
                    frame_number, frame_data = row
                    frame_number = int(frame_number)
                    # Parse the JSON string
                    frame_data = json.loads(frame_data)

                    processed_frame_data = self.process_frame(
                        frame_data, human_idx=self.human_idx or human_idx
                    )  # TODO: This is hardcoded to 1, need to change to self.human_idx or the parameter

                    accumulated_data.append(processed_frame_data)
                accumulated_data = np.array(accumulated_data, dtype=float)
                self.init_with_data(accumulated_data)

    def setup_from_data(self, data, human_idx=1):
        # Exact same as setup_from_csv, but we don't need to parse the JSON string, already formatted as [frames][dict:[17 joints][x, y]]
        # It's taking all frames and all humans, but it only keeps the data of one human
        accumulated_data = []
        for frame_data in data:
            processed_frame_data = self.process_frame(
                frame_data, human_idx=self.human_idx or human_idx
            )
            accumulated_data.append(processed_frame_data)
        accumulated_data = np.array(accumulated_data, dtype=float)
        self.init_with_data(accumulated_data)

    def process_frame(self, frame_data, human_idx=1):
        if isinstance(human_idx, int):
            human_idx = str(human_idx)
        if human_idx in frame_data:
            return frame_data[human_idx]
        else:
            return [[None, None] for _ in range(len(self.body_joints))]

    def process_frame_and_append(self, frame_data, human_idx=1):
        if isinstance(human_idx, int):
            human_idx = str(human_idx)

        if human_idx in frame_data:
            self.process_profile_and_append(frame_data[human_idx])
            return frame_data[human_idx]
        else:
            self.append_empty_frames(frames=1)
            return [[None, None] for _ in range(len(self.body_joints))]

    def process_profile_and_append(self, points: list[list[float]]) -> None:
        assert len(points) == len(
            self.body_joints
        ), f"Expected {len(self.body_joints)} points, got {len(points)}"

        for each, body_joint in self.body_joints.items():
            body_joint.append_points(points[each])

    def init_with_data(self, body_joints_data):
        # body_joints_data shape: [frames][17 joints][x, y]
        if isinstance(body_joints_data, list):
            body_joints_data = np.array(body_joints_data, dtype=float)
        self.body_joints = {
            i: Point(
                point_idx=i,
                fps=self.calibrationHelper["fps"],
                data=body_joints_data[:, i],
            )
            for i in range(17)
        }

    def append_empty_frames(self, frames=1):
        for each, body_joint in self.body_joints.items():
            body_joint.append_empty_frames(frames=frames)

    # EXPORT
    def export(
        self,
        save_dirs=None,
        mode=None,
        frequency=30,
        metrics_to_export=list(PointMetrics),
        meter_per_pixel=None,
    ):
        human_dir = Path(save_dirs) / f"human_{self.human_idx}"
        for idx, (each, body_joint) in enumerate(self.body_joints.items()):
            body_joint.export(
                human_dir=human_dir,
                mode=mode,
                frequency=frequency,
                meter_per_pixel=meter_per_pixel,
                metrics_to_export=metrics_to_export,
            )

        for each, customRecord in self.customRecord.items():
            customRecord.export(
                human_dir=human_dir,
                mode=mode,
                frequency=frequency,
                meter_per_pixel=meter_per_pixel,
                metrics_to_export=metrics_to_export,
            )

    def export_metrics(self, dir):
        if self.human_idx:
            name = f"{self.human_idx}_{self.name}"
        else:
            name = self.name
        # export json
        json_output = self.get_metrics()
        json_file = Path(dir) / get_time()
        json_file = json_file / f"{name}.json"
        json_file.parent.mkdir(parents=True, exist_ok=True)
        with open(json_file, "w") as f:
            json.dump(json_output, f, indent=2)

    def export_csv(self, dir):
        # let's turn the data into csv file
        if self.human_idx:
            name = f"{self.human_idx}_{self.name}"
        else:
            name = self.name
        # First, let's get the metrics
        raw_result = self.get_metrics_pd()
        df = pd.DataFrame(raw_result)
        csv_file = Path(dir) / get_time()
        csv_file = csv / f"{name}.csv"
        csv_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_file)

    def get_metrics_flatten(self):
        # First, let's get the metrics
        metrics = self.get_metrics()
        # We need to flatten the metrics
        result = {}
        # First, let's get the metrics_info
        body_joints_metrics = metrics["body_joints_metrics"]

        for body, metric_data in body_joints_metrics.items():
            for metric_name, metric in metric_data["metrics"].items():
                result[f"{body}_{metric_name}"] = metric

        custom_metrics = metrics["custom_metrics"]
        for body_part, metric_data in custom_metrics.items():
            for metric_name, metric in metric_data.items():
                result[f"{body_part}_{metric_name}"] = metric

        result["time"] = metrics["time"]
        return result
