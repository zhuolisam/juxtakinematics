import csv
import json
import uuid
from pathlib import Path
from typing import Any, List

import numpy as np

from .angle import Angle
from .point import Point, PointMetrics

# HumanProfile is always initialized with body_joints of 17 of empty data.
# Use setup_from_csv to populate the data.
# setup_from_csv can either populate the body_joints again or just append the data to the existing body_joints (call different function in Point)
# Which API shall Kinematics use? So it's the same, Kinematics will create a HumanProfile with empty body_joints, then call process_frame to populate the data. In such case, Kinematics can only use the append function in Point, which is not efficient.
# Hmm, are there any ways to make this efficient? Abo create HumanProfile after accumulate. {body_joint: (num_frames,17,2)}. Then pass the whole shit.


# So I would like Point class to also either initialize with all datas, or initialize with totally empty data.


class HumanProfile:
    def __init__(
        self,
        human_idx=None,
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
        if not body_format or body_format == "coco":
            self.body_joints = {
                i: Point(
                    point_idx=i,
                    x_meter_per_pixel=self.calibrationHelper["x_meter_per_pixel"],
                    y_meter_per_pixel=self.calibrationHelper["y_meter_per_pixel"],
                    fps=self.calibrationHelper["fps"],
                    num_frames=1,
                )
                for i in range(17)
            }

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

        self.customRecord["Left Angle"] = Angle(
            angle_name="Left Angle",
            dependencies=[
                self.body_joints[11],
                self.body_joints[13],
                self.body_joints[15],
            ],
        )

        self.customRecord["Left Angle"] = Angle(
            angle_name="Right Angle",
            dependencies=[
                self.body_joints[12],
                self.body_joints[14],
                self.body_joints[16],
            ],
        )
        # TODO
        # 'center_of_mass' : Point()

        for each, customRecord in self.customRecord.items():
            customRecord.preprocess()
            customRecord.compute()

    @property
    def get_num_frames(self):
        n = len(self.body_joints[0].data["Raw Ori X"])
        assert n == len(
            self.body_joints[0].data["Raw Ori Y"]
        ), "Inconsistent data length"
        return n

    def associate_human_profile(self, associate_human: List["HumanProfile"]):
        assert self == associate_human[0]
        for frame in range(self.get_num_frames):
            if self.body_joints[0].data["Raw Ori X"][frame]:
                pass
            else:
                for each in associate_human[1:]:
                    if each[frame] is not None:
                        for each_joint in self.body_joints:
                            self.body_joints[each_joint].raw_ori_x[
                                frame
                            ] = each.body_joints[each_joint].raw_ori_x[frame]
                            self.body_joints[each_joint].raw_ori_y[
                                frame
                            ] = each.body_joints[each_joint].raw_ori_y[frame]
                        break
                    else:
                        continue
        return self

    def get_data(self):
        data = {}
        for each, body_joint in self.body_joints.items():
            data[each] = {key: item.tolist() for key, item in body_joint.data}

    def get_metrics(self):
        data = {
            "metadata": {
                "x_meter_per_pixel": self.calibrationHelper["x_meter_per_pixel"],
                "y_meter_per_pixel": self.calibrationHelper["y_meter_per_pixel"],
                "fps": self.calibrationHelper["fps"],
                "human_idx": self.human_idx,
                "name": self.name,
            },
            "body_joints": {
                key: joints.get_metrics() for key, joints in self.body_joints.items()
            },
            "custom_metrics": {
                key: metrics.get_metrics() for key, metrics in self.customRecord.items()
            },
            "time": [
                i / self.calibrationHelper["fps"] for i in range(self.get_num_frames)
            ],
        }
        if None in data["body_joints"]:
            del data["body_joint"][None]
        if None in data["custom_metrics"]:
            del data["custom_metrics"][None]
        return data

    def setup_from_csv(self, run_csv):
        if isinstance(run_csv, str):
            self.source_path = Path(run_csv)
            with open(run_csv, "r") as csv_file:
                csv_reader = csv.reader(csv_file)

                csv_list = list(csv_reader)

                accumulated_data = []  # Concat data from all frames
                for row in csv_list:
                    frame_number, frame_data = row
                    frame_number = int(frame_number)
                    # Parse the JSON string
                    frame_data = json.loads(frame_data)

                    # METHOD 1: Use append API
                    processed_frame_data = self.process_frame(frame_data, 1)

                    # METHOD 2: Collect and init ( I prefer this method)
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
            self.process_profile(frame_data[human_idx])
            return frame_data[human_idx]
        else:
            self._appendEmptyPoints()
            return [[None, None] for _ in range(len(self.body_joints))]

    def process_profile_and_append(self, points: list[list[float]]) -> None:
        assert (
            len(points) == len(self.body_joints)
        ), (
            f"Expected {len(self.body_joints)} points, got {len(points)}"
        )  # BAD: This function assumes self.body_joints is already initialized

        for each, body_joint in self.body_joints.items():
            body_joint.append_points(points[each])

    def init_with_data(self, body_joints_data):
        self.body_joints = {
            i: Point(
                point_idx=i,
                x_meter_per_pixel=self.calibrationHelper["x_meter_per_pixel"],
                y_meter_per_pixel=self.calibrationHelper["y_meter_per_pixel"],
                fps=self.calibrationHelper["fps"],
                data=body_joints_data[:, i],
            )
            for i in range(17)
        }

    def _appendEmptyPoints(
        self,
    ):
        for each, body_joint in self.body_joints.items():
            body_joint.appendEmptyPoints()

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

    def export_metrics(self, save_dirs=None):
        data = self.get_metrics()
        # save the json
        human_dir = Path(save_dirs) / f"human_{self.human_idx}"
        json_file = Path(human_dir) / "metrics.json"
        with open(json_file, "w") as f:
            json.dump(data, f, indent=4)
