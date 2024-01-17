import csv
import json
from io import StringIO
from pathlib import Path
from typing import Any, Literal

from .human_profile import HumanProfile

AngleMetrics = ["Angular Degree", "Angular Velocity"]


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


class Kinematics(Record):
    def __init__(
        self,
        num_frames=None,
        save_dirs=None,
        results=None,
        run_path=None,
    ) -> None:
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
                    num_frames=self.num_frames, human_idx=profile_id
                )
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
                [self.profileSet[i] for i in each]
            )
            for toRemove in each[1:]:
                self.profileSet.pop(toRemove)

    def compute(
        self,
        angle=None,
        filter_on=None,
        interpolate_on=None,
        smoothing_on=None,
        preprocess_smoothing_on=None,
    ):
        for each, profile in self.profileSet.items():
            profile.compute(
                filter_on=filter_on,
                interpolate_on=interpolate_on,
                smoothing_on=smoothing_on,
                preprocess_smoothing_on=preprocess_smoothing_on,
            )

    def export(
        self,
        mode: Literal["compact, complete"],
        save_dirs=None,
        overwrite=False,
        meter_per_pixel=None,
    ) -> None:
        save_dirs = Path(self.source_path).parent if save_dirs is None else save_dirs
        for each, profile in self.profileSet.items():
            profile.export(
                save_dirs=save_dirs, mode=mode, meter_per_pixel=meter_per_pixel
            )

    def export_metrics(self, save_dirs=None) -> None:
        data = {
            "metadata": {
                "num_frames": self.num_frames,
                "calibrationHelper": self.calibrationHelper,
            },
            "human_profiles": [each.get_metrics() for each in self.profileSet.values()],
        }
        # save the json
        save_dirs = Path(self.source_path).parent if save_dirs is None else save_dirs
        json_file = Path(save_dirs) / "metrics.json"
        with open(json_file, "w") as f:
            json.dump(data, f, indent=4)
        return data

    def __call__(
        self,
        save=True,
        save_dirs=None,
        overwrite=True,
        mode="compact",
        meter_per_pixel=None,
        associate_human_profile=[],
        onlyHumanProfile=[],
        filter_on=None,
        interpolate_on=None,
        smoothing_on=None,
        preprocess_smoothing_on=None,
    ) -> Any:
        # associate human profile
        if len(associate_human_profile) > 0 and len(associate_human_profile[0]) > 0:
            self.associate_human_profile(associate_human_profile)

        if len(onlyHumanProfile) > 0:
            self.profileSet = {each: self.profileSet[each] for each in onlyHumanProfile}

        # compute
        self.compute(
            filter_on=filter_on,
            interpolate_on=interpolate_on,
            smoothing_on=smoothing_on,
            preprocess_smoothing_on=preprocess_smoothing_on,
        )

        if save:
            # self.export(save_dirs=save_dirs, overwrite=overwrite,
            #             mode=mode, meter_per_pixel=meter_per_pixel)
            self.export_metrics(save_dirs=save_dirs)
