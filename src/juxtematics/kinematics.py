import csv
import json
from io import StringIO
from pathlib import Path
from typing import Any, Literal

from .human_profile import HumanProfile
from .utils import get_time


class Kinematics:
    def __init__(
        self,
        num_frames=None,
        save_dirs=None,
        raw_data=None,
        run_path=None,
        run_path_v2=True,
        x_meter_per_pixel=None,
        y_meter_per_pixel=None,
        fps=None,
    ) -> None:
        super().__init__()
        self.frequency = None
        self.profileSet: dict[str, HumanProfile] = {}
        self.source_path = None
        self.save_dirs = None
        self.num_frames = None
        self.calibrationHelper = None

        if raw_data is not None:
            self.setup(raw_data)
        elif run_path is not None and run_path_v2:
            self.setup_from_csv_v2(run_path)
        elif run_path is not None and not run_path_v2:
            self.setup_from_csv(run_path)

    def setup_from_csv(self, run_csv) -> None:
        if isinstance(run_csv, str):
            self.source_path = Path(run_csv)
            with open(run_csv, "r") as csv_file:
                csv_reader = csv.reader(csv_file)

                csv_list = list(csv_reader)
                accumulated_data = []  # Concat data from all frames
                for row in csv_list:
                    _, frame_data = row
                    frame_data = json.loads(frame_data)
                    accumulated_data.append(frame_data)

            # Broadcast accumulated data to each human profile
            for frame_idx, frame_data in enumerate(accumulated_data):
                # loop through each person in the frame
                for human_idx in frame_data:
                    if human_idx == "":
                        continue

                    # if the person does not exists, create a new one, and append idx empty frames, then process the frame
                    if human_idx not in self.profileSet:
                        self.profileSet[human_idx] = HumanProfile(human_idx=human_idx)
                        self.profileSet[human_idx].append_empty_frames(frames=frame_idx)
                        self.profileSet[human_idx].process_frame_and_append(
                            frame_data, human_idx=human_idx
                        )
                    # else if the person exists, just pass the whole frame into it, using human.process_frame(frame_data, human_idx=idx). Every human needs this to append empty frames
                    else:
                        self.profileSet[human_idx].process_frame_and_append(
                            frame_data, human_idx=human_idx
                        )
        else:
            raise TypeError("run_csv should be a string")

    def setup_from_csv_v2(self, run_csv):
        if isinstance(run_csv, str):
            self.source_path = Path(run_csv)
            with open(run_csv, "r") as csv_file:
                csv_reader = csv.reader(csv_file)

                csv_list = list(csv_reader)
                accumulated_data = []  # Concat data from all frames
                for row in csv_list:
                    _, frame_data = row
                    frame_data = json.loads(frame_data)
                    accumulated_data.append(frame_data)

                    # Keep Track of the human_idx and the frame data
                    for human_idx in frame_data:
                        if human_idx not in self.profileSet and human_idx != "":
                            self.profileSet[human_idx] = HumanProfile(
                                human_idx=human_idx
                            )

                    # Loop through each person, and pass all accumulated data to each person
                for human_idx, human in self.profileSet.items():
                    human.setup_from_data(accumulated_data, human_idx=human_idx)

    def setup_from_data(self, data):
        # This one, we already have everything formatted, as {human_idx: [frame_data * 17 * 2]}
        # Init three human profiles, and pass their respective data. Should use init_with_data function
        return

    def setup_from_stringio(self, run_csv) -> None:
        if isinstance(run_csv, StringIO):
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
        else:
            raise TypeError("run_csv should be a StringIO")

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
        filter_threshold=None,
        preprocess_interpolate_on=False,
        preprocess_filter_on=False,
        preprocess_smoothing_on=False,
        postcalculate_filter_on=False,
        postcalculate_smoothing_on=False,
    ):
        for each, profile in self.profileSet.items():
            profile.compute(
                filter_threshold=filter_threshold,
                preprocess_interpolate_on=preprocess_interpolate_on,
                preprocess_filter_on=preprocess_filter_on,
                preprocess_smoothing_on=preprocess_smoothing_on,
                postcalculate_filter_on=postcalculate_filter_on,
                postcalculate_smoothing_on=postcalculate_smoothing_on,
            )

    def get_metrics(self):
        return {
            "metadata": {
                "num_frames": self.num_frames,
                "calibrationHelper": self.calibrationHelper,
            },
            "human_profiles": [each.get_metrics() for each in self.profileSet.values()],
        }

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
        data = self.get_metrics()
        # save the json
        save_dirs = Path(self.source_path).parent if save_dirs is None else save_dirs
        json_file = Path(save_dirs) / get_time()
        json_file = json_file / "metrics.json"
        json_file.parent.mkdir(parents=True, exist_ok=True)
        with open(json_file, "w") as f:
            json.dump(data, f, indent=2)
        return

    def __call__(
        self,
        save=True,
        save_dirs=None,
        overwrite=True,
        mode="compact",
        meter_per_pixel=None,
        associate_human_profile=[],
        onlyHumanProfile=[],
        preprocess_interpolate_on=False,
        preprocess_filter_on=False,
        preprocess_smoothing_on=False,
        postcalculate_filter_on=False,
        postcalculate_smoothing_on=False,
    ) -> Any:
        # associate human profile
        if len(associate_human_profile) > 0 and len(associate_human_profile[0]) > 0:
            self.associate_human_profile(associate_human_profile)

        if len(onlyHumanProfile) > 0:
            print(self.profileSet.keys())
            self.profileSet = {
                each: self.profileSet[str(each)] for each in onlyHumanProfile
            }

        # compute
        self.compute(
            preprocess_interpolate_on=preprocess_interpolate_on,
            preprocess_filter_on=preprocess_filter_on,
            preprocess_smoothing_on=preprocess_smoothing_on,
            postcalculate_filter_on=postcalculate_filter_on,
            postcalculate_smoothing_on=postcalculate_smoothing_on,
        )

        if save:
            # self.export(save_dirs=save_dirs, overwrite=overwrite,
            #             mode=mode, meter_per_pixel=meter_per_pixel)
            self.export_metrics(save_dirs=save_dirs)

    # def _setup_from_csv_deprecated(self, run_csv) -> None:
    #     # Get the file name without extension using pathlib library
    #     if isinstance(run_csv, str):
    #         self.source_path = Path(run_csv)
    #         with open(run_csv, "r") as csv_file:
    #             csv_reader = csv.reader(csv_file)

    #             csv_list = list(csv_reader)
    #             self.num_frames = len(csv_list)
    #             for row in csv_list:
    #                 frame_number, keypoints_data = row
    #                 frame_number = int(frame_number)
    #                 # Parse the JSON string
    #                 keypoints_data = json.loads(keypoints_data)
    #                 self.process_frame(frame_number - 1, keypoints_data)
    #     else:
    #         raise TypeError("run_csv should be a string")

    # def process_frame(self, frame_number, keypoints_data):
    #     # make sure correct dimension
    #     for profile_id, profile_points in keypoints_data.items():
    #         if profile_id == "":  # cuz there's empty datapoints
    #             continue
    #         profile_id = int(profile_id)  # Convert profile_id to float
    #         if profile_id not in self.profileSet:
    #             self.profileSet[profile_id] = HumanProfile(
    #                 num_frames=self.num_frames, human_idx=profile_id
    #             )
    #         self.profileSet[profile_id].appendFrame(
    #             profile_points, frame_no=frame_number
    #         )
