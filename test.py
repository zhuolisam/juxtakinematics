from kinematics import kinematics
from src.kinematics import ops

profiler_kinematics = ops.Profile()  # count the time

with profiler_kinematics:
    kinematics = kinematics.Kinematics(run_path="data/bike/bike.csv")

    kinematics(
        save=True,
        overwrite=True,
        associate_human_profile=[],
        onlyHumanProfile=[1, 2, 3],
        filter_on=True,
        interpolate_on=True,
        smoothing_on=True,
        preprocess_smoothing_on=True,
    )
print(f"Project saved, took {profiler_kinematics.dt * 1E3:.1f}ms")
