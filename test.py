from src.kinematics import kinematics_v2, ops

profiler_kinematics = ops.Profile()  # count the time

with profiler_kinematics:
    # kinematics = kinematics.Kinematics(run_path="20230929-022531/Men 100m Heat-1.csv")
    # kinematics = kinematics_v2.Kinematics(run_path = "bike_short/bike_shortt.csv")
    kinematics = kinematics_v2.Kinematics(run_path="data/bike/bike.csv")

    kinematics(save=True, overwrite=True, associate_human_profile=[],
               onlyHumanProfile=[1, 2, 3], filter_on=True, interpolate_on=True, smoothing_on=True, preprocess_smoothing_on=True)
print(f"Project saved, took {profiler_kinematics.dt * 1E3:.1f}ms")
