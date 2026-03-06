import os

from bench2drive_phantom_objects import (
    DatasetBuilder,
    LidarConfig,
    build_mask,
    download_bech2drive_dataset,
)
from bench2drive_phantom_objects.dataset_builder import (
    CLASS_BLUEPRINT_MAP,
    MESH_ID_MAP,
    Perturbation,
    PerturbationPattern,
)

DATA_DIR = "./data"
OUTPUT_DIR = "./data/recorded-lidar-demo"


def demo_pipeline():
    if not os.path.exists(DATA_DIR):
        download_bech2drive_dataset()

    perturbation = Perturbation(
        blueprint="static.prop.trafficcone01",
        rotation_angle=0,
        spawn_distance_from_ego=5,
        global_position_fixed=False,
        position_jitter=0.0,
    )

    mask = build_mask(PerturbationPattern.IN_AND_OUT, 32)

    builder = DatasetBuilder(
        perturbation=perturbation,
        mask=mask,
        source_dataset_path=DATA_DIR + "./Bench2Drive-mini",
        fixed_delta_seconds=0.05,
        output_path=OUTPUT_DIR,
        lidar_config=LidarConfig(),
        class_blueprint_map=CLASS_BLUEPRINT_MAP,
        mesh_id_map=MESH_ID_MAP,
        instance_filter=None,
        verbose=True,
    )

    builder.build_dataset()


if __name__ == "__main__":
    demo_pipeline()

# feat: implement an improved laz_utils with some sanity checks etc
# feat: align default configs with the waymo open
#
# todo: fix readme with proper documentation
#
# feat: impliment a bench2drive to waymo pipline, can be a data to data pipeline

"""
The core constraint is this relationship:

fixed_delta_seconds = 1 / rotation_frequency

This is the hard floor — go below it and the LiDAR won't complete a full 360° rotation per tick, giving you incomplete scans.
To minimize delta (maximize tick rate):

rotation_frequency = target_hz          # e.g. 50 for 20ms delta
fixed_delta_seconds = 1 / rotation_frequency
points_per_scan = points_per_second / rotation_frequency

So if you want consistent point density as you increase frequency, scale points_per_second up proportionally:
points_per_second = desired_points_per_scan * rotation_frequency

"""
