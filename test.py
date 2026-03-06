import os
from bench2drive_phantom_objects import (
  download_bech2drive_dataset,
  DatasetBuilder,
  LidarConfig,
  build_mask
)
from bench2drive_phantom_objects.dataset_builder import (
  Perturbation,
  PerturbationPattern,
  CLASS_BLUEPRINT_MAP,
  MESH_ID_MAP,
)

data_dir = "./data"
if not os.path.exists(data_dir):
  download_bech2drive_dataset()

perturbation = Perturbation(
  blueprint='static.prop.trafficcone01', # change to cars for now
  rotation_angle=45,
  spawn_distance_from_ego=5,
  global_position_fixed=False,
  position_jitter=0.0,
)

dataset_builder = DatasetBuilder(
  perturbation=perturbation,
  mask = build_mask(PerturbationPattern.IN_AND_OUT, 32),
  source_dataset_path=data_dir + "/Bench2Drive-mini",
  fixed_delta_seconds = 0.1, # note: in the current implementations, the physuics are tunred off, so the only thing this really 
                             # impacts is the density of lidar scans (along with frequency etc in lidar config)
  output_path='./data/recorded-lidar',
  lidar_config=LidarConfig(),
  class_blueprint_map=CLASS_BLUEPRINT_MAP,
  mesh_id_map=MESH_ID_MAP,
  frame_split=32,
  verbose=True
  # fix: Perturbaiton does not pivot with the car with global: false setting
  # 
  # IMP MODIFICATION:  we are not splitting dataset ionto frames, rather we're applying windows of perturbed or clean of size n (32 in this casee) onto the entire thing, need to be 20-30 second each sample
  # 
  # feat: implement an improved laz_utils with some sanity checks etc
  # feat: align default configs with the waymo open
  # 
  # todo: fix readme with proper documentation
  # 
  # feat: impliment a bench2drive to waymo pipline, can be a data to data pipeline
  # feat: check coordinates, global vs ego-vehicle coordinate space, which is applied to perturbations, stick with the waymo format
)

dataset_builder.build_dataset()

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