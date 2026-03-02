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
  blueprint='static.prop.trafficcone01',
  rotation_angle=45,
  spawn_distance_from_ego=5,
  global_position_fixed=False,
  position_jitter=0.0,
)

dataset_builder = DatasetBuilder(
  perturbation=perturbation,
  mask = build_mask(PerturbationPattern.IN_AND_OUT, 32),
  source_dataset_path=data_dir + "/Bench2Drive-mini",
  fixed_delta_seconds = 0.1,
  output_path='./data/recorded-lidar',
  lidar_config=LidarConfig(),
  class_blueprint_map=CLASS_BLUEPRINT_MAP,
  mesh_id_map=MESH_ID_MAP,
  frame_split=32,
  verbose=True
  # bug fixes, tick fix, datset instance filtering
  # generate mini subsest of spoofed data samples
  # have a kind of random build mask function
  # lenght of perturbed history should be varible
  # review meeting notes
)

dataset_builder.build_dataset()