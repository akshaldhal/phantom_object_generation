# Bench2Drive Phantom Objects

Tools for generating phantom objects and recording LiDAR data in the [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive) CARLA simulation dataset.

## Installation

```bash
# From GitHub
pip install git+https://github.com/<user>/carla_trajectories.git

# Local development
git clone <repo-url>
cd carla_trajectories
pip install .
```

> **Note:** CARLA (`carla>=0.9.16`) must be available in your Python environment. You may need to install it separately depending on your CARLA server version.

## Python API

```python
from bench2drive_phantom_objects import (
    LidarRecorder,
    laz_to_obj,
    download_bech2drive_dataset,
    DatasetSize,
)

# Download the mini dataset
download_bech2drive_dataset("data/Bench2Drive", DatasetSize.MINI)

# Record LiDAR with persistent phantom objects
recorder = LidarRecorder(
    dataset_path="data/Bench2Drive-mini/",
    spawn_frequency=(3, 8),
    spawn_range=(5, 20),
    spawn_persist_time_range=(10, 30),
    allowed_objects=["vehicle.tesla.model3", "static.prop.trafficcone01"],
)
recorder.run()

# Convert a point cloud
laz_to_obj("scan.laz", "output.obj")
```

## Available Phantom Object Blueprints

| Category | Blueprints |
|----------|-----------|
| **Static props** | `static.prop.trafficcone01`, `static.prop.trafficwarning`, `static.prop.streetbarrier`, `static.prop.constructioncone`, `static.prop.bin`, `static.prop.fountain`, `static.prop.kiosk_01` |
| **Vehicles** | `vehicle.tesla.model3`, `vehicle.audi.a2`, `vehicle.audi.etron`, `vehicle.bmw.grandtourer`, `vehicle.chevrolet.impala`, `vehicle.dodge.charger_2020`, `vehicle.ford.mustang`, `vehicle.jeep.wrangler_rubicon`, `vehicle.lincoln.mkz_2020`, `vehicle.mercedes.coupe`, `vehicle.mini.cooper_s`, `vehicle.nissan.micra`, `vehicle.nissan.patrol`, `vehicle.toyota.prius`, `vehicle.volkswagen.t2` |

## License

MIT
