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

## CLI Usage

After installation, the `bench2drive-phantom` command is available:

### Download Dataset

```bash
bench2drive-phantom download --size mini --dir data/Bench2Drive
```

| Flag | Default | Description |
|------|---------|-------------|
| `--size` | `mini` | Dataset size: `mini`, `base`, or `full` |
| `--dir` | `data/Bench2Drive` | Target download directory |

### Record LiDAR Scans

```bash
bench2drive-phantom record \
  --dataset-path data/Bench2Drive-mini \
  --output-path data/recorded-lidar \
  --spawn-min 3 --spawn-max 8 \
  --spawn-persist-min 10 --spawn-persist-max 30 \
  --allowed-objects vehicle.tesla.model3 static.prop.trafficcone01
```

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset-path` | `data/Bench2Drive-mini/` | Path to Bench2Drive dataset |
| `--output-path` | `./data/recorded-lidar` | Output directory for scans |
| `--host` / `--port` | `127.0.0.1` / `2000` | CARLA server address |
| `--channels` | `64` | LiDAR channels |
| `--spawn-min` / `--spawn-max` | `5` / `10` | Random object count range |
| `--spawn-range-min` / `--spawn-range-max` | `5.0` / `10.0` | Spawn distance range (m) |
| `--spawn-persist-min` / `--spawn-persist-max` | `10` / `20` | Frame persistence range |
| `--spawn-rotation-min` / `--spawn-rotation-max` | `-180` / `180` | Spawn rotation range (°) |
| `--allowed-objects` | all | Space-separated list of allowed blueprint IDs |
| `--verbose` | off | Enable verbose output |

### Convert LAZ to OBJ

```bash
bench2drive-phantom convert scan.laz -o output.obj --color-by height --colormap rainbow
```

| Flag | Default | Description |
|------|---------|-------------|
| `input` | *(required)* | Input `.laz` file or directory |
| `-o` / `--output` | `output.obj` | Output `.obj` file path |
| `--color-by` | `height` | Coloring: `height`, `intensity`, `file`, `none` |
| `--colormap` | `rainbow` | Color scheme: `gray`, `hot`, `viridis`, `jet`, `rainbow`, `terrain`, `ocean` |
| `--subsample` | none | Max points to keep |

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
