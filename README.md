# CARLA Trajectories

Tools for phantom object generation in bench2drive dataset.

## Installation

```bash
pip install carla-trajectories
```

## Quick Start

```python
from carla_trajectories import download_bech2drive_dataset, LidarRecorder, DatasetSize

# Download dataset
download_bech2drive_dataset("data/Bench2Drive", DatasetSize.MINI)

# Record LiDAR scans
recorder = LidarRecorder(
    dataset_path='data/Bench2Drive-mini/',
    spawn_frequency=(2, 5),  # Random objects per frame
    spawn_range=(10, 30)     # Distance from ego vehicle
)
recorder.run()
```

## Features
