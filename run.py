# for each sample randomly sasmple 1 variation
# 
# json to record metadata
# 
# run on the mini dataset
# 
# make distances a range, np.random.sample, similar for angles


import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from bench2drive_phantom_objects import DatasetBuilder, LidarConfig, build_mask
from bench2drive_phantom_objects.dataset_builder import Perturbation, PerturbationPattern

SOURCE_DATASET_PATH: str = "data/Bench2Drive-mini/"
BASE_OUTPUT_PATH: str    = "./data/recorded-lidar"
HOST: str                = "127.0.0.1"
PORT: int                = 2000

RANDOM_SEED: int = 42

WAYMO_LIDAR_CONFIG = LidarConfig(
    channels=64,
    points_per_second=1_700_000,
    rotation_frequency=10,
    range=75,
    upper_fov=2,
    lower_fov=-17,
)

FIXED_DELTA_SECONDS: float = 1.0 / WAYMO_LIDAR_CONFIG.rotation_frequency
N_FRAMES: int = 32

BLUEPRINT_TOKEN: str = "MercedesCCC"
BLUEPRINT: str       = "vehicle.mercedes.coupe_2020"

DISTANCE_RANGE: tuple[float, float] = (1.0, 7.0)
ANGLE_RANGE:    tuple[float, float] = (0.0, 360.0)

MASK_PATTERNS: list[PerturbationPattern] = [
    PerturbationPattern.CONSTANT,
    PerturbationPattern.CLEAN_HISTORY,
    PerturbationPattern.CLEAN_OBSERVATION,
    PerturbationPattern.NOISY_MIDDLE,
    PerturbationPattern.IN_AND_OUT,
]

MASK_NAMES: dict[PerturbationPattern, str] = {
    PerturbationPattern.CONSTANT:           "constant",
    PerturbationPattern.CLEAN_HISTORY:      "clean_history",
    PerturbationPattern.CLEAN_OBSERVATION:  "clean_observation",
    PerturbationPattern.NOISY_MIDDLE:       "noisy_middle",
    PerturbationPattern.IN_AND_OUT:         "in_and_out",
}

JITTER_SETS: list[tuple[str, Optional[float]]] = [
    ("no_jitter",   0.0),
    ("jitter_0p5m", 0.5),
    ("jitter_rnd",  None),
]

POSITION_MODES: list[tuple[str, bool]] = [
    ("ego_relative", False),
    ("world_fixed",  True),
]

@dataclass
class RunConfig:
    instance_name:         str
    blueprint_token:       str
    blueprint:             str
    distance:              float
    angle:                 float
    pattern:               PerturbationPattern
    jitter_token:          str
    jitter:                float
    position_token:        str
    global_position_fixed: bool

    def to_metadata(self) -> dict:
        return {
            "instance":              self.instance_name,
            "blueprint_token":       self.blueprint_token,
            "blueprint":             self.blueprint,
            "distance_m":            round(self.distance, 4),
            "angle_deg":             round(self.angle, 4),
            "mask_pattern":          MASK_NAMES[self.pattern],
            "jitter_token":          self.jitter_token,
            "jitter_m":              round(self.jitter, 4),
            "position_token":        self.position_token,
            "global_position_fixed": self.global_position_fixed,
        }


def _resolve_jitter(jitter_val: Optional[float]) -> float:
    if jitter_val is None:
        return round(float(np.random.uniform(0.25, 0.75)), 4)
    return jitter_val


def build_configs(instances: list[Path]) -> list[RunConfig]:
    configs = []
    for inst in instances:
        configs.append(RunConfig(
            instance_name=inst.name,
            blueprint_token=BLUEPRINT_TOKEN,
            blueprint=BLUEPRINT,
            distance=float(np.random.uniform(*DISTANCE_RANGE)),
            angle=float(np.random.uniform(*ANGLE_RANGE)),
            pattern=random.choice(MASK_PATTERNS),
            jitter_token=(j := random.choice(JITTER_SETS))[0],
            jitter=_resolve_jitter(j[1]),
            position_token=(pm := random.choice(POSITION_MODES))[0],
            global_position_fixed=pm[1],
        ))
    return configs


def run(start_from: int = 0) -> None:
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    instances = sorted([d for d in Path(SOURCE_DATASET_PATH).iterdir() if d.is_dir()])
    configs   = build_configs(instances)
    total     = len(configs)

    output_root = Path(BASE_OUTPUT_PATH)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  LiDAR sweep  |  {total} instances  |  seed {RANDOM_SEED}")
    print(f"  {BLUEPRINT_TOKEN} ({BLUEPRINT})")
    print(f"  dist {DISTANCE_RANGE}m  angle {ANGLE_RANGE}°")
    print(f"  output → {output_root}")
    print(f"{'=' * 60}\n")

    # Write sweep-level metadata
    with open(output_root / "sweep_metadata.json", "w") as f:
        json.dump({"random_seed": RANDOM_SEED, "n_frames": N_FRAMES,
                   "distance_range": DISTANCE_RANGE, "angle_range": ANGLE_RANGE,
                   "blueprint": BLUEPRINT,
                   "configs": [c.to_metadata() for c in configs]}, f, indent=2)

    completed = 0
    for i, cfg in enumerate(configs[start_from:], start=start_from + 1):
        meta = cfg.to_metadata()
        print(f"[{i:03d}/{total}] {cfg.instance_name}  "
              f"dist={meta['distance_m']}m  angle={meta['angle_deg']}°  "
              f"pattern={meta['mask_pattern']}  jitter={meta['jitter_m']}m  "
              f"pos={cfg.position_token}")

        perturbation = Perturbation(
            blueprint=cfg.blueprint,
            rotation_angle=cfg.angle,
            spawn_distance_from_ego=cfg.distance,
            global_position_fixed=cfg.global_position_fixed,
            position_jitter=cfg.jitter,
        )

        builder = DatasetBuilder(
            perturbation=perturbation,
            mask=build_mask(cfg.pattern, N_FRAMES),
            source_dataset_path=SOURCE_DATASET_PATH,
            fixed_delta_seconds=FIXED_DELTA_SECONDS,
            # output_root directly — DatasetBuilder appends instance_name inside
            output_path=str(output_root),
            instance_filter=[cfg.instance_name],
            host=HOST,
            port=PORT,
            lidar_config=WAYMO_LIDAR_CONFIG,
        )

        try:
            builder.build_dataset()
            completed += 1

        except KeyboardInterrupt:
            print(f"\nInterrupted at {i}/{total}. Resume with start_from={i-1}")
            sys.exit(0)
        except Exception as e:
            print(f"  ERROR {cfg.instance_name}: {e}")
            import traceback; traceback.print_exc()

    print(f"\nDone. {completed}/{total - start_from} succeeded -> {output_root}\n")


if __name__ == "__main__":
    run(start_from=0)