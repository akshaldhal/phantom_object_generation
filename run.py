import itertools
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from bench2drive_phantom_objects import DatasetBuilder, LidarConfig, build_mask
from bench2drive_phantom_objects.dataset_builder import Perturbation, PerturbationPattern

SOURCE_DATASET_PATH: str = "data/Bench2Drive-mini/"
BASE_OUTPUT_PATH: str    = "./data/recorded-lidar"
HOST: str                = "127.0.0.1"
PORT: int                = 2000

RANDOM_SEED: int = 42

# needs to be checked, already got what i could from the paper
WAYMO_LIDAR_CONFIG = LidarConfig(
    channels=32,
    points_per_second=2_240_000,
    rotation_frequency=10,
    range=75,
    upper_fov=2,
    lower_fov=-17,
)

FIXED_DELTA_SECONDS: float = 1.0 / WAYMO_LIDAR_CONFIG.rotation_frequency  # 0.1 s

# FIX ME
N_FRAMES: int = 200

BLUEPRINTS: dict[str, str] = {
    "lincoln":  "vehicle.lincoln.mkz_2020",
    "tesla":    "vehicle.tesla.model3",
    "dodge":    "vehicle.dodge.charger_2020",
    "ford":     "vehicle.ford.crown",
    "mercedes": "vehicle.mercedes.coupe_2020",
}

DISTANCES: list[int] = [5, 10, 20]  # metres from ego

# 8 compass directions in 45° steps
# 0°=ahead  45°=front-left  90°=left  135°=rear-left
# 180°=rear 225°=rear-right 270°=right 315°=front-right
ANGLES: list[int] = list(range(0, 360, 45))

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

# (directory token, fixed jitter value — None means sample randomly per config)
JITTER_SETS: list[tuple[str, Optional[float]]] = [
    ("no_jitter",   0.0),
    ("jitter_0p5m", 0.5),
    ("jitter_rnd",  None),   # actual value sampled from U[0.25, 0.75]
]

# (directory token, global_position_fixed value)
POSITION_MODES: list[tuple[str, bool]] = [
    ("ego_relative", False),  # object tracks ego
    ("world_fixed",  True),   # object pinned in world space
]

@dataclass
class RunConfig:
    blueprint_token:      str
    blueprint:            str
    distance:             int
    angle:                int
    pattern:              PerturbationPattern
    jitter_token:         str
    jitter:               float   # resolved (never None)
    position_token:       str
    global_position_fixed: bool

    @property
    def dir_name(self) -> str:
        return (
            f"{self.blueprint_token}"
            f"__{self.distance}m"
            f"__{self.angle}deg"
            f"__{MASK_NAMES[self.pattern]}"
            f"__{self.jitter_token}"
            f"__{self.position_token}"
        )

    def __str__(self) -> str:
        return self.dir_name


def _resolve_jitter(jitter_val: Optional[float]) -> float:
    """Resolve a jitter specification: None → sample from U[0.25, 0.75]."""
    if jitter_val is None:
        return round(random.uniform(0.25, 0.75), 4)
    return jitter_val


def build_all_configs() -> list[RunConfig]:
    configs: list[RunConfig] = []
    product = itertools.product(
        BLUEPRINTS.items(),
        DISTANCES,
        ANGLES,
        MASK_PATTERNS,
        JITTER_SETS,
        POSITION_MODES,
    )
    for (bp_token, bp), dist, angle, pattern, (jitter_token, jitter_val), (pos_token, gpf) in product:
        configs.append(RunConfig(
            blueprint_token=bp_token,
            blueprint=bp,
            distance=dist,
            angle=angle,
            pattern=pattern,
            jitter_token=jitter_token,
            jitter=_resolve_jitter(jitter_val),
            position_token=pos_token,
            global_position_fixed=gpf,
        ))
    return configs

def print_header(configs: list[RunConfig]) -> None:
    n_blueprints  = len(BLUEPRINTS)
    n_distances   = len(DISTANCES)
    n_angles      = len(ANGLES)
    n_masks       = len(MASK_PATTERNS)
    n_jitter      = len(JITTER_SETS)
    n_positions   = len(POSITION_MODES)

    print(f"\n{'=' * 72}")
    print("  Waymo-aligned LiDAR perturbation sweep — run_dataset.py")
    print(f"{'=' * 72}")
    print(f"  LiDAR channels       : {WAYMO_LIDAR_CONFIG.channels}")
    print(f"  LiDAR range          : {WAYMO_LIDAR_CONFIG.range} m  (Waymo Table 2)")
    print(f"  LiDAR FOV            : [{WAYMO_LIDAR_CONFIG.lower_fov}°, +{WAYMO_LIDAR_CONFIG.upper_fov}°]  (Waymo Table 2)")
    print(f"  Rotation frequency   : {WAYMO_LIDAR_CONFIG.rotation_frequency} Hz  (Waymo Sec 4.2)")
    print(f"  Points/sec           : {WAYMO_LIDAR_CONFIG.points_per_second:,}")
    print(f"  Delta seconds        : {FIXED_DELTA_SECONDS} s")
    print(f"  Frames per scene     : {N_FRAMES}  (20 s × 10 Hz, Waymo Sec 4.2)")
    print(f"{'=' * 72}")
    print(f"  Blueprints  ({n_blueprints:2d})     : {list(BLUEPRINTS.keys())}")
    print(f"  Distances   ({n_distances:2d})     : {DISTANCES} m")
    print(f"  Angles      ({n_angles:2d})     : {ANGLES}°  ({n_angles} directions)")
    print(f"  Mask patterns ({n_masks:2d})   : {[MASK_NAMES[p] for p in MASK_PATTERNS]}")
    print(f"  Jitter sets ({n_jitter:2d})     : {[t for t, _ in JITTER_SETS]}")
    print(f"  Position modes ({n_positions:2d})  : {[t for t, _ in POSITION_MODES]}")
    print(f"{'=' * 72}")
    print(f"  Total configurations : {len(configs)}"
          f"  ({n_blueprints} × {n_distances} × {n_angles} × {n_masks} × {n_jitter} × {n_positions})")
    print(f"  Random seed          : {RANDOM_SEED}")
    print(f"  Source               : {SOURCE_DATASET_PATH}")
    print(f"  Output root          : {BASE_OUTPUT_PATH}/perturbed/")
    print(f"{'=' * 72}\n")

def run(start_from: int = 0) -> None:
    configs = build_all_configs()
    total   = len(configs)

    print_header(configs)

    if start_from > 0:
        print(f"Resuming from config index {start_from} (0-based).\n")

    completed = 0
    for i, cfg in enumerate(configs[start_from:], start=start_from + 1):
        output_path = str(Path(BASE_OUTPUT_PATH) / "perturbed" / cfg.dir_name)

        print(f"\n[{i:04d}/{total}] {cfg.dir_name}")
        print(f"         blueprint       : {cfg.blueprint}")
        print(f"         distance        : {cfg.distance} m")
        print(f"         angle           : {cfg.angle}°")
        print(f"         pattern         : {MASK_NAMES[cfg.pattern]}")
        print(f"         jitter          : {cfg.jitter} m")
        print(f"         position mode   : {cfg.position_token}  (global_position_fixed={cfg.global_position_fixed})")
        print(f"         output          : {output_path}")

        perturbation = Perturbation(
            blueprint=cfg.blueprint,
            rotation_angle=cfg.angle,
            spawn_distance_from_ego=cfg.distance,
            global_position_fixed=cfg.global_position_fixed,
            position_jitter=cfg.jitter,
        )

        mask = build_mask(cfg.pattern, N_FRAMES)

        builder = DatasetBuilder(
            perturbation=perturbation,
            mask=mask,
            source_dataset_path=SOURCE_DATASET_PATH,
            fixed_delta_seconds=FIXED_DELTA_SECONDS,
            output_path=output_path,
            instance_filter=None,
            host=HOST,
            port=PORT,
            lidar_config=WAYMO_LIDAR_CONFIG,
        )

        try:
            builder.build_dataset()
            completed += 1
        except KeyboardInterrupt:
            print(f"\nInterrupted at config {i}/{total}.")
            print(f"To resume, set  start_from={i - 1}  in the run() call at the bottom of this file.")
            sys.exit(0)
        except Exception as e:
            print(f"  ERROR — {cfg.dir_name}: {e}")
            import traceback
            traceback.print_exc()
            print("  Skipping to next config...")

    print(f"\n{'=' * 72}")
    print(f"  Sweep complete.  {completed}/{total - start_from} configs succeeded.")
    print(f"  Output: {BASE_OUTPUT_PATH}/perturbed/")
    print(f"{'=' * 72}\n")

if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    run(start_from=0)