import glob
import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import carla

from .lidar_recorder import LidarRecorder, Perturbation

CLASS_BLUEPRINT_MAP: dict[str, str] = {
    "ego_vehicle": "vehicle.lincoln.mkz_2020",
    "vehicle": "vehicle.tesla.model3",
    "walker": "walker.pedestrian.0001",
    "traffic_light": "static.prop.streetsign",
    "traffic_sign": "static.prop.streetsign",
}

MESH_ID_MAP: dict[str, str] = {
    "Charger": "vehicle.dodge.charger_2020",
    "FordCrown": "vehicle.ford.crown",
    "Lincoln": "vehicle.lincoln.mkz_2020",
    "MercedesCCC": "vehicle.mercedes.coupe_2020",
    "NissanPatrol2021": "vehicle.nissan.patrol_2021",
}


class PerturbationPattern(Enum):
    CLEAN_HISTORY = 1  # First part clean, second part perturbed
    CLEAN_OBSERVATION = 2  # First part perturbed, second part clean
    CONSTANT = 3  # All perturbed
    IN_AND_OUT = 4  # Randomly enter and exit perturbation
    NOISY_MIDDLE = 5  # Clean, then perturbed, then clean again


def build_mask(pattern: PerturbationPattern, n: int) -> list[bool]:
    match pattern:
        case PerturbationPattern.CLEAN_HISTORY:
            split = random.randint(1, n - 1)
            return [False] * split + [True] * (n - split)

        case PerturbationPattern.CLEAN_OBSERVATION:
            split = random.randint(1, n - 1)
            return [True] * split + [False] * (n - split)

        case PerturbationPattern.CONSTANT:
            return [True] * n

        case PerturbationPattern.IN_AND_OUT:
            start = random.randint(0, n - 1)
            end = random.randint(start + 1, n)
            return [True if start <= i < end else False for i in range(n)]

        case PerturbationPattern.NOISY_MIDDLE:
            a = random.randint(1, n - 2)
            b = random.randint(a + 1, n - 1)
            return [False] * a + [True] * (b - a) + [False] * (n - b)

        case _:
            raise ValueError(f"unknown pattern: {pattern}")


def _chunk(lst: list, size: int) -> list[list]:
    return [lst[i : i + size] for i in range(0, len(lst), size)]


@dataclass
class LidarConfig:
    channels: int = 64
    points_per_second: int = 2_240_000
    rotation_frequency: int = 10
    range: int = 100
    upper_fov: int = 10
    lower_fov: int = -30

    def to_dict(self) -> dict:
        return {
            "channels": self.channels,
            "points_per_second": self.points_per_second,
            "rotation_frequency": self.rotation_frequency,
            "range": self.range,
            "upper_fov": self.upper_fov,
            "lower_fov": self.lower_fov,
        }


@dataclass
class DatasetBuilder:
    perturbation: Perturbation
    mask: list[bool]

    source_dataset_path: str = "data/Bench2Drive-mini/"
    fixed_delta_seconds: float = 0.1
    output_path: str = "./data/recorded-lidar"
    instance_filter: Optional[list[str]] = None
    host: str = "127.0.0.1"
    port: int = 2000
    lidar_config: LidarConfig = field(default_factory=LidarConfig)
    class_blueprint_map: dict[str, str] = field(
        default_factory=lambda: CLASS_BLUEPRINT_MAP.copy()
    )
    mesh_id_map: dict[str, str] = field(default_factory=lambda: MESH_ID_MAP.copy())
    verbose: bool = False

    def _collect_instances(self) -> list[Path]:
        base = Path(self.source_dataset_path)
        if (base / "anno").exists():
            return [base]
        instances = [d for d in base.iterdir() if d.is_dir() and (d / "anno").exists()]
        if self.instance_filter:
            instances = [
                d for d in instances if any(f in d.name for f in self.instance_filter)
            ]

        def town_key(p: Path):
            part = next(
                (s for s in p.name.split("_") if s.startswith("Town")), "Town00"
            )
            return part

        return sorted(instances, key=town_key)

    def build_dataset(self):
        instances = self._collect_instances()
        if not instances:
            print(f"no instances found at {self.source_dataset_path}")
            return

        try:
            client = carla.Client(self.host, self.port)
            client.set_timeout(200.0)
        except Exception as e:
            print(f"carla connection failed: {e}")
            return

        recorder = LidarRecorder(
            class_blueprint_map=self.class_blueprint_map,
            mesh_id_map=self.mesh_id_map,
            lidar_config=self.lidar_config.to_dict(),
            fixed_delta_seconds=self.fixed_delta_seconds,
            output_path=self.output_path,
            verbose=self.verbose,
        )

        n = len(self.mask)
        total_scans = 0
        for instance in instances:
            anno_files = sorted(glob.glob(str(instance / "anno" / "*.json.gz")))
            if not anno_files:
                continue

            try:
                recorder.output_path = Path(self.output_path)
                if not recorder.initialize_scene(instance, client):
                    continue

                scans_saved = 0
                for frame_idx, anno_file in enumerate(anno_files):
                    saved = recorder.process_frame(
                        anno_file=anno_file,
                        frame_idx=frame_idx,
                        perturbation=self.perturbation
                        if self.mask[frame_idx % n]
                        else None,
                    )
                    if saved:
                        scans_saved += 1

                print(f"{instance.name}: {scans_saved}/{len(anno_files)} scans")
                total_scans += scans_saved

            except KeyboardInterrupt:
                return
            except Exception as e:
                print(f"error on {instance.name}: {e}")
                import traceback

                traceback.print_exc()
            finally:
                recorder.cleanup()

        print(f"done. total scans: {total_scans}")
