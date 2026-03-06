import gzip
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import carla
import laspy
import numpy as np


@dataclass
class Perturbation:
    blueprint: str
    rotation_angle: int  # yaw; also used as ego-relative spawn direction
    spawn_distance_from_ego: int
    global_position_fixed: bool
    position_jitter: float = 0.0  # +- metres applied to x/y each frame


class LidarRecorder:
    def __init__(
        self,
        class_blueprint_map: dict,
        mesh_id_map: dict,
        lidar_config: dict,
        fixed_delta_seconds: float,
        output_path: str = "./data/recorded-lidar",
        verbose: bool = False,
    ):
        self.class_blueprint_map = class_blueprint_map
        self.mesh_id_map = mesh_id_map
        self.lidar_config = lidar_config
        self.fixed_delta_seconds = fixed_delta_seconds
        self.output_path = Path(output_path)
        self.verbose = verbose

        self._lidar_event = threading.Event()

        self.world: Optional[carla.World] = None
        self.blueprint_library = None
        self.lidar_sensor: Optional[carla.Actor] = None
        self.current_lidar_data: Optional[np.ndarray] = None
        self.lidar_data_ready: bool = False

        self.spawned_actors: dict[str, carla.Actor] = {}
        self.failed_spawn_ids: set[str] = set()

        self._perturbation_actor: Optional[carla.Actor] = None
        self._perturbation_fixed_location: Optional[carla.Location] = None

        self.lidar_dir: Optional[Path] = None
        self.anno_original_dir: Optional[Path] = None
        self.anno_new_dir: Optional[Path] = None

    def _lidar_callback(self, point_cloud):
        data = np.frombuffer(point_cloud.raw_data, dtype=np.float32).reshape(-1, 4)
        self.current_lidar_data = data
        self.lidar_data_ready = True
        self._lidar_event.set()

    def _setup_lidar(self, ego_vehicle: carla.Actor):
        bp = self.blueprint_library.find("sensor.lidar.ray_cast")
        for k, v in self.lidar_config.items():
            bp.set_attribute(k, str(v))
        self.lidar_sensor = self.world.spawn_actor(
            bp, carla.Transform(carla.Location(z=2)), attach_to=ego_vehicle
        )
        self.lidar_sensor.listen(self._lidar_callback)

    def _save_lidar_scan(self, lidar_data: np.ndarray, frame_idx: int) -> bool:
        try:
            header = laspy.LasHeader(point_format=0, version="1.2")
            header.offsets = np.min(lidar_data[:, :3], axis=0)
            header.scales = np.array([0.001, 0.001, 0.001])
            las = laspy.LasData(header)
            las.x, las.y, las.z = lidar_data[:, 0], lidar_data[:, 1], lidar_data[:, 2]
            if lidar_data.shape[1] > 3:
                las.intensity = (lidar_data[:, 3] * 65535).astype(np.uint16)
            las.write(str(self.lidar_dir / f"{frame_idx:05d}.laz"))
            return True
        except Exception as e:
            print(f"lidar save error frame {frame_idx}: {e}")
            return False

    def _resolve_blueprint(self, actor_class: str, type_id: Optional[str]):
        if type_id and type_id.startswith("/Game/"):
            for key, val in self.mesh_id_map.items():
                if key in type_id:
                    return self.blueprint_library.find(val)
            return self.blueprint_library.find(self.class_blueprint_map["vehicle"])

        if type_id:
            try:
                bp = self.blueprint_library.find(type_id)
                if bp:
                    return bp
            except Exception:
                pass

        fallback = self.class_blueprint_map.get(actor_class, "static.prop.fountain")
        try:
            return self.blueprint_library.find(fallback)
        except Exception:
            return self.blueprint_library.find("static.prop.fountain")

    def _apply_perturbation(
        self, p: Perturbation, ego_transform: carla.Transform, frame_idx: int
    ) -> Optional[dict]:
        import random

        if p.global_position_fixed and self._perturbation_fixed_location is not None:
            base_loc = self._perturbation_fixed_location
        else:
            angle_rad = np.deg2rad(p.rotation_angle)
            base_loc = carla.Location(
                x=ego_transform.location.x
                + p.spawn_distance_from_ego * np.cos(angle_rad),
                y=ego_transform.location.y
                + p.spawn_distance_from_ego * np.sin(angle_rad),
                z=ego_transform.location.z,
            )
            if p.global_position_fixed:
                self._perturbation_fixed_location = base_loc

        jx = (
            random.uniform(-p.position_jitter, p.position_jitter)
            if p.position_jitter
            else 0.0
        )
        jy = (
            random.uniform(-p.position_jitter, p.position_jitter)
            if p.position_jitter
            else 0.0
        )
        location = carla.Location(x=base_loc.x + jx, y=base_loc.y + jy, z=base_loc.z)
        rotation = carla.Rotation(yaw=float(p.rotation_angle))
        transform = carla.Transform(location, rotation)

        if self._perturbation_actor is None or not self._perturbation_actor.is_alive:
            self._destroy_perturbation_actor()
            try:
                actor = self.world.try_spawn_actor(
                    self.blueprint_library.find(p.blueprint), transform
                )
                if not actor:
                    print(f"could not spawn perturbation '{p.blueprint}'")
                    return None
                actor.set_simulate_physics(False)
                self._perturbation_actor = actor
            except Exception as e:
                print(f"perturbation spawn failed: {e}")
                return None
        else:
            self._perturbation_actor.set_transform(transform)

        extent = self._perturbation_actor.bounding_box.extent
        return {
            "class": "perturbation",
            "id": f"perturbation_{frame_idx}",
            "type_id": p.blueprint,
            "base_type": "static",
            "location": [location.x, location.y, location.z],
            "rotation": [rotation.pitch, rotation.roll, rotation.yaw],
            "extent": [extent.x, extent.y, extent.z],
            "global_position_fixed": p.global_position_fixed,
        }

    def _destroy_perturbation_actor(self):
        if self._perturbation_actor is not None:
            try:
                if self._perturbation_actor.is_alive:
                    self._perturbation_actor.destroy()
            except Exception:
                pass
            self._perturbation_actor = None
        self._perturbation_fixed_location = None

    def initialize_scene(
        self,
        instance_path: Path,
        client: carla.Client,
    ) -> bool:
        instance_path = Path(instance_path)
        town_name = next(
                (p for p in instance_path.name.split("_") if p.startswith("Town")), None
            )
        if not town_name:
            print(f"could not identify map from '{str(instance_path)}'")
            return False

        try:
            self.world = client.load_world(town_name)
        except RuntimeError as e:
            print(f"load_world failed: {e}")
            return False

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.fixed_delta_seconds
        self.world.apply_settings(settings)

        self.blueprint_library = self.world.get_blueprint_library()
        self.spawned_actors = {}
        self.failed_spawn_ids = set()
        self.lidar_sensor = None
        self.lidar_data_ready = False
        self._lidar_event.clear()
        self._perturbation_actor = None
        self._perturbation_fixed_location = None

        output_base = self.output_path / instance_path.name
        self.lidar_dir = output_base / "lidar"
        self.anno_original_dir = output_base / "anno_original"
        self.anno_new_dir = output_base / "anno_new"
        for d in (self.lidar_dir, self.anno_original_dir, self.anno_new_dir):
            d.mkdir(parents=True, exist_ok=True)

        return True

    def process_frame(
        self,
        anno_file: str,
        frame_idx: int,
        perturbation: Optional[Perturbation] = None,
    ) -> bool:
        if self.world is None:
            raise RuntimeError("call initialize_scene() before process_frame()")

        with gzip.open(anno_file, "rt") as f:
            data = json.load(f)

        with gzip.open(self.anno_original_dir / f"{frame_idx:05d}.json.gz", "wt") as f:
            json.dump(data, f)

        current_ids: set[str] = set()
        ego_transform: Optional[carla.Transform] = None

        for bb in data.get("bounding_boxes", []):
            actor_class = bb.get("class")
            actor_id = bb.get("id")
            current_ids.add(actor_id)

            loc = bb.get("location")
            rot = bb.get("rotation", [0, 0, 0])
            transform = carla.Transform(
                carla.Location(x=loc[0], y=loc[1], z=loc[2]),
                carla.Rotation(pitch=rot[0], roll=rot[1], yaw=rot[2]),
            )

            if actor_class == "ego_vehicle":
                ego_transform = transform

            if (
                actor_id not in self.spawned_actors
                and actor_id not in self.failed_spawn_ids
            ):
                try:
                    bp = self._resolve_blueprint(actor_class, bb.get("type_id"))
                    actor = self.world.try_spawn_actor(
                        bp, transform
                    ) or self.world.try_spawn_actor(
                        bp,
                        carla.Transform(
                            transform.location + carla.Location(z=2.0),
                            transform.rotation,
                        ),
                    )
                    if actor:
                        actor.set_simulate_physics(False)
                        self.spawned_actors[actor_id] = actor
                        if actor_class == "ego_vehicle" and self.lidar_sensor is None:
                            self._setup_lidar(actor)
                    else:
                        self.failed_spawn_ids.add(actor_id)
                        if self.verbose:
                            print(f"failed to spawn {actor_class} {actor_id}")
                except Exception as e:
                    if self.verbose:
                        print(f"spawn error {actor_id}: {e}")
                    self.failed_spawn_ids.add(actor_id)

            if actor_id in self.spawned_actors:
                self.spawned_actors[actor_id].set_transform(transform)

            if actor_class == "ego_vehicle" and actor_id in self.spawned_actors:
                fwd = transform.get_forward_vector()
                self.world.get_spectator().set_transform(
                    carla.Transform(
                        transform.location - fwd * 12 + carla.Location(z=6),
                        carla.Rotation(pitch=-20, yaw=transform.rotation.yaw),
                    )
                )

        perturbation_anno = []
        if perturbation is not None and ego_transform is not None:
            anno = self._apply_perturbation(perturbation, ego_transform, frame_idx)
            if anno:
                perturbation_anno.append(anno)
        else:
            self._destroy_perturbation_actor()

        for aid in set(self.spawned_actors) - current_ids:
            try:
                self.spawned_actors.pop(aid).destroy()
            except Exception:
                pass

        w = data.get("weather", {})
        self.world.set_weather(
            carla.WeatherParameters(
                cloudiness=w.get("cloudiness", 0.0),
                precipitation=w.get("precipitation", 0.0),
                precipitation_deposits=w.get("precipitation_deposits", 0.0),
                wind_intensity=w.get("wind_intensity", 0.0),
                sun_azimuth_angle=w.get("sun_azimuth_angle", 0.0),
                sun_altitude_angle=w.get("sun_altitude_angle", 45.0),
                fog_density=w.get("fog_density", 0.0),
                fog_distance=w.get("fog_distance", 0.0),
                wetness=w.get("wetness", 0.0),
                fog_falloff=w.get("fog_falloff", 0.0),
            )
        )

        self.world.tick()

        scan_saved = False
        if self.lidar_sensor is not None:
            fired = self._lidar_event.wait(timeout=self.fixed_delta_seconds * 3)
            self._lidar_event.clear()
            if fired and self.current_lidar_data is not None:
                scan_saved = self._save_lidar_scan(self.current_lidar_data, frame_idx)
                self.lidar_data_ready = False
            elif not fired:
                print(f"lidar callback timeout frame {frame_idx}")

        new_data = {
            **data,
            "bounding_boxes": data["bounding_boxes"] + perturbation_anno,
        }
        with gzip.open(self.anno_new_dir / f"{frame_idx:05d}.json.gz", "wt") as f:
            json.dump(new_data, f)

        return scan_saved

    def cleanup(self):
        self._destroy_perturbation_actor()

        if self.lidar_sensor is not None:
            try:
                self.lidar_sensor.stop()
                self.lidar_sensor.destroy()
            except Exception as e:
                print(f"lidar cleanup error: {e}")
            self.lidar_sensor = None

        for actor in self.spawned_actors.values():
            try:
                actor.destroy()
            except Exception:
                pass
        self.spawned_actors = {}

        if self.world is not None:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
            self.world = None
