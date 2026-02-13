import carla
import json
import gzip
import glob
import numpy as np
import time
import laspy
import random
from pathlib import Path

CLASS_BLUEPRINT_MAP = {
    'ego_vehicle': 'vehicle.lincoln.mkz_2020',
    'vehicle': 'vehicle.tesla.model3',
    'walker': 'walker.pedestrian.0001',
    'traffic_light': 'static.prop.streetsign',
    'traffic_sign': 'static.prop.streetsign',
}

MESH_ID_MAP = {
    'Charger': 'vehicle.dodge.charger_2020',
    'FordCrown': 'vehicle.ford.crown',
    'Lincoln': 'vehicle.lincoln.mkz_2020',
    'MercedesCCC': 'vehicle.mercedes.coupe_2020',
    'NissanPatrol2021': 'vehicle.nissan.patrol_2021',
}

RANDOM_OBJECT_BLUEPRINTS = [
    'static.prop.trafficcone01',
    'static.prop.trafficwarning',
    'static.prop.streetbarrier',
    'static.prop.constructioncone',
]

class LidarRecorder:
    def __init__(
        self,
        dataset_path='data/Bench2Drive-mini/',
        output_path='./data/recorded-lidar',
        channels=64,
        points_per_second=2240000,
        rotation_frequency=20,
        lidar_range=100,
        upper_fov=10,
        lower_fov=-30,
        spawn_frequency=(5, 10),
        spawn_range=(5, 10),
        host='127.0.0.1',
        port=2000,
        verbose=False
    ):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.verbose = verbose
        self.host = host
        self.port = port
        
        self.lidar_config = {
            'channels': channels,
            'points_per_second': points_per_second,
            'rotation_frequency': rotation_frequency,
            'range': lidar_range,
            'upper_fov': upper_fov,
            'lower_fov': lower_fov,
        }
        
        self.spawn_frequency = spawn_frequency
        self.spawn_range = spawn_range
        
        self.lidar_sensor = None
        self.current_lidar_data = None
        self.lidar_data_ready = False
        
    def lidar_callback(self, point_cloud):
        data = np.frombuffer(point_cloud.raw_data, dtype=np.float32)
        data = np.reshape(data, (int(data.shape[0] / 4), 4))
        self.current_lidar_data = data
        self.lidar_data_ready = True

    def save_lidar_scan(self, lidar_data, output_path, frame_idx):
        """Save LiDAR scan to .laz file in the same format as dataset."""
        try:
            
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            header = laspy.LasHeader(point_format=0, version="1.2")
            header.offsets = np.min(lidar_data[:, :3], axis=0)
            header.scales = np.array([0.001, 0.001, 0.001])
            
            las = laspy.LasData(header)
            las.x = lidar_data[:, 0]
            las.y = lidar_data[:, 1]
            las.z = lidar_data[:, 2]
            
            if lidar_data.shape[1] > 3:
                las.intensity = (lidar_data[:, 3] * 65535).astype(np.uint16)
            
            output_file = output_path / f"{frame_idx:05d}.laz"
            las.write(output_file)
            
            return True
        except Exception as e:
            print(f"Error saving LiDAR scan: {e}")
            return False

    def setup_lidar_sensor(self, world, ego_vehicle):
        """Attach LiDAR sensor to ego vehicle with configurable parameters."""
        blueprint_library = world.get_blueprint_library()
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        
        for key, value in self.lidar_config.items():
            lidar_bp.set_attribute(key, str(value))
        
        lidar_transform = carla.Transform(carla.Location(x=0, y=0, z=2))
        self.lidar_sensor = world.spawn_actor(lidar_bp, lidar_transform, attach_to=ego_vehicle)
        self.lidar_sensor.listen(self.lidar_callback)
        
        print(f"✓ LiDAR sensor attached with config: {self.lidar_config}")
        return self.lidar_sensor

    def get_blueprint(self, blueprint_library, actor_class, type_id):
        """Safely retrieves a CARLA blueprint for a given class and type_id."""
        if type_id and type_id.startswith('/Game/'):
            for key, val in MESH_ID_MAP.items():
                if key in type_id:
                    return blueprint_library.find(val)
            return blueprint_library.find(CLASS_BLUEPRINT_MAP['vehicle'])
        
        if type_id:
            try:
                bp = blueprint_library.find(type_id)
                if bp:
                    return bp
            except Exception:
                pass
        
        fallback_id = CLASS_BLUEPRINT_MAP.get(actor_class, 'static.prop.fountain')
        try:
            return blueprint_library.find(fallback_id)
        except Exception:
            return blueprint_library.find('static.prop.fountain')

    def spawn_random_objects(self, world, ego_transform, frame_idx):
        """Spawn random objects around ego vehicle."""
        num_objects = random.randint(self.spawn_frequency[0], self.spawn_frequency[1])
        spawned_objects = []
        random_actors = []
        
        blueprint_library = world.get_blueprint_library()
        
        for i in range(num_objects):
            # Random distance and angle
            distance = random.uniform(self.spawn_range[0], self.spawn_range[1])
            angle = random.uniform(0, 2 * np.pi)
            
            # Calculate spawn position relative to ego
            x_offset = distance * np.cos(angle)
            y_offset = distance * np.sin(angle)
            
            spawn_location = carla.Location(
                x=ego_transform.location.x + x_offset,
                y=ego_transform.location.y + y_offset,
                z=ego_transform.location.z
            )
            
            spawn_rotation = carla.Rotation(
                pitch=0,
                roll=0,
                yaw=random.uniform(0, 360)
            )
            
            spawn_transform = carla.Transform(spawn_location, spawn_rotation)
            
            # Try to spawn object
            try:
                bp_name = random.choice(RANDOM_OBJECT_BLUEPRINTS)
                bp = blueprint_library.find(bp_name)
                actor = world.try_spawn_actor(bp, spawn_transform)
                
                if actor:
                    actor.set_simulate_physics(False)
                    random_actors.append(actor)
                    
                    # Create annotation for this object
                    bbox = actor.bounding_box
                    extent = bbox.extent
                    
                    annotation = {
                        'class': 'random_object',
                        'id': f'random_{frame_idx}_{i}',
                        'type_id': bp_name,
                        'base_type': 'static',
                        'location': [spawn_location.x, spawn_location.y, spawn_location.z],
                        'rotation': [spawn_rotation.pitch, spawn_rotation.roll, spawn_rotation.yaw],
                        'extent': [extent.x, extent.y, extent.z],
                    }
                    spawned_objects.append(annotation)
                    
            except Exception as e:
                if self.verbose:
                    print(f"Failed to spawn random object: {e}")
        
        return spawned_objects, random_actors

    def replay_and_save_scans(self, instance_path, client):
        instance_path = Path(instance_path)
        anno_dir = instance_path / 'anno'
        
        if not anno_dir.exists():
            print(f"Error: Annotation directory {anno_dir} not found.")
            return None
        
        anno_files = sorted(glob.glob(str(anno_dir / '*.json.gz')))
        if not anno_files:
            print(f"Error: No annotation files found in {anno_dir}.")
            return None
        
        parts = instance_path.name.split('_')
        town_name = next((p for p in parts if p.startswith('Town')), None)
        if not town_name:
            print(f"Error: Could not identify map name from {instance_path.name}")
            return None
        
        # Create output directories
        output_base = self.output_path / instance_path.name
        lidar_dir = output_base / 'lidar'
        anno_original_dir = output_base / 'anno_original'
        anno_new_dir = output_base / 'anno_new'
        
        lidar_dir.mkdir(parents=True, exist_ok=True)
        anno_original_dir.mkdir(parents=True, exist_ok=True)
        anno_new_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"[{instance_path.name}]")
        print(f"{'='*80}")
        print(f"Loading map {town_name}...")
        print(f"Output directory: {output_base}")
        
        try:
            world = client.load_world(town_name)
        except RuntimeError as e:
            print(f"❌ {e}")
            return None
        
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        
        blueprint_library = world.get_blueprint_library()
        spawned_actors = {}
        failed_spawn_ids = set()
        scans_saved = 0
        
        try:
            for frame_idx, anno_file in enumerate(anno_files):
                with gzip.open(anno_file, 'rt') as f:
                    data = json.load(f)
                
                # Save original annotation
                original_anno_file = anno_original_dir / f"{frame_idx:05d}.json.gz"
                with gzip.open(original_anno_file, 'wt') as f:
                    json.dump(data, f)
                
                current_frame_actor_ids = set()
                ego_transform = None
                
                for bb in data.get('bounding_boxes', []):
                    actor_class = bb.get('class')
                    actor_id = bb.get('id')
                    current_frame_actor_ids.add(actor_id)
                    
                    loc_raw = bb.get('location')
                    rot_raw = bb.get('rotation', [0, 0, 0])
                    location = carla.Location(x=loc_raw[0], y=loc_raw[1], z=loc_raw[2])
                    rotation = carla.Rotation(pitch=rot_raw[0], roll=rot_raw[1], yaw=rot_raw[2])
                    transform = carla.Transform(location, rotation)
                    
                    if actor_class == 'ego_vehicle':
                        ego_transform = transform
                    
                    if actor_id not in spawned_actors and actor_id not in failed_spawn_ids:
                        type_id = bb.get('type_id')
                        try:
                            bp = self.get_blueprint(blueprint_library, actor_class, type_id)
                            actor = world.try_spawn_actor(bp, transform)
                            
                            if not actor:
                                safe_transform = carla.Transform(location + carla.Location(z=2.0), rotation)
                                actor = world.try_spawn_actor(bp, safe_transform)
                            
                            if actor:
                                if self.verbose:
                                    print(f"Spawned {actor_class} {actor_id}")
                                actor.set_simulate_physics(False)
                                actor.set_transform(transform)
                                spawned_actors[actor_id] = actor
                                
                                if actor_class == 'ego_vehicle' and self.lidar_sensor is None:
                                    self.setup_lidar_sensor(world, actor)
                            else:
                                failed_spawn_ids.add(actor_id)
                        except Exception as e:
                            if self.verbose:
                                print(f"Error spawning {actor_id}: {e}")
                            failed_spawn_ids.add(actor_id)
                            continue
                    
                    if actor_id in spawned_actors:
                        spawned_actors[actor_id].set_transform(transform)
                    
                    if actor_class == 'ego_vehicle' and actor_id in spawned_actors:
                        spectator = world.get_spectator()
                        fwd = transform.get_forward_vector()
                        spec_loc = location - fwd * 12 + carla.Location(z=6)
                        spec_rot = carla.Rotation(pitch=-20, yaw=rotation.yaw, roll=0)
                        spectator.set_transform(carla.Transform(spec_loc, spec_rot))
                
                # Spawn random objects
                random_objects_anno = []
                random_actors = []
                if ego_transform and self.spawn_frequency[1] > 0:
                    random_objects_anno, random_actors = self.spawn_random_objects(world, ego_transform, frame_idx)
                
                # Remove actors no longer in scene
                actors_to_remove = set(spawned_actors.keys()) - current_frame_actor_ids
                for aid in actors_to_remove:
                    spawned_actors[aid].destroy()
                    del spawned_actors[aid]
                
                # Apply weather
                w_data = data['weather']
                weather = carla.WeatherParameters(
                    cloudiness=w_data['cloudiness'],
                    precipitation=w_data['precipitation'],
                    precipitation_deposits=w_data['precipitation_deposits'],
                    wind_intensity=w_data['wind_intensity'],
                    sun_azimuth_angle=w_data['sun_azimuth_angle'],
                    sun_altitude_angle=w_data['sun_altitude_angle'],
                    fog_density=w_data['fog_density'],
                    fog_distance=w_data['fog_distance'],
                    wetness=w_data['wetness'],
                    fog_falloff=w_data['fog_falloff']
                )
                world.set_weather(weather)
                
                world.tick()
                
                # Save LiDAR scan
                if self.lidar_sensor is not None:
                    time.sleep(0.05)
                    if self.lidar_data_ready and self.current_lidar_data is not None:
                        if self.save_lidar_scan(self.current_lidar_data, lidar_dir, frame_idx):
                            scans_saved += 1
                            if scans_saved % 50 == 0:
                                print(f"  Saved {scans_saved}/{len(anno_files)} scans")
                        self.lidar_data_ready = False
                
                # Save new annotation with random objects
                new_data = data.copy()
                new_data['bounding_boxes'] = data['bounding_boxes'] + random_objects_anno
                
                new_anno_file = anno_new_dir / f"{frame_idx:05d}.json.gz"
                with gzip.open(new_anno_file, 'wt') as f:
                    json.dump(new_data, f)
                
                # Cleanup random objects for next frame
                for actor in random_actors:
                    try:
                        actor.destroy()
                    except:
                        pass
                
                if frame_idx % 50 == 0 and scans_saved == 0:
                    print(f"  Progress: {frame_idx}/{len(anno_files)} frames")
            
            print(f"\n✓ Saved {scans_saved} LiDAR scans to {lidar_dir}")
            return scans_saved
        
        finally:
            print(f"Cleaning up...")
            
            if self.lidar_sensor is not None:
                try:
                    self.lidar_sensor.stop()
                    self.lidar_sensor.destroy()
                    self.lidar_sensor = None
                except Exception as e:
                    print(f"Error destroying LiDAR sensor: {e}")
            
            for actor in spawned_actors.values():
                try:
                    actor.destroy()
                except Exception:
                    pass
            
            settings.synchronous_mode = False
            world.apply_settings(settings)

    def run(self):
        instances = []
        
        if (self.dataset_path / 'anno').exists():
            instances = [self.dataset_path]
        else:
            subdirs = sorted([d for d in self.dataset_path.iterdir() if d.is_dir()])
            for d in subdirs:
                if (d / 'anno').exists():
                    instances.append(d)
        
        if not instances:
            print(f"❌ No instances found at {self.dataset_path}")
            return
        
        print(f"{'='*80}")
        print(f"CARLA LiDAR Scan Recorder")
        print(f"{'='*80}")
        print(f"Found {len(instances)} instances")
        print(f"LiDAR Config: {self.lidar_config}")
        print(f"Spawn Frequency: {self.spawn_frequency}")
        print(f"Spawn Range: {self.spawn_range}")
        print(f"Connecting to CARLA at {self.host}:{self.port}...")
        
        try:
            client = carla.Client(self.host, self.port)
            client.set_timeout(200.0)
            print(f"✓ Connected to CARLA server")
        except Exception as e:
            print(f"❌ Failed to connect to CARLA: {e}")
            return
        
        total_scans = 0
        for i, instance in enumerate(instances, 1):
            print(f"\n[Instance {i}/{len(instances)}]")
            try:
                scans = self.replay_and_save_scans(instance, client)
                if scans:
                    total_scans += scans
            except KeyboardInterrupt:
                print(f"\n\n❌ Interrupted by user")
                break
            except Exception as e:
                print(f"❌ Failed to process {instance.name}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n{'='*80}")
        print(f"✓ Total scans saved: {total_scans}")
        print(f"✓ Output location: {self.output_path}")
        print(f"{'='*80}")

if __name__ == "__main__":
    recorder = LidarRecorder()
    recorder.run()