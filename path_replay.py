import carla
import json
import gzip
import glob
import argparse
import numpy as np
import time
import laspy
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

lidar_sensor = None
current_lidar_data = None
lidar_data_ready = False

def lidar_callback(point_cloud):
    global current_lidar_data, lidar_data_ready
    data = np.frombuffer(point_cloud.raw_data, dtype=np.float32)
    data = np.reshape(data, (int(data.shape[0] / 4), 4))
    current_lidar_data = data  # Shape: (N, 4) - x, y, z, intensity
    lidar_data_ready = True

def save_lidar_scan(lidar_data, output_path, frame_idx):
    """Save LiDAR scan to .laz file in the same format as dataset."""
    try:
        import lazrs
        
        # Create output directory if it doesn't exist
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create LAZ file
        header = laspy.LasHeader(point_format=0, version="1.2")
        header.offsets = np.min(lidar_data[:, :3], axis=0)
        header.scales = np.array([0.001, 0.001, 0.001])  # 1mm precision
        
        las = laspy.LasData(header)
        
        # Set coordinates
        las.x = lidar_data[:, 0]
        las.y = lidar_data[:, 1]
        las.z = lidar_data[:, 2]
        
        # Set intensity if available
        if lidar_data.shape[1] > 3:
            las.intensity = (lidar_data[:, 3] * 65535).astype(np.uint16)
        
        # Save to file
        output_file = output_path / f"{frame_idx:05d}.laz"
        las.write(output_file)
        
        return True
    except Exception as e:
        print(f"Error saving LiDAR scan: {e}")
        return False

def setup_lidar_sensor(world, ego_vehicle, lidar_config=None):
    """Attach LiDAR sensor to ego vehicle with configurable parameters."""
    global lidar_sensor
    
    blueprint_library = world.get_blueprint_library()
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    
    # Default configuration (can be overridden)
    if lidar_config is None:
        lidar_config = {
            'channels': '64',
            'points_per_second': '1120000',
            'rotation_frequency': '20',
            'range': '100',
            'upper_fov': '10',
            'lower_fov': '-30',
        }
    
    # Apply configuration
    for key, value in lidar_config.items():
        lidar_bp.set_attribute(key, str(value))
    
    # Spawn LiDAR on top of ego vehicle
    lidar_transform = carla.Transform(carla.Location(x=0, y=0, z=2))
    lidar_sensor = world.spawn_actor(lidar_bp, lidar_transform, attach_to=ego_vehicle)
    
    # Register callback
    lidar_sensor.listen(lidar_callback)
    
    print(f"✓ LiDAR sensor attached with config: {lidar_config}")
    return lidar_sensor

def get_blueprint(blueprint_library, actor_class, type_id):
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

def replay_and_save_scans(instance_path, client, output_base_dir, lidar_config=None, verbose=False):
    global lidar_sensor, lidar_data_ready, current_lidar_data
    
    instance_path = Path(instance_path)
    anno_dir = instance_path / 'anno'
    
    if not anno_dir.exists():
        print(f"Error: Annotation directory {anno_dir} not found.")
        return None
    
    anno_files = sorted(glob.glob(str(anno_dir / '*.json.gz')))
    if not anno_files:
        print(f"Error: No annotation files found in {anno_dir}.")
        return None
    
    # Extract town name
    parts = instance_path.name.split('_')
    town_name = next((p for p in parts if p.startswith('Town')), None)
    if not town_name:
        print(f"Error: Could not identify map name from {instance_path.name}")
        return None
    
    # Create output directory
    output_dir = Path(output_base_dir) / instance_path.name / 'lidar'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"[{instance_path.name}]")
    print(f"{'='*80}")
    print(f"Loading map {town_name}...")
    print(f"Output directory: {output_dir}")
    
    try:
        world = client.load_world(town_name)
    except RuntimeError as e:
        print(f"❌ {e}")
        return None
    
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05  # 20 FPS
    world.apply_settings(settings)
    
    blueprint_library = world.get_blueprint_library()
    spawned_actors = {}
    failed_spawn_ids = set()
    scans_saved = 0
    
    try:
        for frame_idx, anno_file in enumerate(anno_files):
            with gzip.open(anno_file, 'rt') as f:
                data = json.load(f)
            
            current_frame_actor_ids = set()
            
            for bb in data.get('bounding_boxes', []):
                actor_class = bb.get('class')
                actor_id = bb.get('id')
                current_frame_actor_ids.add(actor_id)
                
                loc_raw = bb.get('location')
                rot_raw = bb.get('rotation', [0, 0, 0])
                location = carla.Location(x=loc_raw[0], y=loc_raw[1], z=loc_raw[2])
                rotation = carla.Rotation(pitch=rot_raw[0], roll=rot_raw[1], yaw=rot_raw[2])
                transform = carla.Transform(location, rotation)
                
                if actor_id not in spawned_actors and actor_id not in failed_spawn_ids:
                    type_id = bb.get('type_id')
                    try:
                        bp = get_blueprint(blueprint_library, actor_class, type_id)
                        actor = world.try_spawn_actor(bp, transform)
                        
                        if not actor:
                            safe_transform = carla.Transform(location + carla.Location(z=2.0), rotation)
                            actor = world.try_spawn_actor(bp, safe_transform)
                        
                        if actor:
                            if verbose:
                                print(f"Spawned {actor_class} {actor_id}")
                            actor.set_simulate_physics(False)
                            actor.set_transform(transform)
                            spawned_actors[actor_id] = actor
                            
                            # Setup LiDAR on ego vehicle
                            if actor_class == 'ego_vehicle' and lidar_sensor is None:
                                setup_lidar_sensor(world, actor, lidar_config)
                        else:
                            failed_spawn_ids.add(actor_id)
                    except Exception as e:
                        if verbose:
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
            
            # Tick the world
            world.tick()
            
            # Save LiDAR scan
            if lidar_sensor is not None:
                time.sleep(0.05)  # Wait for callback
                if lidar_data_ready and current_lidar_data is not None:
                    if save_lidar_scan(current_lidar_data, output_dir, frame_idx):
                        scans_saved += 1
                        if scans_saved % 50 == 0:
                            print(f"  Saved {scans_saved}/{len(anno_files)} scans")
                    lidar_data_ready = False
            
            if frame_idx % 50 == 0 and scans_saved == 0:
                print(f"  Progress: {frame_idx}/{len(anno_files)} frames")
        
        print(f"\n✓ Saved {scans_saved} LiDAR scans to {output_dir}")
        return scans_saved
    
    finally:
        print(f"Cleaning up...")
        
        if lidar_sensor is not None:
            try:
                lidar_sensor.stop()
                lidar_sensor.destroy()
                lidar_sensor = None
            except Exception as e:
                print(f"Error destroying LiDAR sensor: {e}")
        
        for actor in spawned_actors.values():
            try:
                actor.destroy()
            except Exception:
                pass
        
        settings.synchronous_mode = False
        world.apply_settings(settings)

def start_save_scans():
    parser = argparse.ArgumentParser(
        description='Replay Bench2Drive scenarios and save LiDAR scans',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('path', type=str, 
                       help='Path to instance directory or dataset root')
    parser.add_argument('--output', type=str, default='./data/recorded-lidar',
                       help='Output directory for saved scans')
    parser.add_argument('--channels', type=int, default=64,
                       help='LiDAR channels')
    parser.add_argument('--points-per-second', type=int, default=2240000,
                       help='LiDAR points per second (try 2240000 for 2x points)')
    parser.add_argument('--rotation-frequency', type=int, default=20,
                       help='LiDAR rotation frequency (Hz)')
    parser.add_argument('--range', type=int, default=100,
                       help='LiDAR range (meters)')
    parser.add_argument('--upper-fov', type=int, default=10,
                       help='Upper field of view (degrees)')
    parser.add_argument('--lower-fov', type=int, default=-30,
                       help='Lower field of view (degrees)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed spawn messages')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                       help='CARLA server host')
    parser.add_argument('--port', type=int, default=2000,
                       help='CARLA server port')
    
    args = parser.parse_args()
    
    # Build LiDAR config
    lidar_config = {
        'channels': args.channels,
        'points_per_second': args.points_per_second,
        'rotation_frequency': args.rotation_frequency,
        'range': args.range,
        'upper_fov': args.upper_fov,
        'lower_fov': args.lower_fov,
    }
    
    input_path = Path(args.path)
    instances = []
    
    # Find instances
    if (input_path / 'anno').exists():
        instances = [input_path]
    else:
        subdirs = sorted([d for d in input_path.iterdir() if d.is_dir()])
        for d in subdirs:
            if (d / 'anno').exists():
                instances.append(d)
    
    if not instances:
        print(f"❌ No instances found at {input_path}")
        return
    
    print(f"{'='*80}")
    print(f"CARLA LiDAR Scan Recorder")
    print(f"{'='*80}")
    print(f"Found {len(instances)} instances")
    print(f"LiDAR Config: {lidar_config}")
    print(f"Connecting to CARLA at {args.host}:{args.port}...")
    
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(200.0)
        print(f"✓ Connected to CARLA server")
    except Exception as e:
        print(f"❌ Failed to connect to CARLA: {e}")
        return
    
    # Process all instances
    total_scans = 0
    for i, instance in enumerate(instances, 1):
        print(f"\n[Instance {i}/{len(instances)}]")
        try:
            scans = replay_and_save_scans(
                instance,
                client,
                args.output,
                lidar_config=lidar_config,
                verbose=args.verbose
            )
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
    print(f"✓ Output location: {args.output}")
    print(f"{'='*80}")

if __name__ == "__main__":
    start_save_scans()