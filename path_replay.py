import carla
import json
import gzip
import glob
import argparse
from pathlib import Path

# Fallback mapping for classes if type_id is missing or fails to find a blueprint
# Verified available blueprints in this CARLA installation:
# 'static.prop.streetsign', 'static.prop.trafficcone01', 'static.prop.trafficwarning'
CLASS_BLUEPRINT_MAP = {
    'ego_vehicle': 'vehicle.lincoln.mkz_2020',
    'vehicle': 'vehicle.tesla.model3',
    'walker': 'walker.pedestrian.0001',
    'traffic_light': 'static.prop.streetsign',  # Fallback for missing traffic light props
    'traffic_sign': 'static.prop.streetsign',
}

# Specific mapping for static mesh paths found in the dataset
MESH_ID_MAP = {
    'Charger': 'vehicle.dodge.charger_2020',
    'FordCrown': 'vehicle.ford.crown',
    'Lincoln': 'vehicle.lincoln.mkz_2020',
    'MercedesCCC': 'vehicle.mercedes.coupe_2020',
    'NissanPatrol2021': 'vehicle.nissan.patrol_2021',
}

def get_blueprint(blueprint_library, actor_class, type_id):
    """Safely retrieves a CARLA blueprint for a given class and type_id."""
    # 1. Handle non-standard mesh paths (e.g., /Game/Carla/Static/Car/...)
    if type_id and type_id.startswith('/Game/'):
        for key, val in MESH_ID_MAP.items():
            if key in type_id:
                return blueprint_library.find(val)
        return blueprint_library.find(CLASS_BLUEPRINT_MAP['vehicle'])

    # 2. Try to find the explicit type_id
    if type_id:
        try:
            bp = blueprint_library.find(type_id)
            if bp:
                return bp
        except Exception:
            pass

    # 3. Fallback to class-based mapping
    fallback_id = CLASS_BLUEPRINT_MAP.get(actor_class, 'static.prop.fountain')
    try:
        return blueprint_library.find(fallback_id)
    except Exception:
        return blueprint_library.find('static.prop.fountain') # Ultimate safe prop

def replay_instance(instance_path, client):
    instance_path = Path(instance_path)
    anno_dir = instance_path / 'anno'
    
    if not anno_dir.exists():
        print(f"Error: Annotation directory {anno_dir} not found.")
        return

    anno_files = sorted(glob.glob(str(anno_dir / '*.json.gz')))
    if not anno_files:
        print(f"Error: No annotation files found in {anno_dir}.")
        return

    parts = instance_path.name.split('_')
    town_name = next((p for p in parts if p.startswith('Town')), None)
    
    if not town_name:
        print(f"Error: Could not identify map name from {instance_path.name}")
        return

    print(f"\n[{instance_path.name}] Loading map {town_name}...")
    world = client.load_world(town_name)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05 # 20 FPS
    world.apply_settings(settings)
    
    blueprint_library = world.get_blueprint_library()
    
    spawned_actors = {}
    failed_spawn_ids = set()
    
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
                            print(f"Spawned {actor_class} {actor_id} ({getattr(bp, 'id', 'unknown')})")
                            actor.set_simulate_physics(False)
                            actor.set_transform(transform) 
                            spawned_actors[actor_id] = actor
                        else:
                            failed_spawn_ids.add(actor_id)
                    except Exception as e:
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

            actors_to_remove = set(spawned_actors.keys()) - current_frame_actor_ids
            for aid in actors_to_remove:
                spawned_actors[aid].destroy()
                del spawned_actors[aid]

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
            if frame_idx % 100 == 0:
                print(f"  Frame {frame_idx}/{len(anno_files)}")

    finally:
        print(f"Cleaning up instance {instance_path.name}...")
        for actor in spawned_actors.values():
            try:
                actor.destroy()
            except Exception:
                pass
        settings.synchronous_mode = False
        world.apply_settings(settings)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='Path to instance directory or dataset root')
    args = parser.parse_args()

    input_path = Path(args.path)
    
    instances = []
    if (input_path / 'anno').exists():
        instances = [input_path]
    else:
        subdirs = sorted([d for d in input_path.iterdir() if d.is_dir()])
        for d in subdirs:
            if (d / 'anno').exists():
                instances.append(d)
    
    if not instances:
        print(f"No instances found at {input_path}")
        return

    print(f"Found {len(instances)} instances to replay.")

    print(f"Connecting to CARLA at 127.0.0.1:2000...")
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(200.0)

    for instance in instances:
        try:
            replay_instance(instance, client)
        except Exception as e:
            print(f"Failed to replay {instance.name}: {e}")

    print("\nAll replays finished.")

if __name__ == "__main__":
    main()

