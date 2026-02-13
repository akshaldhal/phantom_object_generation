import argparse
import sys
from pathlib import Path
from .download_dataset import download_bech2drive_dataset, DatasetSize
from .laz_to_obj import laz_to_obj
from .lidar_recorder import LidarRecorder

def main():
    parser = argparse.ArgumentParser(description="Bench2Drive Phantom Objects CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Download command
    parser_download = subparsers.add_parser("download", help="Download Bench2Drive dataset")
    parser_download.add_argument("--dir", type=str, default="data/Bench2Drive", help="Target directory")
    parser_download.add_argument("--size", type=str, choices=["mini", "base", "full"], default="mini", help="Dataset size")

    # Record command
    parser_record = subparsers.add_parser("record", help="Record LiDAR data from CARLA")
    parser_record.add_argument("--dataset-path", type=str, default="data/Bench2Drive-mini/", help="Path to Bench2Drive dataset")
    parser_record.add_argument("--output-path", type=str, default="./data/recorded-lidar", help="Output path")
    parser_record.add_argument("--host", type=str, default="127.0.0.1", help="CARLA host")
    parser_record.add_argument("--port", type=int, default=2000, help="CARLA port")
    parser_record.add_argument("--channels", type=int, default=64, help="LiDAR channels")
    parser_record.add_argument("--points-per-second", type=int, default=2240000, help="LiDAR points per second")
    parser_record.add_argument("--rotation-frequency", type=int, default=20, help="LiDAR rotation frequency")
    parser_record.add_argument("--range", type=float, default=100.0, dest="lidar_range", help="LiDAR range")
    parser_record.add_argument("--upper-fov", type=float, default=10.0, help="LiDAR upper FOV")
    parser_record.add_argument("--lower-fov", type=float, default=-30.0, help="LiDAR lower FOV")
    parser_record.add_argument("--spawn-min", type=int, default=5, help="Min random objects")
    parser_record.add_argument("--spawn-max", type=int, default=10, help="Max random objects")
    parser_record.add_argument("--spawn-range-min", type=float, default=5.0, help="Min spawn distance")
    parser_record.add_argument("--spawn-range-max", type=float, default=10.0, help="Max spawn distance")
    parser_record.add_argument("--spawn-rotation-min", type=float, default=-180.0, help="Min spawn rotation")
    parser_record.add_argument("--spawn-rotation-max", type=float, default=180.0, help="Max spawn rotation")
    parser_record.add_argument("--spawn-persist-min", type=int, default=10, help="Min frame persistence")
    parser_record.add_argument("--spawn-persist-max", type=int, default=20, help="Max frame persistence")
    parser_record.add_argument("--allowed-objects", nargs="*", help="List of allowed object blueprints")
    parser_record.add_argument("--verbose", action="store_true", help="Verbose output")

    # Convert command
    parser_convert = subparsers.add_parser("convert", help="Convert LAZ to OBJ")
    parser_convert.add_argument("input", type=str, help="Input LAZ file or directory")
    parser_convert.add_argument("-o", "--output", type=str, default="output.obj", help="Output OBJ file")
    parser_convert.add_argument("--color-by", choices=["height", "intensity", "file", "none"], default="height", help="Coloring method")
    parser_convert.add_argument("--colormap", choices=['gray', 'hot', 'viridis', 'jet', 'rainbow', 'terrain', 'ocean'], default="rainbow", help="Colormap")
    parser_convert.add_argument("--subsample", type=int, help="Subsample points")

    args = parser.parse_args()

    if args.command == "download":
        size_map = {
            "mini": DatasetSize.MINI,
            "base": DatasetSize.BASE,
            "full": DatasetSize.FULL
        }
        download_bech2drive_dataset(local_dir=args.dir, size=size_map[args.size])
    
    elif args.command == "record":
        recorder = LidarRecorder(
            dataset_path=args.dataset_path,
            output_path=args.output_path,
            channels=args.channels,
            points_per_second=args.points_per_second,
            rotation_frequency=args.rotation_frequency,
            lidar_range=args.lidar_range,
            upper_fov=args.upper_fov,
            lower_fov=args.lower_fov,
            spawn_frequency=(args.spawn_min, args.spawn_max),
            spawn_range=(args.spawn_range_min, args.spawn_range_max),
            spawn_rotation_range=(args.spawn_rotation_min, args.spawn_rotation_max),
            spawn_persist_time_range=(args.spawn_persist_min, args.spawn_persist_max),
            allowed_objects=args.allowed_objects,
            host=args.host,
            port=args.port,
            verbose=args.verbose
        )
        recorder.run()

    elif args.command == "convert":
        success = laz_to_obj(
            input_path=args.input,
            output_path=args.output,
            color_by=args.color_by,
            colormap=args.colormap,
            subsample=args.subsample
        )
        if success:
             print("Conversion successful.")
        else:
             sys.exit(1)

    else:
        parser.print_help()
