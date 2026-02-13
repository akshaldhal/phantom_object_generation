# import _typeshed._type_checker_internals
from enum import Enum
from huggingface_hub import snapshot_download
import os
import requests
import json
import hashlib
import tarfile

class DatasetSize(Enum):
    MINI = "mini"
    BASE = "base"
    FULL = "full"

def validate_dataset(dataset_dir: str, json_file: str) -> bool:
    with open(json_file, "r") as f:
        data = json.load(f)
    for file_name in data.keys():
        file_path = os.path.join(dataset_dir, file_name)
        if not os.path.exists(file_path):
            print(f"File {file_name} not found in dataset directory.")
            return False
        # Verify file size
        expected_size = data[file_name]["size"]
        actual_size = os.path.getsize(file_path)
        if expected_size != actual_size:
            print(f"File {file_name} size mismatch: expected {expected_size}, got {actual_size}.")
            return False
        # Verify sha256 checksum
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        actual_sha256 = sha256_hash.hexdigest()
        expected_sha256 = data[file_name]["sha256"]
        if expected_sha256 != actual_sha256:
            print(f"File {file_name} sha256 mismatch: expected {expected_sha256}, got {actual_sha256}.")
            return False
    # Unzip files only after validation
    for file_name in data.keys():
        file_path = os.path.join(dataset_dir, file_name)
        if file_name.endswith(".tar.gz"):
            with tarfile.open(file_path, "r:gz") as tar:
                tar.extractall(path=dataset_dir)
    return True

def download_bech2drive_dataset(local_dir: str = "data/Bench2Drive", size: DatasetSize = DatasetSize.MINI):
    if (size == DatasetSize.BASE):
        json_link = "https://raw.githubusercontent.com/Thinklab-SJTU/Bench2Drive/main/docs/bench2drive_base_1000.json"
        json_file_name = "bench2drive_base.json"
        local_dir = local_dir + "-base"
        os.makedirs(local_dir, exist_ok=True)
        repo_id = "rethinklab/Bench2Drive"
        print(f"Downloading base Bench2Drive dataset into {local_dir} ...")
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
        )
        response = requests.get(json_link)
        with open(os.path.join(local_dir, json_file_name), "wb") as f:
            f.write(response.content)
    elif (size == DatasetSize.FULL):
        json_link = "https://raw.githubusercontent.com/Thinklab-SJTU/Bench2Drive/main/docs/bench2drive_full+sup_13638.json"
        json_file_name = "bench2drive_full.json"
        local_dir = local_dir + "-full"
        os.makedirs(local_dir, exist_ok=True)
        repo_id = "rethinklab/Bench2Drive-Full"
        print(f"Downloading base Bench2Drive-Full dataset into {local_dir} ...")
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
        )
        response = requests.get(json_link)
        with open(os.path.join(local_dir, json_file_name), "wb") as f:
            f.write(response.content)
    else:
        json_link = "https://raw.githubusercontent.com/Thinklab-SJTU/Bench2Drive/main/docs/bench2drive_mini_10.json"
        json_file_name = "bench2drive_mini.json"
        local_dir = local_dir + "-mini"
        repo_id = "rethinklab/Bench2Drive"
        os.makedirs(local_dir, exist_ok=True)
        print(f"Downloading mini Bench2Drive dataset into {local_dir} ...")
        mini_files = [
            "HardBreakRoute_Town01_Route30_Weather3.tar.gz",
            "DynamicObjectCrossing_Town02_Route13_Weather6.tar.gz",
            "Accident_Town03_Route156_Weather0.tar.gz",
            "YieldToEmergencyVehicle_Town04_Route165_Weather7.tar.gz",
            "ConstructionObstacle_Town05_Route68_Weather8.tar.gz",
            "ParkedObstacle_Town10HD_Route371_Weather7.tar.gz",
            "ControlLoss_Town11_Route401_Weather11.tar.gz",
            "AccidentTwoWays_Town12_Route1444_Weather0.tar.gz",
            "OppositeVehicleTakingPriority_Town13_Route600_Weather2.tar.gz",
            "VehicleTurningRoute_Town15_Route443_Weather1.tar.gz"
        ]
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            allow_patterns=[f"*{f}" for f in mini_files]
        )
        response = requests.get(json_link)
        with open(os.path.join(local_dir, json_file_name), "wb") as f:
            f.write(response.content)
    if (validate_dataset(local_dir, os.path.join(local_dir, json_file_name))):
        print("Dataset validation successful.")
    else:
        raise ValueError("Dataset validation failed.")

if __name__ == "__main__":
    download_bech2drive_dataset()