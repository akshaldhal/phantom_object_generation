import carla
from download_dataset import download_bech2drive_dataset

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    available_maps = client.get_available_maps()
    print(available_maps)
    print(len(available_maps))



if __name__ == "__main__":
    main()
