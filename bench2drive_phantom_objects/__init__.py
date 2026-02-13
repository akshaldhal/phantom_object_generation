__version__ = "0.1.0"

_LAZY_IMPORTS = {
    "download_bech2drive_dataset": (".download_dataset", "download_bech2drive_dataset"),
    "DatasetSize": (".download_dataset", "DatasetSize"),
    "LidarRecorder": (".lidar_recorder", "LidarRecorder"),
    "laz_to_obj": (".laz_to_obj", "laz_to_obj"),
    "laz_dir_to_obj": (".laz_to_obj", "laz_dir_to_obj"),
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path, __name__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_LAZY_IMPORTS.keys())