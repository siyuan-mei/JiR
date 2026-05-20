import pathlib
from omegaconf import OmegaConf


def load_config(config_name: str):
    config_filename = config_name + ".yaml"
    return OmegaConf.load(pathlib.Path(__file__).parent.resolve() / config_filename)


default_config = load_config("default_config")


def load_and_merge_config(config_name: str):
    _default_config = OmegaConf.create(default_config)
    loaded_config = load_config(config_name)
    return OmegaConf.merge(_default_config, loaded_config)