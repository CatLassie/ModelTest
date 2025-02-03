import configparser
import logging
from dataclasses import dataclass


@dataclass
class Config:
    def __init__(self):
        config = configparser.ConfigParser()
        config.read('conf.ini')

        default_conf = config['DEFAULT']
        self.log_lvl = default_conf['LogLvl']
        self.model_base_path = default_conf['ModelBasePath']

        self.lg = logging.getLogger(__name__)
        self.lg.info(f'{Config.__name__} initialized')

    log_lvl: str
    model_base_path: str
