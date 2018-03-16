from configparser import ConfigParser
import logging
from os import (path as osp, environ as osenv)

DEFAULT_CONFIG = osp.join(osp.dirname(__file__), 'default.conf')
LOGGER = logging.getLogger(__name__)


def print_config(conf):
    from pprint import pprint
    for section in conf.sections():
        section_format = {section: dict(conf.items(section))}
        pprint(section_format)


def get_config(overrides=None):
    conf = ConfigParser()
    files = [DEFAULT_CONFIG]
    user_file = osenv.get('THUG_CONF', None)
    if user_file:
        files.append(user_file)
    conf.read(files)

    if overrides:
        overrides = dict(overrides)
        for section in conf.sections():
            for k in (dict(conf.items(section))):
                if k in overrides:
                    conf[section][k] = overrides.pop(k)

        if overrides:
            LOGGER.warning(
                'Unkown configuration override parameters: {}'.format(
                    ', '.join([str(k) for k, v in overrides.items()])))

    return conf
