from enum import Enum
from os import (getcwd, path as osp)
import sys

import click
import cv2

from .conf import (get_config, print_config)
from .detect.opencv import HaarCascadeDetector
from .meme.basic import Meme
from .meme.thug import ThugMeme

MEME_RESULT_DIR = getcwd()

CONTEXT = dict(help_option_names=['-h', '--help'])


class Detector(Enum):
    OPEN_CV = 'opencv'
    DLIB = 'dlib'


_common_decorators = [
    click.version_option(None, '-v', '--version'),
    click.argument('fpath', type=click.Path(exists=True)),
    click.argument('txt1'),
    click.argument('txt2'),
    click.option(
        '--override',
        '-o',
        type=(str, str),
        multiple=True,
        help='Override any configuration option: <option_name> <new_value>.'),
    click.option(
        '--show-config',
        is_flag=True,
        help='Show the configuration and exit. Takes into account -o options.')
]


def add_decorators(decorators):
    def _add_decorators(func):
        for decorator in reversed(decorators):
            func = decorator(func)
        return func

    return _add_decorators


def _load_configuration(override, show_and_exit):
    conf = get_config(overrides=override)
    if show_and_exit:
        print_config(conf)
        sys.exit(0)

    return conf


def _form_result_path(orig_path, result_dir, fname_extra=''):
    fname = osp.basename(orig_path)
    base, extension = fname.split('.')
    fname = '{}{}.{}'.format(base, fname_extra, extension)
    return osp.join(result_dir, fname)


@click.command(context_settings=CONTEXT)
@add_decorators(_common_decorators)
def meme(fpath, txt1, txt2, override, show_config):
    """Generate a normal meme."""

    conf = _load_configuration(override, show_config)
    res_path = _form_result_path(
        orig_path=osp.abspath(fpath),
        result_dir=MEME_RESULT_DIR,
        fname_extra=conf['meme']['meme_result_name_add'])
    meme = Meme(config=conf['meme'], img_path=fpath, txt1=txt1, txt2=txt2)
    meme.create(res_file=res_path)


@click.command(context_settings=CONTEXT)
@add_decorators(_common_decorators)
@click.option(
    '--debug',
    is_flag=True,
    help='Show debug information (e.g. the detection results img)')
@click.option(
    '--detector',
    type=click.Choice([Detector.OPEN_CV.value, Detector.DLIB.value]),
    default=Detector.OPEN_CV.value,
    help='Detector to use for finding faces and landmarks.')
def thug_meme(fpath, txt1, txt2, override, show_config, debug, detector):
    """Generate an awesome thug meme."""
    fpath = osp.abspath(fpath)
    conf = _load_configuration(override, show_config)
    res_path = _form_result_path(
        orig_path=fpath,
        result_dir=MEME_RESULT_DIR,
        fname_extra=conf['meme']['thug_result_name_add'])

    if detector == Detector.OPEN_CV.value:
        detector = HaarCascadeDetector(config=conf['detect'])
    elif detector == Detector.DLIB.value:
        from .detect.dlib import DlibDetector
        detector = DlibDetector(config=conf['detect'])

    thugs = detector.find_thug_landmarks(
        img_path=osp.abspath(fpath), show_result=debug)

    meme = ThugMeme(
        config=conf['meme'],
        thug_landmarks=thugs,
        img_path=fpath,
        txt1=txt1,
        txt2=txt2)
    meme.create(res_path)

    if debug:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
