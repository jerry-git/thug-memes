from unittest.mock import (patch, MagicMock)

import cv2
import pytest

from thug.detect.result import ThugLandmarks
from thug.meme.thug import (ThugMeme, ThugError)
from thug.conf import get_config
from ..conftest import (IMG_3_FACES, IMG_GLASSES)

DEFAULT_KWARGS = {
    'config': get_config()['meme'],
    'txt1': 'upper txt',
    'txt2': 'lower txt',
    'img_path': IMG_3_FACES
}


@pytest.fixture
def thug1():
    thug = ThugLandmarks(l_eye=(100, 100), r_eye=(150, 100), mouth=(125, 300))
    return thug


@pytest.fixture
def thug2():
    thug = ThugLandmarks(l_eye=(250, 125), r_eye=(300, 125), mouth=(275, 350))
    return thug


@pytest.fixture
def thug_meme(thug1, thug2):
    meme = ThugMeme(thug_landmarks=[thug1, thug2], **DEFAULT_KWARGS)
    return meme


@patch('thug.meme.basic.cv2.imwrite')
@patch('thug.meme.basic.Meme.create')
@patch('thug.meme.thug.ThugMeme._draw_glasses')
@patch('thug.meme.thug.ThugMeme._draw_cigar')
class TestCreate:
    def test_it_draws_cigar_and_glasses_for_all_thugs(
            self, draw_cigar, draw_glasses, base_cls_create, imwrite,
            thug_meme):

        thug_meme.create(res_file='test', show=False)

        assert draw_glasses.call_count == 2
        assert draw_cigar.call_count == 2
        assert base_cls_create.call_count == 1

    def test_it_does_not_crash_if_thug_error_is_raised(
            self, draw_cigar, draw_glasses, base_cls_create, imwrite,
            thug_meme):
        draw_glasses.side_effect = ThugError(
            'sorry memer, problem with glasses')
        draw_cigar.side_effect = ThugError('sorry memer, problem with cigar')
        thug_meme.create(res_file='test', show=False)

        assert draw_glasses.call_count == 2
        assert draw_cigar.call_count == 2
        assert base_cls_create.call_count == 1


class TestDrawCigar:
    @patch('thug.meme.thug.ThugMeme._draw_on_top')
    def test_it_draws(self, draw_on_top, thug_meme, thug1):
        img = cv2.imread(IMG_3_FACES)
        thug_meme._draw_cigar(img, thug1)

        assert draw_on_top.call_count == 1
        args, _ = draw_on_top.call_args
        assert (args[0] == img).all()


class TestDrawGlasses:
    @patch('thug.meme.thug.ThugMeme._draw_on_top')
    def test_it_draws(self, draw_on_top, thug_meme, thug1):
        img = cv2.imread(IMG_3_FACES)
        thug_meme._draw_glasses(img, thug1)

        assert draw_on_top.call_count == 1
        args, _ = draw_on_top.call_args
        assert (args[0] == img).all()


class TestDrawOnTop:
    def test_it_raises_thug_error_if_sub_img_does_not_fit(self, thug_meme):
        img = cv2.imread(IMG_3_FACES)
        img_h, img_w, _ = img.shape
        glasses = cv2.imread(IMG_GLASSES, -1)
        with pytest.raises(ThugError):
            # try to draw the glasses to the right edge of the img
            thug_meme._draw_on_top(
                img=img, x=img_w, y=0, sub_img=glasses, sub_name='test')
