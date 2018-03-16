from unittest.mock import patch

import pytest

from thug.meme.basic import Meme
from thug.conf import get_config
from ..conftest import IMG_1_FACE


@pytest.fixture
def meme():
    conf = get_config()['meme']  # Default config
    m = Meme(
        config=conf, img_path=IMG_1_FACE, txt1='upper text', txt2='lower text')
    return m


@patch('thug.meme.basic.cv2.waitKey')
@patch('thug.meme.basic.cv2.imshow')
@patch('thug.meme.basic.cv2.imwrite')
class TestCreate:
    @patch('thug.meme.basic.cv2.putText')
    def test_it_writes_texts(self, puttext, imwrite, imshow, waitkey, meme):
        meme.create(res_file='test', show=False)

        assert puttext.call_count == 2
        texts = [args[1] for (args, _) in puttext.call_args_list]
        assert texts[0] == 'upper text'
        assert texts[1] == 'lower text'
        assert not imshow.call_count

    def test_it_shows_result(self, imwrite, imshow, waitkey, meme):
        meme.create(res_file='test', show=True)
        assert imshow.call_count == 1
