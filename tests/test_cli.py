from unittest.mock import (patch, MagicMock)

from click.testing import CliRunner
import pytest

from thug.cli import meme, thug_meme
from .conftest import IMG_1_FACE


@pytest.fixture
def cli_runner():
    return CliRunner()


@patch('thug.cli.Meme')
class TestMeme:
    def test_it_creates_meme(self, meme_cls, cli_runner):
        res = cli_runner.invoke(
            meme, [IMG_1_FACE, 'my awesome', 'test meme'],
            catch_exceptions=False)

        assert res.exit_code == 0
        assert meme_cls.return_value.create.call_count == 1

        _, meme_kwargs = meme_cls.call_args
        assert meme_kwargs['txt1'] == 'my awesome'
        assert meme_kwargs['txt2'] == 'test meme'


@patch('thug.cli.ThugMeme')
class TestThug:
    @pytest.mark.parametrize('detector_arg, detector',
                             [('opencv', 'thug.cli.HaarCascadeDetector'),
                              ('dlib', 'thug.detect.dlib.DlibDetector')])
    def test_it_creates_thug_meme(self, meme_cls, cli_runner, detector_arg,
                                  detector):
        with patch(detector) as detector:
            detector = detector.return_value
            thugs = MagicMock()
            detector.find_thug_landmarks.return_value = thugs

            res = cli_runner.invoke(
                thug_meme, [
                    IMG_1_FACE, 'my awesome', 'thug meme', '--detector',
                    detector_arg
                ],
                catch_exceptions=False)

            assert res.exit_code == 0
            assert detector.find_thug_landmarks.call_count == 1
            assert meme_cls.return_value.create.call_count == 1

            _, meme_kwargs = meme_cls.call_args
            assert meme_kwargs['thug_landmarks'] == thugs
            assert meme_kwargs['txt1'] == 'my awesome'
            assert meme_kwargs['txt2'] == 'thug meme'

    def test_it_does_not_accept_unknown_detector(self, meme_cls, cli_runner):
        res = cli_runner.invoke(
            thug_meme,
            [IMG_1_FACE, 'my awesome', 'thug meme', '--detector', 'unknown'],
            catch_exceptions=False)
        assert res.exit_code == 2
        assert not meme_cls.return_value.create.call_count


@pytest.mark.parametrize('meme_cls, command',
                         [('thug.cli.Meme', meme),
                          ('thug.cli.ThugMeme', thug_meme)])
class TestCommonCliArgs:
    def test_it_does_not_create_with_invalid_path(self, cli_runner, meme_cls,
                                                  command):
        fpath = 'no-meme-for-you'

        with patch(meme_cls) as meme_cls:
            res = cli_runner.invoke(
                command, [fpath, '', ''], catch_exceptions=False)

            assert res.exit_code == 2
            assert not meme_cls.return_value.create.call_count

    @patch('thug.detect.dlib.DlibDetector')
    @patch('thug.cli.HaarCascadeDetector')
    def test_it_takes_into_account_config_overrides(
            self, detector1, detector2, cli_runner, meme_cls, command):
        o1 = ['-o', 'cigar_length', '0.123']
        o2 = ['-o', 'glasses_width', '0.321']
        o3 = ['-o', 'uknown', 'not going to be used']
        args = [IMG_1_FACE, '', ''] + o1 + o2 + o3

        with patch(meme_cls) as meme_cls:
            res = cli_runner.invoke(command, args, catch_exceptions=False)

            assert res.exit_code == 0

            _, kwargs = meme_cls.call_args
            conf = kwargs['config']
            assert conf['cigar_length'] == '0.123'
            assert conf['glasses_width'] == '0.321'
            assert 'unknown' not in conf
