import pytest

from thug.detect.result import (ThugLandmarks, Point)


class TestThugLandmarks:
    @pytest.fixture
    def thug_landmarks(self):
        tlm = ThugLandmarks(l_eye=(10, 20), r_eye=(20, 22), mouth=(15, 50))
        return tlm

    def test_eyes_distance(self, thug_landmarks):
        assert int(thug_landmarks.eyes_distance) == 10

    def test_eyes_center(self, thug_landmarks):
        assert thug_landmarks.eyes_center == Point(x=15, y=21)

    def test_eyes_x_difference(self, thug_landmarks):
        assert thug_landmarks.eyes_x_difference == 10

    def test_eyes_y_difference(self, thug_landmarks):
        assert thug_landmarks.eyes_y_difference == 2

    def test_eyes_angle(self, thug_landmarks):
        assert int(thug_landmarks.eyes_angle) == 11

    # yapf:disable
    @pytest.mark.parametrize(
        'l_eye, r_eye, mouth',
        [('str', (20, 22), (15, 50)),
         (None, (20, 22), (15, 50)),
         (10, (20, 22), (15, 50)),
         ((10, 10), None, (15, 50)),
         ((10, 10), (20, 20, 15), (15, 50))])
    # yapf:enable
    def test_init_with_invalid_params(self, l_eye, r_eye, mouth):
        with pytest.raises(Exception):
            ThugLandmarks(l_eye=l_eye, r_eye=r_eye, mouth=mouth)
