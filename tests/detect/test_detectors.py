"""Common tests for all detectors"""

from collections import namedtuple
from unittest.mock import patch

import pytest

from thug.conf import get_config
from thug.detect.opencv import HaarCascadeDetector
from thug.detect.dlib import DlibDetector
from ..conftest import (IMG_1_FACE, IMG_3_FACES)

Tolerance = namedtuple('Tolerance', 'min max')
_TestCase = namedtuple('TestCase', 'input_file tolerances')


class LocationTolerances:
    def __init__(self,
                 left_eye_x=None,
                 left_eye_y=None,
                 right_eye_x=None,
                 right_eye_y=None,
                 mouth_x=None,
                 mouth_y=None):
        self.left_eye_x = left_eye_x
        self.left_eye_y = left_eye_y
        self.right_eye_x = right_eye_x
        self.right_eye_y = right_eye_y
        self.mouth_x = mouth_x
        self.mouth_y = mouth_y


def _test_cases():
    TOLERANCES_1 = [
        LocationTolerances(
            left_eye_x=Tolerance(188, 195),
            left_eye_y=Tolerance(320, 327),
            right_eye_x=Tolerance(282, 289),
            right_eye_y=Tolerance(299, 305),
            mouth_x=Tolerance(270, 282),
            mouth_y=Tolerance(420, 430))
    ]

    # Ordering left to right
    TOLERANCES_2 = [
        LocationTolerances(
            left_eye_x=Tolerance(385, 395),
            left_eye_y=Tolerance(346, 356),
            right_eye_x=Tolerance(436, 446),
            right_eye_y=Tolerance(350, 360),
            mouth_x=Tolerance(405, 415),
            mouth_y=Tolerance(400, 410)),
        LocationTolerances(
            mouth_x=Tolerance(872, 882), mouth_y=Tolerance(282, 292)),
        LocationTolerances(
            mouth_x=Tolerance(1227, 1237), mouth_y=Tolerance(455, 465))
    ]

    case1 = _TestCase(input_file=IMG_1_FACE, tolerances=TOLERANCES_1)
    case2 = _TestCase(input_file=IMG_3_FACES, tolerances=TOLERANCES_2)

    for case in [case1, case2]:
        yield case


class TestFindThugLandMarks:
    @pytest.mark.parametrize('test_case', _test_cases())
    @pytest.mark.parametrize('detector', [HaarCascadeDetector, DlibDetector])
    def test_it_returns_landmarks_within_tolerances(self, detector, test_case):
        conf = get_config()['detect']  # Default config
        d = detector(config=conf)
        res = d.find_thug_landmarks(
            img_path=test_case.input_file, show_result=False)

        # verify face count
        assert len(res) == len(test_case.tolerances)

        # sort from left to right
        sorted_res = sorted(res, key=lambda l: l.left_eye.x)

        for result, t in zip(sorted_res, test_case.tolerances):
            if t.mouth_x:
                assert t.mouth_x.min <= result.mouth.x <= t.mouth_x.max
            if t.mouth_y:
                assert t.mouth_y.min <= result.mouth.y <= t.mouth_y.max

            if t.left_eye_x:
                assert t.left_eye_x.min <= result.left_eye.x <= t.left_eye_x.max
            if t.left_eye_y:
                assert t.left_eye_y.min <= result.left_eye.y <= t.left_eye_y.max

            if t.right_eye_x:
                assert t.right_eye_x.min <= result.right_eye.x <= t.right_eye_x.max
            if t.left_eye_y:
                assert t.right_eye_y.min <= result.right_eye.y <= t.right_eye_y.max

    @pytest.mark.parametrize(
        'detector, imshow_path',
        [(HaarCascadeDetector, 'thug.detect.opencv.cv2.imshow'),
         (DlibDetector, 'thug.detect.dlib.cv2.imshow')])
    def test_it_shows_results(self, detector, imshow_path):
        conf = get_config()['detect']
        d = detector(config=conf)
        with patch(imshow_path) as imshow:
            d.find_thug_landmarks(img_path=IMG_1_FACE, show_result=True)
            assert imshow.call_count
