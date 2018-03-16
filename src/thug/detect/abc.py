from abc import (ABC, abstractmethod)

import cv2


class Detector(ABC):
    def __init__(self, config, *args, **kwargs):
        self._config = config
        pass

    @property
    def config(self):
        return self._config

    @abstractmethod
    def find_thug_landmarks(self, img_path, show_result=False):
        """
        returns a list of ThugLandmark objects
        """

    @staticmethod
    def _draw_rectangle(img, rect, bgr, thickness=2):
        x, y, w, h = rect
        cv2.rectangle(img, (x, y), (x + w, y + h), bgr, thickness)

    @staticmethod
    def _draw_circle(img, center, bgr, radius=5, thickness=2):
        cv2.circle(img, center, radius, bgr, thickness)

    def _draw_face(self, img, location):
        bgr = (255, 0, 0)
        self._draw_rectangle(img, location, bgr)

    def _draw_thug_result(self, img, thug_result):
        bgr_eye = (0, 0, 255)
        bgr_mouth = (0, 0, 255)
        for eye in (thug_result.left_eye, thug_result.right_eye):
            self._draw_circle(img, eye, bgr_eye, thickness=5)
        self._draw_circle(img, thug_result.mouth, bgr_mouth, thickness=5)