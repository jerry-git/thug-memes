from collections import namedtuple
from math import (atan2, degrees, sqrt, hypot)

Point = namedtuple('Point', 'x y')


class ThugLandmarks:
    def __init__(self, l_eye, r_eye, mouth=None):
        self._l_eye = self._ensure_point_or_none(l_eye)
        self._r_eye = self._ensure_point_or_none(r_eye)
        self._mouth = self._ensure_point_or_none(mouth)
        # require both eyes (I know, a bit discriminatory)
        if self._l_eye:
            assert self._r_eye
        if self._r_eye:
            assert self._l_eye

    @property
    def eyes_available(self):
        return self.left_eye and self.right_eye

    @property
    def mouth_available(self):
        return self.mouth is not None

    @property
    def left_eye(self):
        return self._l_eye

    @property
    def right_eye(self):
        return self._r_eye

    @property
    def eyes_distance(self):
        return hypot(self.eyes_x_difference, self.eyes_y_difference)

    @property
    def eyes_x_difference(self):
        return self.right_eye.x - self.left_eye.x

    @property
    def eyes_y_difference(self):
        return self.right_eye.y - self.left_eye.y

    @property
    def eyes_center(self):
        x = self.left_eye.x + self.eyes_x_difference / 2
        y = self.left_eye.y + self.eyes_y_difference / 2
        return Point(x, y)

    @property
    def mouth(self):
        return self._mouth

    @property
    def eyes_angle(self):
        return degrees(atan2(self.eyes_y_difference, self.eyes_x_difference))

    @staticmethod
    def _ensure_point_or_none(obj):
        if not obj:
            return None
        if not isinstance(obj, Point):
            if not len(obj) == 2:
                raise ValueError('Can not convert to Point: {}'.format(obj))
            return Point(obj[0], obj[1])
        return obj
