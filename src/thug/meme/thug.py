import json

import cv2
import datetime as dt
import numpy as np
import random

from .basic import Meme
from . import (GLASSES_FILE, CIGAR_FILE)


class ThugMeme(Meme):
    def __init__(self, *, thug_landmarks, config, **kwargs):
        self._glasses_width = float(config['glasses_width'])
        self._glasses_down_shift = float(config['glasses_down_shift'])
        self._glasses_right_shift = float(config['glasses_right_shift'])
        self._cigar_length = float(config['cigar_length'])
        self._cigar_mouth_center_pos = tuple(
            json.loads(config['cigar_mouth_center_pos']))
        self._thugs = thug_landmarks
        super().__init__(config=config, **kwargs)

    def create(self, res_file, show=True):
        img = cv2.imread(self._img_path)

        for thug in self._thugs:
            if thug.eyes_available:
                self._draw_glasses(img, thug)

                if thug.mouth_available:
                    self._draw_cigar(img, thug)  # depends also on eyes

        cv2.imwrite(res_file, img)
        self._img_path = res_file

        return super().create(res_file, show)

    def _draw_glasses(self, img, thug):
        orig_img = cv2.imread(GLASSES_FILE, -1)
        rotated = self._rotate(orig_img, thug.eyes_angle)
        h, w, _ = rotated.shape

        gx, gy, gw, gh = self._calculate_glasses_dimensions(w, h, thug)
        scaled = cv2.resize(rotated, (gw, gh), interpolation=cv2.INTER_AREA)

        self._draw_on_top(img, gx, gy, scaled)

    def _calculate_glasses_dimensions(self, origw, origh, thug):
        w = int(thug.eyes_distance * self._glasses_width)
        h = int(origh * (w / origw))

        center_x, center_y = thug.eyes_center
        y = int(center_y - h / 2 + (self._glasses_down_shift * h))
        x = int(center_x - w / 2 + (self._glasses_right_shift * w))

        return x, y, w, h

    def _rotate(self, img, angle):
        rot_matrix, w, h = self._get_rotation_matrix(img, angle)
        rot_img = cv2.warpAffine(img, rot_matrix, (w, h))
        return rot_img

    def _get_rotation_matrix(self, img, angle):
        orig_h, orig_w, _ = img.shape
        c_x, c_y = orig_w // 2, orig_h // 2

        rot_matrix = cv2.getRotationMatrix2D((c_x, c_y), -angle, 1.0)
        cos = np.abs(rot_matrix[0, 0])
        sin = np.abs(rot_matrix[0, 1])

        w = int((orig_h * sin) + (orig_w * cos))
        h = int((orig_h * cos) + (orig_w * sin))

        rot_matrix[0, 2] += (w / 2) - c_x
        rot_matrix[1, 2] += (h / 2) - c_y

        return rot_matrix, w, h

    def _get_rotated_point(self, img, angle, point):
        rot_matrix, _, _ = self._get_rotation_matrix(img, angle)
        rot_point = np.dot(rot_matrix, [point[0], point[1], 1])
        rot_point = int(rot_point[0]), int(rot_point[1])
        return rot_point

    def _draw_cigar(self, img, thug):
        def rand_angle():
            random.seed(dt.datetime.now())
            return random.randrange(360)

        orig = cv2.imread(CIGAR_FILE, -1)
        orig_h, orig_w, _ = orig.shape
        angle = rand_angle()
        rotated = self._rotate(orig, angle)

        rot_h, rot_w, _ = rotated.shape
        w, h = self._calculate_cigar_wh(rot_w, rot_h, thug)
        scaled_rot = cv2.resize(rotated, (w, h))

        scaled_orig_h = int(orig_h * (h / rot_h))
        scaled_orig_w = int(orig_w * (w / rot_w))
        scaled_orig = cv2.resize(orig, (scaled_orig_w, scaled_orig_h))
        cigar_mouth_center = self._get_rotated_point(
            scaled_orig, angle, self._cigar_mouth_center_pos)
        x = thug.mouth.x - cigar_mouth_center[0]
        y = thug.mouth.y - cigar_mouth_center[1]

        self._draw_on_top(img, x, y, scaled_rot)

    def _calculate_cigar_wh(self, orig_w, orig_h, thug):
        h = int(self._cigar_length * (thug.mouth.y - thug.left_eye.y))
        w = int(orig_w * (h / orig_h))
        return w, h

    def _draw_on_top(self, img, x, y, sub_img):
        # TODO: some intelligent scaling needed here if sub img does not fit
        h, w, _ = sub_img.shape
        mask = sub_img[:, :, 3]
        mask_inv = cv2.bitwise_not(mask)
        sub_img_ = sub_img[:, :, :3]

        background = img[y:y + h, x:x + w]
        background = cv2.bitwise_and(background, background, mask=mask_inv)
        foreground = cv2.bitwise_and(sub_img_, sub_img_, mask=mask)
        sum_ = cv2.add(background, foreground)

        img[y:y + h, x:x + w] = sum_
