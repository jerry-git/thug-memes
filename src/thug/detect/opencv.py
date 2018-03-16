import cv2
import numpy as np

from .abc import Detector
from .debug import timestamp
from .result import ThugLandmarks
from . import (FACE_CASCADE_FILE, EYE_CASCADE_FILE, MOUTH_CASCADE_FILE)


class HaarCascadeDetector(Detector):
    def __init__(self, config, *args, **kwargs):
        self._face_cascade = cv2.CascadeClassifier(FACE_CASCADE_FILE)
        self._eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_FILE)
        self._mouth_cascade = cv2.CascadeClassifier(MOUTH_CASCADE_FILE)

        self._min_mouth_w = float(config['opencv_min_mouth_w'])
        self._max_mouth_w = float(config['opencv_max_mouth_w'])
        self._min_mouth_h = float(config['opencv_min_mouth_h'])
        self._max_mouth_h = float(config['opencv_max_mouth_h'])

        self._face_scale_factor = float(config['opencv_face_scale_factor'])
        self._face_min_neighbors = int(config['opencv_face_min_neighbors'])
        self._eye_scale_factor = float(config['opencv_eye_scale_factor'])
        self._eye_min_neighbors = int(config['opencv_eye_min_neighbors'])
        self._mouth_scale_factor = float(config['opencv_mouth_scale_factor'])
        self._mouth_min_neighbors = int(config['opencv_mouth_min_neighbors'])

        self._min_eye_w = float(config['opencv_min_eye_w'])
        self._max_eye_w = float(config['opencv_max_eye_w'])
        self._min_eye_h = float(config['opencv_min_eye_h'])
        self._max_eye_h = float(config['opencv_max_eye_h'])

        self._fallback_mouth_x = float(config['opencv_fallback_mouth_x'])
        self._fallback_mouth_y = float(config['opencv_fallback_mouth_y'])

        self._fallback_left_eye_x = float(config['opencv_fallback_left_eye_x'])
        self._fallback_left_eye_y = float(config['opencv_fallback_left_eye_y'])
        self._fallback_right_eye_x = float(
            config['opencv_fallback_right_eye_x'])
        self._fallback_right_eye_y = float(
            config['opencv_fallback_right_eye_y'])

        self._mouth_y_offset = float(config['opencv_mouth_y_offset'])

        super().__init__(config, *args, **kwargs)

    def find_thug_landmarks(self, img_path, show_result=False):
        def to_base(rectangle, base):
            rectangle[0] += base[0]
            rectangle[1] += base[1]
            return rectangle

        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self._find_faces(gray)
        faces = self._remove_rects_inside_rects(faces)

        thugs = []
        for face in faces:
            x, y, w, h = face
            base = x, y
            roi_gray = gray[y:y + h, x:x + w]

            eyes, mouth = (None, None), None
            eyes_in_roi, mouth_in_roi = self._find_land_marks(roi_gray)

            if self._values_exist(eyes_in_roi):
                eyes = to_base(eyes_in_roi[0], base), to_base(
                    eyes_in_roi[1], base)

            if self._value_exists(mouth_in_roi):
                mouth = to_base(mouth_in_roi, base)

            thug_result = self._create_thug_detection_result(face, eyes, mouth)

            thugs.append(thug_result)

            if show_result:
                self._draw_result(img, face, eyes, mouth, thug_result)

        if show_result:
            cv2.imshow('detection_result-{}'.format(timestamp()), img)
            cv2.waitKey(1)

        return thugs

    def _find_faces(self, img):
        return self._face_cascade.detectMultiScale(
            img,
            scaleFactor=self._face_scale_factor,
            minNeighbors=self._face_min_neighbors)

    def _find_land_marks(self, face_img_gray):
        eyes = self._find_eyes(face_img_gray)
        mouths = self._find_mouths(face_img_gray)
        return eyes, mouths

    def _find_eyes(self, gray_face_img):
        def find_eye(img):
            results, neighbours, _ = self._eye_cascade.detectMultiScale3(
                img,
                scaleFactor=self._eye_scale_factor,
                minNeighbors=self._eye_min_neighbors,
                minSize=min_size,
                maxSize=max_size,
                outputRejectLevels=True)

            if not self._value_exists(results):
                return None

            eye = results[np.argmax(neighbours)]
            return eye

        h, w = gray_face_img.shape
        min_size = (int(self._min_eye_w * w), int(self._min_eye_h * h))
        max_size = (int(self._max_eye_w * w), int(self._max_eye_h * h))

        half_w = int(w / 2)
        upper_h = int(0.6 * h)
        left_eye_search_area = gray_face_img[:upper_h, :half_w]
        right_eye_search_area = gray_face_img[:upper_h, half_w:]

        left = find_eye(left_eye_search_area)
        right = find_eye(right_eye_search_area)
        if self._value_exists(right):
            right[0] += half_w

        return left, right

    def _find_mouths(self, gray_face_img):
        h, w = gray_face_img.shape
        min_size = (int(self._min_mouth_w * w), int(self._min_mouth_h * h))
        max_size = (int(self._max_mouth_w * w), int(self._max_mouth_h * h))
        half_y = int(h / 2)
        lower_half = gray_face_img[half_y:h]
        results, neighbours, _ = self._mouth_cascade.detectMultiScale3(
            lower_half,
            scaleFactor=self._mouth_scale_factor,
            minNeighbors=self._mouth_min_neighbors,
            minSize=min_size,
            maxSize=max_size,
            outputRejectLevels=True)

        if not self._value_exists(results):
            return None

        best_one = results[np.argmax(neighbours)]
        best_one[1] += half_y

        return best_one

    def _draw_result(self, img, face, eyes, mouth, thug_result):
        self._draw_face(img, face)
        if self._values_exist(eyes):
            self._draw_eyes(img, eyes)
        if self._value_exists(mouth):
            self._draw_mouth(img, mouth)
        self._draw_thug_result(img, thug_result)

    def _draw_eyes(self, img, eyes):
        bgr = (0, 255, 0)
        for eye in eyes:
            self._draw_rectangle(img, eye, bgr)

    def _draw_mouth(self, img, mouth):
        bgr = (0, 0, 255)
        self._draw_rectangle(img, mouth, bgr)

    def _create_thug_detection_result(self, face, eyes, mouth):
        if self._values_exist(eyes):
            # TODO: indexing could be avoided with named tuples
            left_eye = self._calculate_rectange_center(eyes[0])
            right_eye = self._calculate_rectange_center(eyes[1])
        else:
            left_eye, right_eye = self._fallback_thug_result_eyes(face)

        if self._value_exists(mouth):
            mouth = self._calculate_rectange_center(mouth)
            *_, h = face
            mouth = list(mouth)
            mouth[1] += int(self._mouth_y_offset * h)
        else:
            mouth = self._fallback_thug_result_mouth(face)

        return ThugLandmarks(l_eye=left_eye, r_eye=right_eye, mouth=mouth)

    def _fallback_thug_result_eyes(self, face):
        x, y, w, h = face
        left_x = int(self._fallback_left_eye_x * w) + x
        left_y = int(self._fallback_left_eye_y * h) + y
        right_x = int(self._fallback_right_eye_x * w) + x
        right_y = int(self._fallback_right_eye_y * h) + y
        return (left_x, left_y), (right_x, right_y)

    def _fallback_thug_result_mouth(self, face):
        x, y, w, h = face
        mouth_x = int(self._fallback_mouth_x * w) + x
        mouth_y = int(self._fallback_mouth_y * h) + y
        return mouth_x, mouth_y

    @staticmethod
    def _calculate_rectange_center(rect):
        x, y, w, h = rect
        return int(x + w / 2), int(y + h / 2)

    @staticmethod
    def _value_exists(val):
        if isinstance(val, np.ndarray):
            return val.any()
        return val

    def _values_exist(self, vals):
        for val in vals:
            if not self._value_exists(val):
                return False
        return True

    @staticmethod
    def _remove_rects_inside_rects(rects):
        def intersection(a, b):
            x = max(a[0], b[0])
            y = max(a[1], b[1])
            w = min(a[0] + a[2], b[0] + b[2]) - x
            h = min(a[1] + a[3], b[1] + b[3]) - y
            if w < 0 or h < 0:
                return ()
            return (x, y, w, h)

        to_remove = set()
        for idx, r in enumerate(rects):
            *_, w, h = r
            r_area = w * h
            for r_other in rects:
                if not np.array_equal(r, r_other):
                    intersect = intersection(r, r_other)
                    if intersect:
                        *_, wi, hi = intersect
                        intersect_area = wi * hi
                        if intersect_area == r_area:
                            to_remove.add(idx)
        if to_remove:
            rects = np.delete(rects, list(to_remove), 0)

        return rects
