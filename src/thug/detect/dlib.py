import cv2
import dlib
import face_recognition_models

from .abc import Detector
from .debug import timestamp  # TODO: get rid off this
from .result import ThugLandmarks

LEFT_EYE_CENTER_IDXS = (37, 40)
RIGHT_EYE_CENTER_IDXS = (43, 46)
MOUTH_CENTER_IDXS = (62, 66)


class DlibDetector(Detector):
    def __init__(self, config, *args, **kwargs):
        self._landmarks_model = face_recognition_models.pose_predictor_model_location(
        )
        super().__init__(config, *args, **kwargs)

    def find_thug_landmarks(self, img_path, show_result=False):
        face_det = dlib.get_frontal_face_detector()

        predictor = dlib.shape_predictor(self._landmarks_model)

        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_det(gray)

        thug_results = []

        for f in faces:
            predicted = predictor(gray, f)
            landmarks = predicted.parts()

            left_eye = self._center_of_two_points(
                landmarks[LEFT_EYE_CENTER_IDXS[0]],
                landmarks[LEFT_EYE_CENTER_IDXS[1]])

            right_eye = self._center_of_two_points(
                landmarks[RIGHT_EYE_CENTER_IDXS[0]],
                landmarks[RIGHT_EYE_CENTER_IDXS[1]])

            mouth = self._center_of_two_points(landmarks[MOUTH_CENTER_IDXS[0]],
                                               landmarks[MOUTH_CENTER_IDXS[1]])

            thug = ThugLandmarks(l_eye=left_eye, r_eye=right_eye, mouth=mouth)
            thug_results.append(thug)

            if show_result:
                self._draw_result(img, f, landmarks, thug)

        if show_result:
            cv2.imshow('detection_result-{}'.format(timestamp()), img)
            cv2.waitKey(1)

        return thug_results

    def _draw_result(self, img, face, landmarks, thug_result):
        for point in landmarks:
            self._draw_circle(img, (point.x, point.y), bgr=(0, 255, 255))
        self._draw_thug_result(img, thug_result)
        self._draw_face(img, self.dlib_rect_to_xywh(face))

    @staticmethod
    def _center_of_two_points(point1, point2):
        x = int(point1.x + (point2.x - point1.x) / 2)
        y = int(point1.y + (point2.y - point1.y) / 2)
        return x, y

    @staticmethod
    def dlib_rect_to_xywh(r):
        x, y = r.left(), r.top()
        w, h = r.right() - x, r.bottom() - y
        return x, y, w, h
