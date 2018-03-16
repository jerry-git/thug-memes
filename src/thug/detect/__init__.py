from os import path as osp

DATA_FOLDER = osp.join(osp.dirname(__file__), 'data')
FACE_CASCADE_FILE = osp.join(DATA_FOLDER,
                             'haarcascade_frontalface_default.xml')
EYE_CASCADE_FILE = osp.join(DATA_FOLDER, 'haarcascade_eye.xml')
MOUTH_CASCADE_FILE = osp.join(DATA_FOLDER, 'haarcascade_mouth.xml')
DLIB_68_LANDMARKS_MODEL_FILE = osp.join(
    DATA_FOLDER, 'shape_predictor_68_face_landmarks.dat')
