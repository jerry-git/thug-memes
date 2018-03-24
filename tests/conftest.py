from os import path as osp


def img_path(fname):
    return osp.abspath(osp.join(IMG_DIR, fname))


IMG_DIR = osp.join(osp.dirname(__file__), 'data', 'img')
IMG_1_FACE = img_path('1_face.jpg')
IMG_3_FACES = img_path('3_faces.jpeg')
IMG_CIGAR = img_path('cigar.png')
IMG_GLASSES = img_path('glasses.png')