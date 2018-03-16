import json

import cv2


class Meme:
    def __init__(self, *, config, img_path, txt1, txt2):
        self._text_width_coverage = float(config['text_width_coverage'])
        self._img_path = img_path
        self._txt1 = txt1
        self._txt2 = txt2
        self._font_color = tuple(json.loads(config['font_bgr']))
        self._tmp_result = None
        self._font = int(config['opencv_font'])
        self._font_thickness = int(config['opencv_font_thickness'])
        self._txt_y_offset = float(config['text_y_offset'])
        self._line_type = int(config['opencv_font_line_type'])

    def create(self, res_file, show=True):
        img = cv2.imread(self._img_path)
        img_h, img_w, _ = img.shape

        if self._txt1 or self._txt2:
            txt = self._get_longest_txt()
            scale = self._select_font_scale(img_w, txt)
            loc1, loc2 = self._get_txt_locations(img_w, img_h, scale)

            if self._txt1:
                cv2.putText(img, self._txt1, loc1, self._font, scale,
                            self._font_color, self._font_thickness,
                            self._line_type)
            if self._txt2:
                cv2.putText(img, self._txt2, loc2, self._font, scale,
                            self._font_color, self._font_thickness,
                            self._line_type)

        cv2.imwrite(res_file, img)

        if show:
            from thug.detect.debug import timestamp
            cv2.imshow('meme-{}'.format(timestamp()), img)
            cv2.waitKey(1)

    def _get_longest_txt(self):
        def txt_width(txt):
            return cv2.getTextSize(txt, self._font, 1,
                                   self._font_thickness)[0][0]

        return max((self._txt1, self._txt2), key=txt_width)

    def _select_font_scale(self, img_w, txt):
        scale = 0.5
        step = 0.05
        (w, _), _ = cv2.getTextSize(txt, self._font, scale,
                                    self._font_thickness)
        while w < img_w * self._text_width_coverage:
            scale += step
            (w, _), _ = cv2.getTextSize(txt, self._font, scale,
                                        self._font_thickness)
        return scale

    def _get_txt_locations(self, img_w, img_h, font_scale):
        (txt1_w, txt1_h), _ = cv2.getTextSize(self._txt1, self._font,
                                              font_scale, self._font_thickness)
        (txt2_w, txt2_h), _ = cv2.getTextSize(self._txt2, self._font,
                                              font_scale, self._font_thickness)
        x1 = (img_w - txt1_w) / 2
        x2 = (img_w - txt2_w) / 2
        y_offset = self._txt_y_offset * img_h
        y1 = y_offset + txt1_h
        y2 = img_h - y_offset
        return (int(x1), int(y1)), (int(x2), int(y2))
