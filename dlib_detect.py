"""
Dlib Face Detection Library
"""
import cv2
import os
import dlib
import numpy as np
from utils import imutil


class DlibDetect:

    def __init__(self, model_path):
        shape_predictor_path = os.path.realpath(model_path)
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('resources/shape_predictor_68_face_landmarks.dat')

    @property
    def _detector_name(self):
        return "dlib"

    def detect_face(self, img,
                    normalize=False, find_largest=True,
                    skip_multi=False, find_landmarks=True):
        """
        :return: bboxes : [N, 4] = [N, [top(y_min), left(x_min), bottom(y_max), right(x_max)]
        :return: landmarks : [N, [LM, 2]] = [N, [LM, [top (y), left (x)]]]
        """
        bboxes = self.detector(img, 1)

        if skip_multi and len(bboxes) > 1:
            return [], []

        if find_largest and len(bboxes) > 1:
            bboxes = [(max(bboxes, key=lambda rect: rect.width() * rect.height()))]

        landmarks = []
        if find_landmarks:
            for bbox in bboxes:
                points = self.find_landmarks(img, bbox)
                landmarks.append(points)

        bboxes = np.array(list(map(lambda bbox: [bbox.top(), bbox.left(), bbox.bottom(), bbox.right()], bboxes)))

        if bboxes is not None and len(bboxes > 0):
            if normalize:
                bboxes = imutil.normalize_bboxes(img, bboxes)
                if len(landmarks) > 0:
                    landmarks = imutil.normalize_coords(img, landmarks)

        return bboxes, landmarks

    def find_landmarks(self, img, bbox):
        """
        Find the landmarks of a face.
        :param img: RGB image to process. Shape: (height, width, 3)
        :type img: numpy.ndarray
        :param bbox: Bounding box around the face to find landmarks for.
        :type bbox: dlib.rectangle
        :return: Detected landmark locations.
        :rtype: list of (y,x) tuples
        """
        assert img is not None
        assert bbox is not None

        if isinstance(bbox, np.ndarray):
            bbox = dlib.rectangle(bbox[1], bbox[0], bbox[3], bbox[2])

        points = self.predictor(img, bbox)
        return list(map(lambda p: (p.y, p.x), points.parts()))


if __name__ == '__main__':
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        a= DlibDetect("resources/shape_predictor_68_face_landmarks.dat")
        bboxes, landmarks =a.detect_face(frame)
        print(bboxes)
        print(landmarks)



