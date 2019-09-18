
import json
import cv2
from flask import Flask, request
import numpy as np
import urllib.request
from dlib_detect import DlibDetect

app = Flask(__name__, template_folder='../templates')
detector = None


@app.route('/detect', methods=['POST'])
def detect_face():
    """
        API for detecting human face from url image
        :param image: url of image
        :return: boundary_boxes, landmarks, success: True or False and msg
        """
    try:
        image_url = request.form.get('image')

        req = urllib.request.urlopen(image_url)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        image = cv2.imdecode(arr, -1)
        bboxes = []
        success = False
        if image is None:
            msg = "Invalid URL"
        else:
            bboxes, landmarks = detector.detect_face(image)
            if len(bboxes) > 0:
                msg = "Human Face Detected."
                success = True
                bboxes= bboxes.tolist()
            else:
                success = True
                msg = 'Human Face not Detected. '
        img_data = {'bboxes': bboxes, 'success': success, 'msg': msg}
        final_message = json.dumps(img_data)
        return final_message
    except Exception as e:
        img_data = {'bboxes': None, 'success': False, 'msg': str(e)}
        final_message = json.dumps(img_data)
        return final_message


if __name__ == '__main__':
    detector = DlibDetect("resources/shape_predictor_68_face_landmarks.dat")
    app.run(debug=True)
