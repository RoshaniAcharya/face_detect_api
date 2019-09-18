import urllib
import numpy as np
import cv2


# METHOD #1: OpenCV, NumPy, and urllib
def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urllib.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # return the image
    return image

if __name__ == '__main__':
    img = "https://www.facebook.com/photo.php?fbid=2634989646530950&set=a.205583316138274&type=3&theater"
    new_img = url_to_image(img)
    print("new img", new_img)