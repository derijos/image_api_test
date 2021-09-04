import cv2
import base64
import requests
import numpy as np
import json

camera = cv2.VideoCapture(1)
def image_to_base64(a):
    return [base64.b64encode(a).decode("utf-8"), str(a.dtype), a.shape]

def base64_to_image(a):
    (a, dtype, shape) = a[:3]
    a = bytes(a, 'utf-8')
    a = np.frombuffer(base64.decodebytes(a), dtype=dtype).reshape(shape)
    return a

while True:
    grabbed, frame = camera.read()

    if not grabbed:
        break

    img = image_to_base64(frame)
    # print(type(img))
    # print(img)
    # data = {"image" : img}
    img_64 = requests.post(url="http://127.0.0.1:5000/model_api", json={"image":img}).json()
    image = base64_to_image(img_64["image"])

    # print(image)
    # print(img_64)
    # image = base64_to_image(img_64["image"])

    cv2.imshow("frame", image)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()




# image = cv2.imread("abc.jpg")
# a = image_to_base64(image)
# a = base64_to_image(a)
# cv2.imshow("qa",a)
# cv2.waitKey(0)