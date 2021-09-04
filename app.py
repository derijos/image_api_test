'''
@author - Dereck Jos
Image API
json format - {"image":"bse64"}
'''
from flask import Flask, request, jsonify
import base64
import json
import cv2
import numpy as np

app = Flask(__name__)

def image_to_base64(a):
    return [base64.b64encode(a).decode("utf-8"), str(a.dtype), a.shape]

def base64_to_image(a):
    (a, dtype, shape) = a[:3]
    a = bytes(a, 'utf-8')
    a = np.frombuffer(base64.decodebytes(a), dtype=dtype).reshape(shape)
    return a

@app.route("/", methods=["GET", "POST"])
def home():
    return "Image API"



@app.route("/model_api", methods=["POST"])
def api():
    detector = cv2.CascadeClassifier('.\cascades\haarcascade_frontalface_default.xml')
    data = json.loads(request.data)
    # print(data["image"])
    # print(data["image"][2])
    # img_64 = [i for i in data[0]]
    # img_64 = data["image"]
    # print(data["image"])
    decoded_image = base64_to_image(data["image"])
    # print(decoded_image)
    # print(decoded_image)
    gray = cv2.cvtColor(decoded_image, cv2.COLOR_BGR2GRAY)
    faceRects = detector.detectMultiScale(gray, scaleFactor=1.05, minSize=(30,30), minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE)

    for x,y,w,h in faceRects:
        cv2.rectangle(decoded_image, (x,y), (x+w, y+h), (0,255,0), 2)

    encode = image_to_base64(decoded_image)
    # img_64 = base64.b64encode(gray).decode('utf-8')
    return jsonify({"image":encode})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')









# image = cv2.imread("abc.jpg")
# b = base64.b64encode(image).decode("utf-8")
#
# de = base64.b64decode((b))
# image = cv2.imread("abc.jpg")
# cv2.imshow("hj", image)
# cv2.waitKey(0)
#detector = cv2.CascadeClassifier('cascades\haarcascade_frontalface_default.xml')
