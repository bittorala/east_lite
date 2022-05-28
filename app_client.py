import requests
import cv2
import numpy as np
ENDPOINT_URL = 'http://127.0.0.1:5000/infer'

image = cv2.imread('/home/bittor/data/ocr/test/img_1.jpg')[:,:,::-1]
data = {'image': image.tolist() }
response = requests.post(ENDPOINT_URL, json = data)
response.raise_for_status()
print(response.content)