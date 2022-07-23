from typing import Union
from fastapi import FastAPI, File, UploadFile
import uuid
from io import BytesIO
import numpy as np
import cv2

from model import model
from infer import infer_im

m = model()
m.load_weights('/tmp/ckpt/ckpt')

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/model-info/")
def read_item():
    return [l.__str__ for l in m.layers]


def list_to_dict(list):
    order = ['tlx', 'tly', 'trx', 'try', 'brx', 'bry', 'blx', 'bly']
    return {name: value for (name,value) in zip(order, list)}

@app.post("/image/")
async def upload_file(image: UploadFile):
    # image.filename = f"{uuid.uuid4()}.jpg"
    # contents = await image.read()

    # with open(f"{image.filename}", "wb") as f:
    #     f.write(contents)
    contents = await image.read()
    nparr = np.fromstring(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    boxes, score, geo, timer = infer_im(m, image)

    print(f"Did it in {timer} time")
    print(np.shape(boxes))

    return {"array": boxes.tolist()}

    result = {'boxes': {f'b_{i}': list_to_dict(list(np.reshape(b, 8))) for (i, b) in enumerate(boxes)}}
    print(result)
    return result
