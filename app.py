from typing import Union
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uuid
from io import BytesIO
import numpy as np
import cv2
import os
import tensorflow as tf

from model import model
from infer import infer_im
from download_ckpt import load_ckpt


load_ckpt()
m = model()
m.load_weights("/tmp/ckpt/ckpt")


print("-" * 50)
if len(tf.config.list_physical_devices("GPU")):
    print("GPU acceleration is ON")
else:
    print("No GPU acceleration! Please install CUDA to speed up" " inference massively")
print("-" * 50)

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/model-info/")
def read_item():
    return [l.__str__ for l in m.layers]


def list_to_dict(list):
    order = ["tlx", "tly", "trx", "try", "brx", "bry", "blx", "bly"]
    return {name: value for (name, value) in zip(order, list)}


@app.post("/image/")
async def upload_file(image: UploadFile):
    contents = await image.read()
    nparr = np.fromstring(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    boxes, _, _, timer = infer_im(m, image)

    print(f"Did it in {timer} time")
    print(np.shape(boxes))

    return {"array": boxes.tolist()}
