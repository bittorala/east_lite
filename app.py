from flask import Flask, request, jsonify, send_file
from config import cfg
from model import model
from utils import draw_boxes
from infer import infer_im
from skimage.io import imsave
import cStringIO as StringIO
import numpy as np
import os
import traceback


app = Flask(__name__)

APP_ROOT = os.getenv('APP_ROOT', '/infer')
MODEL = None

@app.route(APP_ROOT, methods=["POST"])
def infer():
    data = request.json
    image = data['image']
    image = np.array(image, dtype=np.uint8)
    boxes, _, _ = infer_im(MODEL, image)
    im = draw_boxes(image, boxes)
    strIO = StringIO.StringIO()
    imsave(strIO, im, plugin='pil', format_str='png')
    strIO.seek(0)
    return send_file(strIO, mimetype='image/png')
    # return boxes.__str__()


@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify(stackTrace=traceback.format_exc())


if __name__ == '__main__':
    MODEL = model()
    MODEL.load_weights(cfg.checkpoint_path).expect_partial()
    app.run()
