import cv2
import numpy as np
import os
import time
import tensorflow as tf
from datetime import datetime
import json

from model import model

import utils
import lanms
from config import cfg

SCORE_MAP_THRESHOLD = 0.8
BOX_THRESHOLD = 0.1
NMS_THRESHOLD = 0.2


def get_images(validation_dataset):
    """
    find image files in test data path
    :return: list of files found
    """
    files = []
    exts = ["jpg", "png", "jpeg", "JPG"]
    for parent, _, filenames in os.walk(
        cfg.validation_data_path if validation_dataset else cfg.data_path
    ):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print(f"Find {len(files)} images")
    return files


def resize_image(im, max_side_len=2400):
    """
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    """
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = (
            float(max_side_len) / resize_h
            if resize_h > resize_w
            else float(max_side_len) / resize_w
        )
    else:
        ratio = 1.0
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    resize_h = max(32, resize_h)
    resize_w = max(32, resize_w)
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def detect(score_map, geo_map, timer):
    """
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    """
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[
            0,
            :,
            :,
        ]
    # filter the score map
    xy_text = np.argwhere(score_map > SCORE_MAP_THRESHOLD)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    start = time.time()
    text_box_restored = utils.restore_rectangle_rbox(
        xy_text[:, ::-1] * 4, geo_map[xy_text[:, 0], xy_text[:, 1], :]
    )  # N*4*2
    # print(f"{text_box_restored.shape[0]} text boxes before nms")
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    timer["restore"] = time.time() - start
    # nms part
    start = time.time()
    boxes = lanms.merge_quadrangle_n9(boxes.astype("float32"), NMS_THRESHOLD)
    timer["nms"] = time.time() - start

    if boxes.shape[0] == 0:
        return None, timer

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > BOX_THRESHOLD]

    return boxes, timer


def get_run_name():
    _date_and_time = str(datetime.now().isoformat()).split("T")
    return _date_and_time[0].replace("-", "") + _date_and_time[1].replace(":", "")[:6]


def write_result(boxes, im_fn, run_name, timers):
    if not os.path.exists("output"):
        os.mkdir("output")
    if not os.path.exists(f"output/{run_name}"):
        os.mkdir(f"output/{run_name}")
    file_name = f"res_{os.path.basename(im_fn).split('.')[0]}.txt"
    with open(os.path.join(f"output/{run_name}", file_name), "w") as f:
        line_txts = []
        for box in boxes:
            line_txts.append(",".join([str(c) for coord in box for c in coord]))
        f.write("\n".join(line_txts))
    if cfg.write_timer:
        with open(os.path.join(f"output/{run_name}", f"timer_{run_name}.json"), "w") as f:
            json.dump(timers, f)


def infer(m, visualize_inferred_map=False, validation_dataset=False):
    run_name = get_run_name()
    imgs = get_images(validation_dataset)
    timers = []
    for im_fn in imgs:
        print(f"Inferring {im_fn}")
        im = cv2.imread(im_fn)[:, :, ::-1]
        boxes, score, geo, timer = infer_im(m, im) 
        timers.append(timer)
        if visualize_inferred_map:
            utils.visualize_inferred(im, score[0, :, :, 0], geo[0, ...])
        if cfg.visualize:
            utils.visualize_boxes(im[:, :, ::-1], boxes)
        if not cfg.dont_write:
            write_result(boxes, im_fn, run_name, timers)
    return f"output/{run_name}"


def infer_im(m, im):
    im_resized, (ratio_h, ratio_w) = resize_image(im)

    timer = {}
    t0 = time.time()
    output = m(tf.convert_to_tensor(im_resized[np.newaxis, :, :, :]))
    timer["network"] = time.time() - t0
    geo, score = tf.split(output, [5, 1], -1)
    boxes, timer = detect(np.array(score), np.array(geo), timer)
    if boxes is not None:
        boxes = boxes[:, :8].reshape(-1, 4, 2)
        boxes[:, :, 0] /= ratio_w
        boxes[:, :, 1] /= ratio_h
        boxes = boxes.astype(np.int32)
    else:
        boxes = []
    return boxes, score, geo, timer


if __name__ == "__main__":
    m = model()
    m.load_weights(cfg.checkpoint_path).expect_partial()
    infer(m)
