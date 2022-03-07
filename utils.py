# coding:utf-8
import glob
import csv
import cv2
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as Patches
import argparse
from zipfile import ZipFile
import re
from datetime import datetime
import json

import tensorflow as tf
from icdar_detection_script import script
import infer
from config import cfg

def get_images(path):
    files = []
    for ext in ["jpg", "png", "jpeg", "JPG"]:
        files.extend(glob.glob(os.path.join(path, "*.{}".format(ext))))
    return files


def get_test_images():
    files = []
    for ext in ["jpg", "png", "jpeg", "JPG"]:
        files.extend(
            glob.glob(os.path.join(cfg.training_data_path, "*.{}".format(ext)))
        )
    return files


def load_annotation(p):
    """
    load annotation from the text file
    :param p:
    :return:
    """
    text_polys = []
    text_tags = []
    if not os.path.exists(p):
        return np.array(text_polys, dtype=np.float32)
    with open(p, "r") as f:
        reader = csv.reader(f)
        for line in reader:
            label = line[-1]
            # strip BOM. \ufeff for python3,  \xef\xbb\bf for python2
            line = [i.strip("\ufeff").strip("\xef\xbb\xbf") for i in line]

            x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, line[:8]))
            text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            if label == "*" or label == "###":
                text_tags.append(True)
            else:
                text_tags.append(False)
        return np.array(text_polys, dtype=np.float32), np.array(
            text_tags, dtype=np.bool
        )


def polygon_area(poly):
    """
    compute area of a polygon
    :param poly:
    :return:
    """
    edge = [
        (poly[1][0] - poly[0][0]) * (poly[1][1] + poly[0][1]),
        (poly[2][0] - poly[1][0]) * (poly[2][1] + poly[1][1]),
        (poly[3][0] - poly[2][0]) * (poly[3][1] + poly[2][1]),
        (poly[0][0] - poly[3][0]) * (poly[0][1] + poly[3][1]),
    ]
    return np.sum(edge) / 2.0


def check_and_validate_polys(polys, tags, xxx_todo_changeme):
    """
    check so that the text poly is in the same direction,
    and also filter some invalid polygons
    :param polys:
    :param tags:
    :return:
    """
    (h, w) = xxx_todo_changeme
    if polys.shape[0] == 0:
        return polys
    polys[:, :, 0] = np.clip(polys[:, :, 0], 0, w - 1)
    polys[:, :, 1] = np.clip(polys[:, :, 1], 0, h - 1)

    validated_polys = []
    validated_tags = []
    for poly, tag in zip(polys, tags):
        p_area = polygon_area(poly)
        if abs(p_area) < 1:
            # print poly
            print("invalid poly")
            continue
        if p_area > 0:
            print("poly in wrong direction")
            poly = poly[(0, 3, 2, 1), :]
        validated_polys.append(poly)
        validated_tags.append(tag)
    return np.array(validated_polys), np.array(validated_tags)


def visualize_maps(im, score_map, geo_map, training_mask=None, text_polys=None):
    _, axs = plt.subplots(3, 2, figsize=(20, 30))
    axs[0, 0].imshow(im[:, :, ::-1])
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])
    if text_polys is not None:
        for poly in text_polys:
            poly_h = min(abs(poly[3, 1] - poly[0, 1]), abs(poly[2, 1] - poly[1, 1]))
            poly_w = min(abs(poly[1, 0] - poly[0, 0]), abs(poly[2, 0] - poly[3, 0]))
            axs[0, 0].add_artist(
                Patches.Polygon(
                    poly,
                    facecolor="none",
                    edgecolor="green",
                    linewidth=2,
                    linestyle="-",
                    fill=True,
                )
            )
            axs[0, 0].text(
                poly[0, 0],
                poly[0, 1],
                "{:.0f}-{:.0f}".format(poly_h, poly_w),
                color="purple",
            )
    axs[0, 1].imshow(score_map[::, ::])
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])
    axs[1, 0].imshow(geo_map[::, ::, 0])
    axs[1, 0].set_xticks([])
    axs[1, 0].set_yticks([])
    axs[1, 1].imshow(geo_map[::, ::, 1])
    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])
    axs[2, 0].imshow(geo_map[::, ::, 4])
    axs[2, 0].set_xticks([])
    axs[2, 0].set_yticks([])
    if training_mask is not None:
        axs[2, 1].imshow(training_mask[::, ::])
        axs[2, 1].set_xticks([])
        axs[2, 1].set_yticks([])
    plt.tight_layout()
    plt.show()
    plt.close()


def visualize_inferred(im, score_map, geo_map):
    _, axs = plt.subplots(3, 2, figsize=(20, 30))
    axs[0, 0].imshow(im[:, :, ::-1])
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])
    axs[0, 1].imshow(score_map[::, ::])
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])
    axs[1, 0].imshow(geo_map[::, ::, 0])
    axs[1, 0].set_xticks([])
    axs[1, 0].set_yticks([])
    axs[1, 1].imshow(geo_map[::, ::, 1])
    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])
    axs[2, 0].imshow(geo_map[::, ::, 4])
    axs[2, 0].set_xticks([])
    axs[2, 0].set_yticks([])
    plt.tight_layout()
    plt.show()
    plt.close()


def visualize_boxes(im, boxes):
    for box in boxes:
        im = cv2.polylines(
            im,
            [box.reshape((-1, 1, 2))],
            isClosed=True,
            color=(255, 255, 0),
            thickness=1,
        )
    plt.imshow(im[:, :, ::-1])
    plt.show()


def crop_area(im, polys, tags, crop_background=False, max_tries=50):
    """
    make random crop from the input image
    :param im:
    :param polys:
    :param tags:
    :param crop_background:
    :param max_tries:
    :return:
    """
    h, w, _ = im.shape
    pad_h = h // 10
    pad_w = w // 10
    h_array = np.zeros((h + pad_h * 2), dtype=np.int32)
    w_array = np.zeros((w + pad_w * 2), dtype=np.int32)
    for poly in polys:
        poly = np.round(poly, decimals=0).astype(np.int32)
        minx = np.min(poly[:, 0])
        maxx = np.max(poly[:, 0])
        w_array[minx + pad_w : maxx + pad_w] = 1
        miny = np.min(poly[:, 1])
        maxy = np.max(poly[:, 1])
        h_array[miny + pad_h : maxy + pad_h] = 1
    # ensure the cropped area not across a text
    h_axis = np.where(h_array == 0)[0]
    w_axis = np.where(w_array == 0)[0]
    if len(h_axis) == 0 or len(w_axis) == 0:
        return im, polys, tags
    for i in range(max_tries):
        xx = np.random.choice(w_axis, size=2)
        xmin = np.min(xx) - pad_w
        xmax = np.max(xx) - pad_w
        xmin = np.clip(xmin, 0, w - 1)
        xmax = np.clip(xmax, 0, w - 1)
        yy = np.random.choice(h_axis, size=2)
        ymin = np.min(yy) - pad_h
        ymax = np.max(yy) - pad_h
        ymin = np.clip(ymin, 0, h - 1)
        ymax = np.clip(ymax, 0, h - 1)
        if (
            xmax - xmin < cfg.min_crop_side_ratio * w
            or ymax - ymin < cfg.min_crop_side_ratio * h
        ):
            # area too small
            continue
        if polys.shape[0] != 0:
            poly_axis_in_area = (
                (polys[:, :, 0] >= xmin)
                & (polys[:, :, 0] <= xmax)
                & (polys[:, :, 1] >= ymin)
                & (polys[:, :, 1] <= ymax)
            )
            selected_polys = np.where(np.sum(poly_axis_in_area, axis=1) == 4)[0]
        else:
            selected_polys = []
        if len(selected_polys) == 0:
            # no text in this area
            if crop_background:
                return (
                    im[ymin : ymax + 1, xmin : xmax + 1, :],
                    polys[selected_polys],
                    tags[selected_polys],
                )
            else:
                continue
        im = im[ymin : ymax + 1, xmin : xmax + 1, :]
        polys = polys[selected_polys]
        tags = tags[selected_polys]
        polys[:, :, 0] -= xmin
        polys[:, :, 1] -= ymin
        return im, polys, tags

    return im, polys, tags


def shrink_poly(poly, r):
    """
    fit a poly inside the origin poly, maybe bugs here...
    used for generate the score map
    :param poly: the text poly
    :param r: r in the paper
    :return: the shrinked poly
    """
    # shrink ratio
    R = 0.3
    # find the longer pair
    if np.linalg.norm(poly[0] - poly[1]) + np.linalg.norm(
        poly[2] - poly[3]
    ) > np.linalg.norm(poly[0] - poly[3]) + np.linalg.norm(poly[1] - poly[2]):
        # first move (p0, p1), (p2, p3), then (p0, p3), (p1, p2)
        ## p0, p1
        theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
        poly[0][0] += R * r[0] * np.cos(theta)
        poly[0][1] += R * r[0] * np.sin(theta)
        poly[1][0] -= R * r[1] * np.cos(theta)
        poly[1][1] -= R * r[1] * np.sin(theta)
        ## p2, p3
        theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
        poly[3][0] += R * r[3] * np.cos(theta)
        poly[3][1] += R * r[3] * np.sin(theta)
        poly[2][0] -= R * r[2] * np.cos(theta)
        poly[2][1] -= R * r[2] * np.sin(theta)
        ## p0, p3
        theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
        poly[0][0] += R * r[0] * np.sin(theta)
        poly[0][1] += R * r[0] * np.cos(theta)
        poly[3][0] -= R * r[3] * np.sin(theta)
        poly[3][1] -= R * r[3] * np.cos(theta)
        ## p1, p2
        theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
        poly[1][0] += R * r[1] * np.sin(theta)
        poly[1][1] += R * r[1] * np.cos(theta)
        poly[2][0] -= R * r[2] * np.sin(theta)
        poly[2][1] -= R * r[2] * np.cos(theta)
    else:
        ## p0, p3
        # print poly
        theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
        poly[0][0] += R * r[0] * np.sin(theta)
        poly[0][1] += R * r[0] * np.cos(theta)
        poly[3][0] -= R * r[3] * np.sin(theta)
        poly[3][1] -= R * r[3] * np.cos(theta)
        ## p1, p2
        theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
        poly[1][0] += R * r[1] * np.sin(theta)
        poly[1][1] += R * r[1] * np.cos(theta)
        poly[2][0] -= R * r[2] * np.sin(theta)
        poly[2][1] -= R * r[2] * np.cos(theta)
        ## p0, p1
        theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
        poly[0][0] += R * r[0] * np.cos(theta)
        poly[0][1] += R * r[0] * np.sin(theta)
        poly[1][0] -= R * r[1] * np.cos(theta)
        poly[1][1] -= R * r[1] * np.sin(theta)
        ## p2, p3
        theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
        poly[3][0] += R * r[3] * np.cos(theta)
        poly[3][1] += R * r[3] * np.sin(theta)
        poly[2][0] -= R * r[2] * np.cos(theta)
        poly[2][1] -= R * r[2] * np.sin(theta)
    return poly


def point_dist_to_line(p1, p2, p3):
    x = np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
    return x


def fit_line(p1, p2):
    # fit a line ax+by+c = 0
    if p1[0] == p1[1]:
        return [1.0, 0.0, -p1[0]]
    else:
        [k, b] = np.polyfit(p1, p2, deg=1)
        return [k, -1.0, b]


def line_cross_point(line1, line2):
    # line1 0= ax+by+c, compute the cross point of line1 and line2
    if line1[0] != 0 and line1[0] == line2[0]:
        print("Cross point does not exist")
        return None
    if line1[0] == 0 and line2[0] == 0:
        print("Cross point does not exist")
        return None
    if line1[1] == 0:
        x = -line1[2]
        y = line2[0] * x + line2[2]
    elif line2[1] == 0:
        x = -line2[2]
        y = line1[0] * x + line1[2]
    else:
        k1, _, b1 = line1
        k2, _, b2 = line2
        x = -(b1 - b2) / (k1 - k2)
        y = k1 * x + b1
    return np.array([x, y], dtype=np.float32)


def line_vertical(line, point):
    # get the vertical line from line across point
    if line[1] == 0:
        vertical = [0, -1, point[1]]
    else:
        if line[0] == 0:
            vertical = [1, 0, -point[0]]
        else:
            vertical = [-1.0 / line[0], -1, point[1] - (-1 / line[0] * point[0])]
    return vertical


def rectangle_from_parallelogram(poly):
    """
    fit a rectangle from a parallelogram
    :param poly:
    :return:
    """
    p0, p1, p2, p3 = poly
    angle_p0 = np.arccos(
        np.dot(p1 - p0, p3 - p0) / (np.linalg.norm(p0 - p1) * np.linalg.norm(p3 - p0))
    )
    if angle_p0 < 0.5 * np.pi:
        if np.linalg.norm(p0 - p1) > np.linalg.norm(p0 - p3):
            # p0 and p2
            ## p0
            p2p3 = fit_line([p2[0], p3[0]], [p2[1], p3[1]])
            p2p3_vertical = line_vertical(p2p3, p0)

            new_p3 = line_cross_point(p2p3, p2p3_vertical)
            ## p2
            p0p1 = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
            p0p1_vertical = line_vertical(p0p1, p2)

            new_p1 = line_cross_point(p0p1, p0p1_vertical)
            return np.array([p0, new_p1, p2, new_p3], dtype=np.float32)
        else:
            p1p2 = fit_line([p1[0], p2[0]], [p1[1], p2[1]])
            p1p2_vertical = line_vertical(p1p2, p0)

            new_p1 = line_cross_point(p1p2, p1p2_vertical)
            p0p3 = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
            p0p3_vertical = line_vertical(p0p3, p2)

            new_p3 = line_cross_point(p0p3, p0p3_vertical)
            return np.array([p0, new_p1, p2, new_p3], dtype=np.float32)
    else:
        if np.linalg.norm(p0 - p1) > np.linalg.norm(p0 - p3):
            # p1 and p3
            ## p1
            p2p3 = fit_line([p2[0], p3[0]], [p2[1], p3[1]])
            p2p3_vertical = line_vertical(p2p3, p1)

            new_p2 = line_cross_point(p2p3, p2p3_vertical)
            ## p3
            p0p1 = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
            p0p1_vertical = line_vertical(p0p1, p3)

            new_p0 = line_cross_point(p0p1, p0p1_vertical)
            return np.array([new_p0, p1, new_p2, p3], dtype=np.float32)
        else:
            p0p3 = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
            p0p3_vertical = line_vertical(p0p3, p1)

            new_p0 = line_cross_point(p0p3, p0p3_vertical)
            p1p2 = fit_line([p1[0], p2[0]], [p1[1], p2[1]])
            p1p2_vertical = line_vertical(p1p2, p3)

            new_p2 = line_cross_point(p1p2, p1p2_vertical)
            return np.array([new_p0, p1, new_p2, p3], dtype=np.float32)


def sort_rectangle(poly):
    # sort the four coordinates of the polygon, points in poly should be sorted clockwise
    # First find the lowest point
    p_lowest = np.argmax(poly[:, 1])
    if np.count_nonzero(poly[:, 1] == poly[p_lowest, 1]) == 2:
        # 底边平行于X轴, 那么p0为左上角 - if the bottom line is parallel to x-axis, then p0 must be the upper-left corner
        p0_index = np.argmin(np.sum(poly, axis=1))
        p1_index = (p0_index + 1) % 4
        p2_index = (p0_index + 2) % 4
        p3_index = (p0_index + 3) % 4
        return poly[[p0_index, p1_index, p2_index, p3_index]], 0.0
    else:
        # 找到最低点右边的点 - find the point that sits right to the lowest point
        p_lowest_right = (p_lowest - 1) % 4
        p_lowest_left = (p_lowest + 1) % 4
        angle = np.arctan(
            -(poly[p_lowest][1] - poly[p_lowest_right][1])
            / (poly[p_lowest][0] - poly[p_lowest_right][0])
        )
        # assert angle > 0
        if angle <= 0:
            print(angle, poly[p_lowest], poly[p_lowest_right])
        if angle / np.pi * 180 > 45:
            # 这个点为p2 - this point is p2
            p2_index = p_lowest
            p1_index = (p2_index - 1) % 4
            p0_index = (p2_index - 2) % 4
            p3_index = (p2_index + 1) % 4
            return poly[[p0_index, p1_index, p2_index, p3_index]], -(np.pi / 2 - angle)
        else:
            # 这个点为p3 - this point is p3
            p3_index = p_lowest
            p0_index = (p3_index + 1) % 4
            p1_index = (p3_index + 2) % 4
            p2_index = (p3_index + 3) % 4
            return poly[[p0_index, p1_index, p2_index, p3_index]], angle


def restore_rectangle_rbox(origin, geometry):
    d = geometry[:, :4]
    angle = geometry[:, 4]
    # for angle > 0
    origin_0 = origin[angle >= 0]
    d_0 = d[angle >= 0]
    angle_0 = angle[angle >= 0]
    if origin_0.shape[0] > 0:
        p = np.array(
            [
                np.zeros(d_0.shape[0]),
                -d_0[:, 0] - d_0[:, 2],
                d_0[:, 1] + d_0[:, 3],
                -d_0[:, 0] - d_0[:, 2],
                d_0[:, 1] + d_0[:, 3],
                np.zeros(d_0.shape[0]),
                np.zeros(d_0.shape[0]),
                np.zeros(d_0.shape[0]),
                d_0[:, 3],
                -d_0[:, 2],
            ]
        )
        p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2

        rotate_matrix_x = np.array([np.cos(angle_0), np.sin(angle_0)]).transpose((1, 0))
        rotate_matrix_x = (
            np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))
        )  # N*5*2

        rotate_matrix_y = np.array([-np.sin(angle_0), np.cos(angle_0)]).transpose(
            (1, 0)
        )
        rotate_matrix_y = (
            np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))
        )

        p_rotate_x = np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
        p_rotate_y = np.sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1

        p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

        p3_in_origin = origin_0 - p_rotate[:, 4, :]
        new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
        new_p1 = p_rotate[:, 1, :] + p3_in_origin
        new_p2 = p_rotate[:, 2, :] + p3_in_origin
        new_p3 = p_rotate[:, 3, :] + p3_in_origin

        new_p_0 = np.concatenate(
            [
                new_p0[:, np.newaxis, :],
                new_p1[:, np.newaxis, :],
                new_p2[:, np.newaxis, :],
                new_p3[:, np.newaxis, :],
            ],
            axis=1,
        )  # N*4*2
    else:
        new_p_0 = np.zeros((0, 4, 2))
    # for angle < 0
    origin_1 = origin[angle < 0]
    d_1 = d[angle < 0]
    angle_1 = angle[angle < 0]
    if origin_1.shape[0] > 0:
        p = np.array(
            [
                -d_1[:, 1] - d_1[:, 3],
                -d_1[:, 0] - d_1[:, 2],
                np.zeros(d_1.shape[0]),
                -d_1[:, 0] - d_1[:, 2],
                np.zeros(d_1.shape[0]),
                np.zeros(d_1.shape[0]),
                -d_1[:, 1] - d_1[:, 3],
                np.zeros(d_1.shape[0]),
                -d_1[:, 1],
                -d_1[:, 2],
            ]
        )
        p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2

        rotate_matrix_x = np.array([np.cos(-angle_1), -np.sin(-angle_1)]).transpose(
            (1, 0)
        )
        rotate_matrix_x = (
            np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))
        )  # N*5*2

        rotate_matrix_y = np.array([np.sin(-angle_1), np.cos(-angle_1)]).transpose(
            (1, 0)
        )
        rotate_matrix_y = (
            np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))
        )

        p_rotate_x = np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
        p_rotate_y = np.sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1

        p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

        p3_in_origin = origin_1 - p_rotate[:, 4, :]
        new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
        new_p1 = p_rotate[:, 1, :] + p3_in_origin
        new_p2 = p_rotate[:, 2, :] + p3_in_origin
        new_p3 = p_rotate[:, 3, :] + p3_in_origin

        new_p_1 = np.concatenate(
            [
                new_p0[:, np.newaxis, :],
                new_p1[:, np.newaxis, :],
                new_p2[:, np.newaxis, :],
                new_p3[:, np.newaxis, :],
            ],
            axis=1,
        )  # N*4*2
    else:
        new_p_1 = np.zeros((0, 4, 2))
    return np.concatenate([new_p_0, new_p_1])


def generate_rbox(im_size, polys, tags):
    h, w = im_size
    poly_mask = np.zeros((h, w), dtype=np.uint8)
    score_map = np.zeros((h, w), dtype=np.uint8)
    geo_map = np.zeros((h, w, 5), dtype=np.float32)
    training_mask = np.ones((h, w), dtype=np.uint8)
    for poly_idx, poly_tag in enumerate(zip(polys, tags)):
        poly = poly_tag[0]
        tag = poly_tag[1]

        r = [None, None, None, None]
        for i in range(4):
            r[i] = min(
                np.linalg.norm(poly[i] - poly[(i + 1) % 4]),
                np.linalg.norm(poly[i] - poly[(i - 1) % 4]),
            )
        shrunk_poly = shrink_poly(poly.copy(), r).astype(np.int32)[np.newaxis, :, :]
        cv2.fillPoly(score_map, shrunk_poly, 1)
        cv2.fillPoly(poly_mask, shrunk_poly, poly_idx + 1)
        poly_h = min(
            np.linalg.norm(poly[0] - poly[3]), np.linalg.norm(poly[1] - poly[2])
        )
        poly_w = min(
            np.linalg.norm(poly[0] - poly[1]), np.linalg.norm(poly[2] - poly[3])
        )
        if min(poly_h, poly_w) < cfg.min_text_size:
            cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
        if tag:
            cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)

        xy_in_poly = np.argwhere(poly_mask == (poly_idx + 1))
        box = cv2.minAreaRect(
            poly
        )  # Finds min area rotated rectangle containing four points.
        # p0_rect, p1_rect, p2_rect, p3_rect = cv2.boxPoints(box)  # Gets four corners from (center, (w, h), angle)
        (p0_rect, p1_rect, p2_rect, p3_rect), angle = sort_rectangle(cv2.boxPoints(box))
        for y, x in xy_in_poly:
            point = np.array([x, y], dtype=np.float32)
            # top
            geo_map[y, x, 0] = point_dist_to_line(p0_rect, p1_rect, point)
            # right
            geo_map[y, x, 1] = point_dist_to_line(p1_rect, p2_rect, point)
            # down
            geo_map[y, x, 2] = point_dist_to_line(p2_rect, p3_rect, point)
            # left
            geo_map[y, x, 3] = point_dist_to_line(p3_rect, p0_rect, point)
            # angle
            # geo_map[y, x, 4] = box[2]
            geo_map[y, x, 4] = angle
    return score_map, geo_map, training_mask


def generate_non_augmented(filenames, input_size=512, vis=False):
    images = []
    image_fns = []
    score_maps = []
    geo_maps = []
    for im_fn in filenames:
        im = cv2.imread(im_fn)
        h, w, _ = im.shape
        txt_fn = im_fn.replace(os.path.basename(im_fn).split(".")[1], "txt")
        if not os.path.exists(txt_fn):
            print(f"text file {txt_fn} does not exist")
            continue

        text_polys, text_tags = load_annotation(txt_fn)
        text_polys, text_tags = check_and_validate_polys(text_polys, text_tags, (h, w))
        im = cv2.resize(im, dsize=(input_size, input_size))
        ratio_w = input_size / w
        ratio_h = input_size / h
        text_polys[:, :, 0] *= ratio_w
        text_polys[:, :, 1] *= ratio_h
        score_map, geo_map, training_mask = generate_rbox(
            (input_size, input_size), text_polys, text_tags
        )
        if vis:
            visualize_maps(im, score_map, geo_map, training_mask, text_polys)

        images.append(im[:, :, ::-1].astype(np.float32))
        image_fns.append(im_fn)
        score_maps.append(score_map[::4, ::4, np.newaxis].astype(np.float32))
        geo_maps.append(geo_map[::4, ::4, :].astype(np.float32))

    return images, image_fns, score_maps, geo_maps, None


def generate(
    filenames,
    input_size=512,
    background_ratio=3.0 / 8,
    random_scale=np.array([0.5, 1, 2.0, 3.0]),
    vis=False,
):
    images = []
    image_fns = []
    score_maps = []
    geo_maps = []
    training_masks = []
    for i in range(len(filenames)):
        im_fn = filenames[i]
        im = cv2.imread(im_fn)
        # print im_fn
        h, w, _ = im.shape
        txt_fn = im_fn.replace(os.path.basename(im_fn).split(".")[1], "txt")
        if not os.path.exists(txt_fn):
            print(f"text file {txt_fn} does not exist")
            continue

        text_polys, text_tags = load_annotation(txt_fn)

        text_polys, text_tags = check_and_validate_polys(text_polys, text_tags, (h, w))
        # if text_polys.shape[0] == 0:
        #     continue
        # random scale this image
        rd_scale = np.random.choice(random_scale)
        im = cv2.resize(im, dsize=None, fx=rd_scale, fy=rd_scale)
        text_polys *= rd_scale
        # print rd_scale
        # random crop a area from image
        if np.random.rand() < background_ratio:
            # crop background
            im, text_polys, text_tags = crop_area(
                im, text_polys, text_tags, crop_background=True
            )
            if text_polys.shape[0] > 0:
                continue
                # cannot find background
            # pad and resize image
            new_h, new_w, _ = im.shape
            max_h_w_i = np.max([new_h, new_w, input_size])
            im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
            im_padded[:new_h, :new_w, :] = im.copy()
            im = cv2.resize(im_padded, dsize=(input_size, input_size))
            score_map = np.zeros((input_size, input_size), dtype=np.uint8)
            geo_map_channels = 5  # ONLY RBOX SUPPORTED. WITH QUAD IT WOULD BE 8
            geo_map = np.zeros(
                (input_size, input_size, geo_map_channels), dtype=np.float32
            )
            training_mask = np.ones((input_size, input_size), dtype=np.uint8)
        else:
            im, text_polys, text_tags = crop_area(
                im, text_polys, text_tags, crop_background=False
            )
            if text_polys.shape[0] == 0:
                continue
            h, w, _ = im.shape

            # pad the image to the training input size or the longer side of image
            new_h, new_w, _ = im.shape
            max_h_w_i = np.max([new_h, new_w, input_size])
            im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
            im_padded[:new_h, :new_w, :] = im.copy()
            im = im_padded
            # resize the image to input size
            new_h, new_w, _ = im.shape
            resize_h = input_size
            resize_w = input_size
            im = cv2.resize(im, dsize=(resize_w, resize_h))
            resize_ratio_3_x = resize_w / float(new_w)
            resize_ratio_3_y = resize_h / float(new_h)
            text_polys[:, :, 0] *= resize_ratio_3_x
            text_polys[:, :, 1] *= resize_ratio_3_y
            new_h, new_w, _ = im.shape
            score_map, geo_map, training_mask = generate_rbox(
                (new_h, new_w), text_polys, text_tags
            )

        if vis:
            visualize_maps(im, score_map, geo_map, training_mask, text_polys)

        images.append(im[:, :, ::-1].astype(np.float32))
        image_fns.append(im_fn)
        score_maps.append(score_map[::4, ::4, np.newaxis].astype(np.float32))
        geo_maps.append(geo_map[::4, ::4, :].astype(np.float32))
        training_masks.append(training_mask[::4, ::4, np.newaxis].astype(np.float32))

    return images, image_fns, score_maps, geo_maps, training_masks


class IcdarTrainingSequence(tf.keras.utils.Sequence):
    def __init__(self):
        self.filenames = np.array(get_images(cfg.training_data_path))
        self.batch_size = cfg.batch_size
        self.index = np.array(range(len(self.filenames)))
        np.random.shuffle(self.index)

    def __len__(self):
        return int(np.ceil(len(self.filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        filenames = self.filenames[
            self.index[idx * self.batch_size : (idx + 1) * self.batch_size]
        ]
        images, _, score_maps, geo_maps, training_masks = generate(filenames)
        if len(images) == 0:
            return self.__getitem__((idx + 1) % self.__len__())
        return tf.convert_to_tensor(images), tf.convert_to_tensor(np.concatenate((geo_maps, score_maps, training_masks), axis=-1))


class IcdarValidationSequence(tf.keras.utils.Sequence):
    def __init__(self):
        self.filenames = np.array(get_images(cfg.validation_data_path))
        self.batch_size = cfg.batch_size
        self.index = np.array(range(len(self.filenames)))
        np.random.shuffle(self.index)

    def __len__(self):
        return int(np.ceil(len(self.filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        filenames = self.filenames[
            self.index[idx * self.batch_size : (idx + 1) * self.batch_size]
        ]
        images, _, score_maps, geo_maps, training_masks = generate(filenames)
        if len(images) == 0:
            return self.__getitem__((idx + 1) % self.__len__())
        return tf.convert_to_tensor(images), tf.convert_to_tensor(np.concatenate((geo_maps, score_maps, training_masks), axis=-1))


class IcdarEvaluationCallback(tf.keras.callbacks.Callback):
    def __init__(self, N = 5):
        super().__init__()
        self.N = N
        self.name = re.sub('[\-\s\:]', '', str(datetime.utcnow()))[:14]
        self.training_results_path = f"training_results_{self.name}"
        os.mkdir(self.training_results_path)
        self.logs_file = os.path.join(self.training_results_path, 'logs')
        f = open(self.logs_file, 'w')
        f.close()
        with open(f'{self.logs_file}_cfg.json', 'w') as f:
            json.dump(cfg.__dict__, f, indent=2)


    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.N == 0:
            results = infer_and_test(self.model)
            # print(f'precision {results["precision"]} | recall {results["recall"]}'
            # f'| hmean {results["hmean"]}')
            logs = {**logs, **results}
            logs['epoch'] = epoch
            print(logs)
        with open(self.logs_file, 'a') as f:
            json.dump(logs, f)
        self.model.save_weights(os.path.join(self.training_results_path, f"ckpt_epoch_{epoch}"))


def infer_and_test(model):
    output_path = infer.infer(model)
    zip_path = 'tmp_results.zip' 
    with ZipFile(zip_path, 'w') as zipObj:
        for fn in os.listdir(output_path):
            if not '.txt' in fn:
                continue
            zipObj.write(os.path.join(output_path, fn), fn)


    res_dic = script.main(zip_path)
    # Return a dictionary with precision, recall and hmean
    return res_dic['method']
