import os

import matplotlib.colors as colors
import numpy as np
from pycocotools.coco import COCO

all_object_colors = [
    'k',  # 0 background
    'm',  # 1 box
    'w',  # 2 bottle
    'c',  # 3 cola_can
    'g',  # 4 power_bank
]
all_object_colors = [colors.to_rgb(object_color) for object_color in all_object_colors]


def get_object_ids(data_root, ann_file, object_names):
    coco = COCO(os.path.join(data_root, 'annotations', ann_file))
    object_ids_map = {cat['name']: cat['id'] for cat in coco.dataset['categories']}
    return [object_ids_map[object_name] for object_name in object_names]


def get_object_colors(data_root, ann_file, object_names):
    object_ids = get_object_ids(data_root, ann_file, object_names)
    object_colors = {
        object_name: all_object_colors[object_id]
        for object_name, object_id in zip(object_names, object_ids)
    }
    return object_colors


def get_object_colormap(data_root, ann_file, object_names):
    object_ids = get_object_ids(data_root, ann_file, object_names)
    object_colormap = np.zeros((max(object_ids) + 1, 3))
    for object_id in object_ids:
        object_colormap[object_id] = all_object_colors[object_id]
    return object_colormap
