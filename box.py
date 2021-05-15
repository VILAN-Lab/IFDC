# import cv2
import numpy as np
import os
import json
import math
from operator import itemgetter
from creat_coord import *
import argparse
from tqdm import tqdm

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

class HybridLoader:
    """
    If db_path is a director, then use normal file loading
    If lmdb, then load from lmdb
    The loading method depend on extention.
    """

    def __init__(self, db_path, ext):
        self.db_path = db_path
        self.ext = ext
        if self.ext == '.npy':
            self.loader = lambda x: np.load(x)
        self.db_type = 'dir'

    def get(self, key):

        f_input = os.path.join(self.db_path, key + self.ext)
        # load image
        feat = self.loader(f_input)
        return feat

class box:
    def __init__(self):
        # self.box_loader = HybridLoader("/home/ubuntu/LY/self-critical.pytorch-master/data/bottom_up_box", '.npy')
        self.box1_loader = HybridLoader("/home/ubuntu/LY/self-critical.pytorch-master/data/bottom_up_box1", '.npy')
    def get_box(self, id):
        # box_feat = self.box_loader.get(str(id))
        box1_feat = self.box1_loader.get(str(id))

        return box1_feat

def sort(box_feat):
    c = np.zeros((36, 7))
    for k, i in enumerate(box_feat):
        x1, y1, x2, y2 = i[0], i[1], i[2], i[3]
        s = (x2 - x1) * (y2 - y1)
        w = x2 - x1
        h = y2 - y1
        x = x1 + w/2
        y = y1 + h/2
        c[k] = (x1, y1, x2, y2, s, x, y)
    sorted_feat = sorted(c, key=lambda x: x[4])
    sorted_feat = sorted_feat[0:25]
    sorted_feat = sorted(sorted_feat, key=lambda x: x[5])

    return sorted_feat

def cluster(box_feat, label, id):
    lab = []
    k = max(label)
    for index, value in enumerate(label):
        t = (index, value)
        lab.append(t)
    clu = dict()
    for i in range(k+1):
        a = []
        for j in lab:
            if j[1] == i:
                a.append(j)
        clu[i] = a
    coord = dict()
    for i in range(k+1):
        b = clu[i]
        x_sum = 0
        y_sum = 0
        for j in b:
            x_sum += box_feat[j[0]][5]
            y_sum += box_feat[j[0]][6]
        x = x_sum / len(b)
        y = y_sum / len(b)
        coord[i] = (x, y)
    example = {"clu": clu, "coord": coord, "id": id}
    return example

def main(params):
    file = open(params['output_json'], 'w', encoding='utf-8')
    images = json.load(open(params['input_json'], 'r'))
    imgs = images['images']
    for img in tqdm(imgs):
        id = img['id']
        box_feat = box.get_box(id)
        sorted_box_feat = sort(box_feat)
        coordinate_list = create_coordinate_list(sorted_box_feat)
        dbscan = DBSCAN(eps=20, min_samples=1, metric=get_distance).fit(coordinate_list)
        label = dbscan.labels_
        res = cluster(sorted_box_feat, label, id)
        file.write(json.dumps(res, cls=NpEncoder) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json', default='/home/ubuntu/LY/self-critical.pytorch-master/data/outfile/4+4/output.json')
    parser.add_argument('--output_json', default='/home/ubuntu/LY/self-critical.pytorch-master/data/label/label1_25_4+4.json')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    box = box()
    main(params)