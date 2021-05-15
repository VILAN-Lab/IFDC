import numpy as np
from creat_coord import *

def cluster(box_feat, label):
    lab = []
    k = max(label)
    for index, value in enumerate(label):
        t = (index, value)
        lab.append(t)
    b = []
    for i in range(k+1):
        a = []
        for j in lab:
            if j[1] == i:
                a.append(j)
        b += [a]
    d = []
    for i in b:
        for j in i:
            c = j[0]
            d.append(c)
    new_box = np.zeros((25, 4))
    box_pos = np.zeros((25, 1))
    for k, i in enumerate(d):
        new_box[k] = box_feat[i][:4]
        box_pos[k] = box_feat[i][7]

    return new_box, box_pos

def sort(box_feat):
    c = np.zeros((36, 8))
    for k, i in enumerate(box_feat):
        x1, y1, x2, y2 = i[0], i[1], i[2], i[3]
        s = (x2 - x1) * (y2 - y1)
        w = x2 - x1
        h = y2 - y1
        x = x1 + w / 2
        y = y1 + h / 2
        c[k] = (x1, y1, x2, y2, s, x, y, k)
    sorted_feat = sorted(c, key=lambda x: x[4])
    sorted_feat = sorted_feat[0:25]
    sorted_feat = sorted(sorted_feat, key=lambda x: x[5])
    return sorted_feat

def box_transfer(box_feat):
    sorted_box_feat = sort(box_feat)
    coordinate_list = create_coordinate_list(sorted_box_feat)
    dbscan = DBSCAN(eps=20, min_samples=1, metric=get_distance).fit(coordinate_list)
    label = dbscan.labels_
    new_box_feat, box_pos = cluster(sorted_box_feat, label)
    return new_box_feat, box_pos

def att_transfer(box_pos, att):
    sorted_att_feat = np.zeros((25, 2048))
    for k, i in enumerate(box_pos):
        sorted_att_feat[k] = att[int(i)]
    return sorted_att_feat