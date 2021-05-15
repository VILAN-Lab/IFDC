from math import *
from sklearn.cluster import DBSCAN
import math

def get_distance(array_1, array_2):
    lon_a = array_1[0]
    lat_a = array_1[1]
    lon_b = array_2[0]
    lat_b = array_2[1]
    y = math.sqrt(pow((lon_a - lon_b), 2) + pow((lat_a - lat_b), 2))
    radlat1 = radians(lat_a)
    radlat2 = radians(lat_b)
    a = radlat1 - radlat2
    b = radians(lon_a) - radians(lon_b)
    s = 2 * asin(sqrt(pow(sin(a/2),2) + cos(radlat1) * cos(radlat2)*pow(sin(b/2),2)))
    return y


def create_coordinate_list(sorted_box_feat):
    result = []
    for i in sorted_box_feat:
        coord = i[5], i[6]
        result.append(coord)

    return result


def main():
    coordinate_list = create_coordinate_list()

    # DBSCAN Cluster
    dbscan = DBSCAN(eps=20, min_samples=1, metric=get_distance).fit(coordinate_list)

    print(dbscan.labels_)


if __name__ == '__main__':
    main()
