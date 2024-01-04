import os
from time import strptime,mktime,localtime,strftime
from math import radians, cos, sin, asin, sqrt


def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def get_distance (lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    distance = c * r
    return distance

def string2stamp(timeString):
    _timeTuple = strptime(timeString, "%Y-%m-%d %H:%M:%S")
    return int(mktime(_timeTuple))

def stamp2string(timeStamp):
    _timeTuple = localtime(float(timeStamp))
    return strftime("%Y-%m-%d %H:%M:%S", _timeTuple)