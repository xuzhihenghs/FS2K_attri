import json
from torchvision import transforms
import math
import time


# 解析json文件
def Dejson(selected_attrs, json_path):
    fp = open(json_path, 'r')
    data = json.load(fp)
    img_path_list = list()
    attrs_list = list()
    for item in data:
        str = item['image_name'].replace('photo', 'sketch').replace('image', 'sketch')
        str += '.jpg' if (
                '1' in item['image_name'].split('/')[0] or '3' in item['image_name'].split('/')[0]) else '.png'
        img_path_list.append(str)
        attrs = list()
        for attr in selected_attrs:
            attrs.append(item[attr])
        attrs_list.append(attrs)
    return img_path_list, attrs_list


# 将图片属性 变成正方形 -> 变为tensor变量 -> 进行标准化
def set_transform():
    transform = []
    transform.append(transforms.Resize(size=(224, 224)))
    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                          std=[0.5, 0.5, 0.5]))
    transform = transforms.Compose(transform)
    return transform


# 计算当前运行时间
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    return '%d h %d m %d s' % (h, m, s)
