from torch.utils import data
import os
from PIL import Image
# 下面是自己写的文件
import config as cfg
from utils import Dejson


# 加载FS2K数据集的类
class FS2K(data.Dataset):
    def __init__(self, json_path, selected_attrs, transform, mode="train"):
        self.img_path_list, self.labels_list = Dejson(selected_attrs, json_path)
        self.transform = transform

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        labels = self.labels_list[index]
        image = Image.open(os.path.join(cfg.root, img_path)).convert('RGB')
        if self.transform != None:
            image = self.transform(image)
        return image, labels


def get_loader(json_path, selected_attrs, batch_size, mode='train', transform=None):
    dataset = FS2K(json_path, selected_attrs, transform, mode)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode == 'train'),
                                  drop_last=True)
    return data_loader
