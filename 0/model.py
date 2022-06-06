from __future__ import print_function, division
import torch.nn as nn
from torchvision import transforms, models


# 特征提取类:让输入对图像经过baseline的(去掉全连接层)卷积神经网络 得到图像特征
class FeatureExtraction(nn.Module):
    def __init__(self, pretrained, model_type="Resnet18"):
        super(FeatureExtraction, self).__init__()

        if model_type == "Resnet18":
            self.model = models.resnet18(pretrained=pretrained)
        elif model_type == "AlexNet":
            self.model = models.alexnet(pretrained=pretrained)
        elif model_type == "VGG16":
            self.model = models.vgg16(pretrained=pretrained)

        self.model = nn.Sequential(*list(self.model.children())[:-1])  # 去掉网络的最后全连接层

    def forward(self, image):
        return self.model(image)  # 以resnet为例 [B,3,224,224] 将变为 [B,512,1,1]


# 自定义的全连接层
class FullConnect(nn.Module):
    def __init__(self, input_dim=512, output_dim=2):
        super(FullConnect, self).__init__()

        output_dim = output_dim

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 32),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(32, output_dim)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 以resnet为例 [B,512,1,1] -> [B,512]
        res = self.fc(x)
        res = self.sigmoid(res)
        return res


# 多属性分类模型 利用一个卷积神经网络对图像提取特征 生成对图像特征给多个fc全连接层 不同对fc对应不同对分类任务
class multiattribute_Model(nn.Module):
    def __init__(self, model_type, pretrained):
        super(multiattribute_Model, self).__init__()
        self.featureExtractor = FeatureExtraction(pretrained, model_type)

        if model_type == "Resnet18":
            self.feature_dim = 512
        elif model_type == "VGG16":
            self.feature_dim = 25088
        elif model_type == "AlexNet":
            self.feature_dim = 9216

        self.FC_hair = FullConnect(input_dim=self.feature_dim)
        self.FC_gender = FullConnect(input_dim=self.feature_dim)
        self.FC_earring = FullConnect(input_dim=self.feature_dim)
        self.FC_smile = FullConnect(input_dim=self.feature_dim)
        self.FC_frontal = FullConnect(input_dim=self.feature_dim)
        self.FC_style = FullConnect(input_dim=self.feature_dim, output_dim=3)

    def forward(self, image):
        # 得到图像特征
        features = self.featureExtractor(image)
        # 不同fc对不同属性进行分类
        hair = self.FC_hair(features)
        gender = self.FC_gender(features)
        earring = self.FC_earring(features)
        smile = self.FC_smile(features)
        frontal = self.FC_frontal(features)
        style = self.FC_style(features)
        return hair, gender, earring, smile, frontal, style
