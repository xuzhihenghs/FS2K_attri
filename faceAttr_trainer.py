import torch.nn.functional as F
import torch
import torch.optim as optim
import json
import copy
import time
import pandas as pd
import numpy as np
# 自己写的文件
import config as cfg
from model import multiattribute_Model
from utils import set_transform, timeSince
from FS2K import get_loader


class Classifier_Trainer(object):

    def __init__(self, epoches, batch_size, learning_rate, model_type, pretrained=True):
        self.epoches = epoches  # 迭代次数
        self.batch_size = batch_size  # 批处理大小
        self.learning_rate = learning_rate  # 学习率
        self.selected_attrs = cfg.selected_attrs  # 选择的属性 头发|笑容｜性别...
        self.json_train_path = cfg.json_train_path  # train_json路径 来加载训练数据集
        self.json_test_path = cfg.json_test_path  # test_json路径 来加载测试数据集
        self.device = torch.device("cuda:" + str(cfg.DEVICE_ID) if torch.cuda.is_available() else "cpu")  # 计算设备
        self.pretrained = pretrained  # 是否采用预训练的模型来微调 (微调更宜)
        self.model_type = model_type  # 模型种类 resnet18 | vgg16 | AlexNet | ...
        self.start_time = 0  # 开始运行的时间
        self.transform = set_transform()  # 设置数据处理器
        self.train_loader = get_loader(self.json_train_path, self.selected_attrs, self.batch_size, 'train',
                                       self.transform)  # 训练数据加载器
        self.test_loader = get_loader(self.json_test_path, self.selected_attrs, self.batch_size, 'test',
                                      self.transform)  # 测试数据加载器

        self.model = multiattribute_Model(model_type, pretrained).to(self.device)  # 根据model_type构建模型
        self.optimer = optim.Adam(self.model.parameters(), lr=self.learning_rate)  # 优化器 负责更新参数
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimer, [30, 80], gamma=0.1)  # 负责调节学习率

    def train(self, epoch):
        self.model.train()
        temp_loss = 0  # 记录这个epoch的loss
        batch_idx = 0
        for batch_idx, data in enumerate(self.train_loader):
            images, labels = data  # 得到图片以及label
            images = images.to(self.device)  # 将图片放到显卡上进行计算
            hair, gender, earring, smile, frontal, style = self.model(images)  # 得到模型出来的结果
            # 计算各个属性的 交叉熵损失
            hair_loss = F.cross_entropy(input=hair, target=labels[0].to(self.device))
            gender_loss = F.cross_entropy(gender, labels[1].to(self.device))
            earring_loss = F.cross_entropy(earring, labels[2].to(self.device))
            smile_loss = F.cross_entropy(smile, labels[3].to(self.device))
            frontal_loss = F.cross_entropy(frontal, labels[4].to(self.device))
            style_loss = F.cross_entropy(style, labels[5].to(self.device))
            # 总共的Loss 可以设置权重 total_loss = 2*hair_loss + 3*gender_loss + ......
            total_loss = hair_loss + gender_loss + earring_loss + smile_loss + frontal_loss + style_loss
            total_loss.backward()  # 误差回传
            self.optimer.step()  # 更新参数
            self.optimer.zero_grad()  # 梯度归零
            temp_loss += total_loss.item()  # 累加这个batch(这批数据)的loss
            # 打印loss信息
            if (batch_idx + 1) % (len(self.train_loader)//4) == 0:
                print("Epoch: %d/%d, training batch_idx:%d , time: %s, loss: %.4f" % (
                    epoch, self.epoches, batch_idx + 1, timeSince(self.start_time), total_loss.item()))
        # 返回epoch_loss
        return temp_loss / (batch_idx + 1)

    def evaluate(self):  # 在测试集上进行评估性能
        self.model.eval()

        correct_dict = {}  # 统计正确率
        predict_dict = {}  # 保存预测值
        label_dict = {}  # 保存label
        for attr in self.selected_attrs:
            correct_dict[attr] = 0
            predict_dict[attr] = list()
            label_dict[attr] = list()

        with torch.no_grad():
            for batch_idx, data in enumerate(self.test_loader):
                images, labels = data
                images = images.to(self.device)
                hair, gender, earring, smile, frontal, style = self.model(images)
                out_dict = {'hair': hair, 'gender': gender, 'earring': earring,
                            'smile': smile, 'frontal_face': frontal,
                            'style': style}
                batch = len(out_dict['hair'])
                for i in range(batch):
                    for attr_idx, attr in enumerate(self.selected_attrs):
                        pred = np.argmax(out_dict[attr][i].data.cpu().numpy())  # 得到预测值
                        true_label = labels[attr_idx].data.cpu().numpy()[i]  # 得到label
                        if pred == true_label:
                            correct_dict[attr] = correct_dict[attr] + 1
                        predict_dict[attr].append(pred)
                        label_dict[attr].append(true_label)
        mAP = 0
        for attr in self.selected_attrs:
            correct_dict[attr] = correct_dict[attr] * 100 / (len(self.test_loader) * self.batch_size)
            mAP += correct_dict[attr]
        mAP /= len(self.selected_attrs)
        return correct_dict, mAP, predict_dict, label_dict

    def fit(self, model_path=None):
        # 如果有参数文件
        if model_path is not None:
            # 加载模型文件
            self.model.load_state_dict(torch.load(model_path))
            print("加载参数文件: {}".format(model_path))
        # 记录最好的模型以及正确率
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        # 记录训练过程的损失
        train_losses = []
        # 统计每个epoch的正确率
        eval_acc_dict = {}
        for attr in self.selected_attrs:
            eval_acc_dict[attr] = []

        self.start_time = time.time()
        for epoch in range(self.epoches):
            running_loss = self.train(epoch)
            if epoch > self.epoches // 2:
                self.scheduler.step()
            print("Epoch: %d, time: %s, loss: %.4f , lr:%.7f" % (epoch, timeSince(self.start_time), running_loss, self.learning_rate))
            correct_dict, mAP, predict_dict, label_dict = self.evaluate()
            print("Epoch: {} accuracy:{}".format(epoch, correct_dict))
            print("Epoch: {} mAP: {}".format(epoch, mAP))
            train_losses.append(running_loss)
            for attr in self.selected_attrs:
                eval_acc_dict[attr].append(correct_dict[attr])

            # 比较正确率并保存最佳模型
            if mAP > best_acc:
                best_acc = mAP
                best_model_wts = copy.deepcopy(self.model.state_dict())
                best_predict_dict = predict_dict
                best_label_dict = label_dict

        # 保存每个epoch的每个属性的正确率
        eval_acc_csv = pd.DataFrame(eval_acc_dict, index=[i for i in range(self.epoches)]).T
        eval_acc_csv.to_csv("./result/" + self.model_type + "-eval_accuracy" + ".csv")
        # 报错训练过程的loss
        train_losses_csv = pd.DataFrame(train_losses)
        train_losses_csv.to_csv("./result/" + self.model_type + "-losses" + ".csv")
        # 保存best model
        model_save_path = "./result/" + self.model_type + "-best_model_params" + ".pth"
        torch.save(best_model_wts, model_save_path)
        print("The model has saved in {}".format(model_save_path))
        # 保存预测值
        pred_csv = pd.DataFrame(best_predict_dict)
        pred_csv.to_csv("./result/" + self.model_type + "-predict" + ".csv")
        # 保存真实值
        label_csv = pd.DataFrame(best_label_dict)
        label_csv.to_csv("./result/" + self.model_type + "-label" + ".csv")
        # 保存模型信息
        report_dict = {}
        report_dict["model"] = self.model_type
        report_dict["best_mAP"] = best_acc
        report_dict["lr"] = self.learning_rate
        report_dict["optim"] = 'Adam'
        report_dict['Batch_size'] = self.batch_size
        report_json = json.dumps(report_dict)
        report_file = open("./result/" + self.model_type + "-report.json", 'w')
        report_file.write(report_json)
        report_file.close()
        print("完成")
