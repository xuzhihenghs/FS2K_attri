"""
全局变量
"""
# 基本不变的参数
root = 'FS2K/sketch'  # 图像的根目录
selected_attrs = ['hair', 'gender', 'earring', 'smile', 'frontal_face', 'style']  # 选择的属性
json_train_path = 'FS2K/anno_train.json'
json_test_path = 'FS2K/anno_test.json'

# 可能需要更改的参数
DEVICE_ID = '0'  # 显卡ID
Epoches = 32
batch_szie = 16
lr = 1e-5

model_type = 'VGG16'
