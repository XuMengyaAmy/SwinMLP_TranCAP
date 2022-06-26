import os
import numpy as np
import torch
import torch.nn
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable 
import torch.cuda
import torchvision.transforms as transforms
from PIL import Image

import random
from glob import glob

TARGET_IMG_SIZE = 224
img_to_tensor = transforms.ToTensor() # ToTensor: get value within 0 to 1

# Downloading: "https://download.pytorch.org/models/vgg16-397923af.pth" to /home/ren2/.cache/torch/hub/checkpoints/vgg16-397923af.pth

def seed_everything(seed=1234):
    print('=================== set the seed :', seed)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# def make_model():
#     # model=models.vgg16(pretrained=True).features[:28]	# features[:28] 其实就是定位到第28层，对照着上面的key看就可以理解
#     model=models.vgg16(pretrained=True).features
#     model=model.eval()	# 一定要有这行，不然运算速度会变慢（要求梯度）而且会影响结果
#     model.cuda()	# 将模型从CPU发送到GPU,如果没有GPU则删除该行
#     return model

# def make_model():
#     # 快速去除预训练模型本身的网络层并添加新的层 https://blog.csdn.net/weixin_42118374/article/details/103761795
#     # model=models.vgg16(pretrained=True).features[:28]	# features[:28] 其实就是定位到第28层，对照着上面的key看就可以理解
#     model=models.vgg16(pretrained=True)
#     #先将模型的网络层列表化，每一层对应列表一个元素，bottleneck对应一个序列层
#     net_structure = list(model.children()) # 3 elements in this list: ['features block', 'avgpool', 'classifier block']
#     print(net_structure)
#     # #去除最后两层得到新的网络
#     model = nn.Sequential(*net_structure[:-2])

#     model= model.eval()	# 一定要有这行，不然运算速度会变慢（要求梯度）而且会影响结果
#     model.cuda()	# 将模型从CPU发送到GPU,如果没有GPU则删除该行
#     return model

# https://blog.csdn.net/weixin_43436958/article/details/107297639
# self-defined VGG16 is used to extract image features
class VGG16_modified(nn.Module):
    def __init__(self):
        super(VGG16_modified, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            # # # nn.Linear(in_features=4096, out_features=1000, bias=True)
        )
    def forward(self, out):
        in_size = out.size(0)
        out= self.features (out)
        out = self.avgpool (out)
        out = out.view(in_size, -1)#拉平 # flatten 512*7*7 into 25088
        out = self.classifier(out)
        return out


def convert_vgg(vgg16):#vgg16是pytorch自带的
    # # https://blog.csdn.net/wangbin12122224/article/details/79949965
    model = VGG16_modified()# 我写的vgg

    vgg_items = model.state_dict().items()
    vgg16_items = vgg16.items()

    pretrain_model = {}
    j = 0
    for k, v in model.state_dict().items():#按顺序依次填入 # 去掉iter前缀即可
        v = list(vgg16_items)[j][1]
        k = list(vgg_items)[j][0]
        pretrain_model[k] = v
        j += 1
    return pretrain_model

def make_model():
    ## model 是我们最后使用的网络，也是我们想要放置weights的网络
    model = VGG16_modified()

    print('load the weight from vgg')
    pretrained_dict = torch.load('/home/ren2/.cache/torch/hub/checkpoints/vgg16-397923af.pth') # 'vgg16.pth'
    pretrained_dict = convert_vgg(pretrained_dict)

    model_dict = model.state_dict()
    # 1. 把不属于我们需要的层剔除
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. 把参数存入已经存在的model_dict
    model_dict.update(pretrained_dict) 
    # 3. 加载更新后的model_dict
    model.load_state_dict(model_dict)
    print ('copy the weight sucessfully')

    model= model.eval()	# 一定要有这行，不然运算速度会变慢（要求梯度）而且会影响结果
    model.cuda()	# 将模型从CPU发送到GPU,如果没有GPU则删除该行
    return model
    
#特征提取
# VGG feature extraction code is refer to https://blog.csdn.net/Geek_of_CSDN/article/details/84343971
# ResNet feature extraction code is refer to https://blog.csdn.net/u010165147/article/details/72829969
# VGG specific layer: https://blog.csdn.net/weixin_43687366/article/details/102536391
# Self-defined VGG16: https://blog.csdn.net/weixin_43436958/article/details/107297639
# Use self-defined VGG16 to extract features: https://www.codeleading.com/article/1302195801/
# 去掉模型的一些层，添加新的层 https://blog.csdn.net/weixin_42118374/article/details/103761795
# 灵活的特征提取，直到Sequential 里的层： https://blog.csdn.net/wangbin12122224/article/details/79949965
def extract_feature(model,imgpath):
    model.eval()		# 必须要有，不然会影响特征提取结果
    
    img=Image.open(imgpath)		# 读取图片
    img=img.resize((TARGET_IMG_SIZE, TARGET_IMG_SIZE))
    tensor=img_to_tensor(img)	# 将图片转化成tensor
    tensor=tensor.cuda()	# 如果只是在cpu上跑的话要将这行去掉
    # print('tensor before:', tensor)
    print('tensor before:', tensor.shape) # tensor before: torch.Size([3, 224, 224])
    
    tensor = tensor.unsqueeze(0) # convert 3-dimension into 4-dimension
    print('tensor after:', tensor.shape) # tensor after: torch.Size([1, 3, 224, 224])
    
    result=model(Variable(tensor))
    result_npy=result.data.cpu().numpy()	# 保存的时候一定要记得转成cpu形式的，不然可能会出错
    
    return result_npy[0]	# 返回的矩阵shape是[1, 512, 14, 14]，这么做是为了让shape变回[512, 14,14]

if __name__=="__main__":
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
    num_gpu = torch.cuda.device_count()
    print('The number of GPUs used:', num_gpu)
    
    model=make_model()

    # To collect the imgpath
    dataset_dir = '/home/ren2/data2/mengya/mengya_dataset/DAISI_High_Res_Splited'
    img_train_dir = os.path.join(dataset_dir, 'Train', 'images')
    img_test_dir = os.path.join(dataset_dir, 'Test', 'images')

    img_train_list = glob(img_train_dir + '/*.jpg')
    random.shuffle(img_train_list)

    img_test_list = glob(img_test_dir + '/*.jpg')
    random.shuffle(img_test_list)
    
    print('train size:', len(img_train_list), 'test size: ', len(img_test_list))
    
    for i in range(len(img_train_list)):
        # imgpath='/home/ren2/data2/mengya/mengya_dataset/DAISI_High_Res/001/001.jpg'
        imgpath = img_train_list[i]
        img_name = os.path.splitext(os.path.basename(imgpath))[0]
        print('img_name', img_name)
        feature = extract_feature(model, imgpath)
        print('feature', feature.shape)	# 打印出得到的tensor的shape (512, 14, 14),   (512, 7, 7) from whole feature block

        save_data_path = os.path.join(dataset_dir, 'features', 'Train')
        # if(not os.path.exists(save_data_path)):
        #     os.mkdir(save_data_path)

        np.save(os.path.join(save_data_path, '{}'.format(img_name)), feature)
        print('save_data_path:', save_data_path)

    for i in range(len(img_test_list)):
        # imgpath='/home/ren2/data2/mengya/mengya_dataset/DAISI_High_Res/001/001.jpg'
        imgpath = img_test_list[i]
        img_name = os.path.splitext(os.path.basename(imgpath))[0]
        print('img_name', img_name)
        feature = extract_feature(model, imgpath)
        print('feature', feature.shape)	# 打印出得到的tensor的shape (512, 14, 14),   (512, 7, 7) from whole feature block
        
        save_data_path = os.path.join(dataset_dir, 'features', 'Test')
        # if(not os.path.exists(save_data_path)):
        #     os.mkdir(save_data_path)
        np.save(os.path.join(save_data_path, '{}'.format(img_name)), feature)
        print('save_data_path:', save_data_path)


# if __name__ == '__main__':
#     model = models.vgg16(pretrained=True)
#     # ========= Check the model architectures ============ #
#     feature = torch.nn.Sequential(*list(model.children())[:])
#     print(feature)
#     print(model._modules.keys()) # odict_keys(['features', 'avgpool', 'classifier'])

# VGG architecture:
# Sequential(
#   (0): Sequential(
#     (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (1): ReLU(inplace=True)
#     (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (3): ReLU(inplace=True)
#     (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (6): ReLU(inplace=True)
#     (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (8): ReLU(inplace=True)
#     (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (11): ReLU(inplace=True)
#     (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (13): ReLU(inplace=True)
#     (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (15): ReLU(inplace=True)
#     (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (18): ReLU(inplace=True)
#     (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (20): ReLU(inplace=True)
#     (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (22): ReLU(inplace=True)
#     (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (25): ReLU(inplace=True)
#     (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (27): ReLU(inplace=True)
#     (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (29): ReLU(inplace=True)
#     (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (1): AdaptiveAvgPool2d(output_size=(7, 7))
#   (2): Sequential(
#     (0): Linear(in_features=25088, out_features=4096, bias=True)
#     (1): ReLU(inplace=True)
#     (2): Dropout(p=0.5, inplace=False)
#     (3): Linear(in_features=4096, out_features=4096, bias=True)
#     (4): ReLU(inplace=True)
#     (5): Dropout(p=0.5, inplace=False)
#     (6): Linear(in_features=4096, out_features=1000, bias=True)
#   )
# )



