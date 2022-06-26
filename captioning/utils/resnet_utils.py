import torch
import torch.nn as nn
import torch.nn.functional as F

class myResnet(nn.Module):
    def __init__(self, resnet):
        super(myResnet, self).__init__()
        self.resnet = resnet

    def forward(self, img, att_size=14):
        x = img.unsqueeze(0)

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x) # torch.Size([2, 2048, 7, 7])

        fc = x.mean(3).mean(2).squeeze()
        att = F.adaptive_avg_pool2d(x,[att_size,att_size]).squeeze().permute(1, 2, 0)
        return fc, att



# Resnet各层输出形状： https://www.jianshu.com/p/58a3d0a7dabf
# In [51]: img.shape
# Out[51]: torch.Size([2, 3, 224, 224])

# In [52]: res_conv1(img).shape
# Out[52]: torch.Size([2, 64, 112, 112])

# In [53]: res_conv1_maxpool(img).shape
# Out[53]: torch.Size([2, 64, 56, 56])

# In [54]: res_layer1(img).shape
# Out[54]: torch.Size([2, 256, 56, 56])

# In [55]: res_layer2(img).shape
# Out[55]: torch.Size([2, 512, 28, 28])

# In [56]: res_layer3(img).shape
# Out[56]: torch.Size([2, 1024, 14, 14])

# In [57]: res_layer4(img).shape
# Out[57]: torch.Size([2, 2048, 7, 7])

# In [58]: res_avgpool(img).shape
# Out[58]: torch.Size([2, 2048, 1, 1])

# In [59]: resnet101(img).shape
# Out[59]: torch.Size([2, 1000])



        # self.classifier = nn.Sequential(
        #     nn.Linear(in_features=25088, out_features=4096, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.5, inplace=False),
        #     nn.Linear(in_features=4096, out_features=4096, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.5, inplace=False),
        #     # # # nn.Linear(in_features=4096, out_features=1000, bias=True)
        # )

