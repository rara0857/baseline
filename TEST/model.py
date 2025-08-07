import os
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

def conv3x3_relu(inplanes, planes,stride =1, rate=1):
    conv3x3_relu = nn.Sequential(
                                 nn.Conv2d(inplanes, planes, kernel_size=3,
                                           stride=stride, padding=rate, dilation=rate),
                                 nn.BatchNorm2d(planes),
                                 nn.ReLU())
    return conv3x3_relu


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1), #torch.Size([2, 32, 16, 16])----->torch.Size([2, 32, 1, 1]) same channel
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels,out_channels, atrous_rates):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))
        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))
    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

def central_crop_layer(input_maps, central_fraction=0.25):
    img_shape = input_maps.shape
    depth = img_shape[1]
    img_h = img_shape[2]
    img_w = img_shape[3]

    bbox_h_start = np.int32((img_h - img_h * central_fraction) / 2)
    bbox_w_start = np.int32((img_w - img_w * central_fraction) / 2)

    bbox_batch_size = img_shape[0]
    bbox_h_size = img_shape[2] - bbox_h_start * 2
    bbox_w_size = img_shape[3] - bbox_w_start * 2

    output_maps = input_maps[:bbox_batch_size,:depth,bbox_h_start:bbox_h_start+bbox_h_size,bbox_w_start:bbox_w_start+bbox_w_size]

    return output_maps


class net_20x(nn.Module):
    def __init__(self):
        super(net_20x, self).__init__()
        self.conv1_1 = conv3x3_relu(3,64)
        self.conv1_2 = conv3x3_relu(64,64)
        self.conv1_3 = conv3x3_relu(64,64,stride=2)

        self.conv2_1 = conv3x3_relu(64, 128)
        self.conv2_2 = conv3x3_relu(128, 128)
        self.conv2_3 = conv3x3_relu(128, 128, stride=2)

        self.conv3_1 = conv3x3_relu(128, 256)
        self.conv3_2 = conv3x3_relu(256, 256)
        self.conv3_3 = conv3x3_relu(256, 256)
        self.conv3_4 = conv3x3_relu(256, 256, stride=2)

        self.conv4_1 = conv3x3_relu(256, 512)
        self.conv4_2 = conv3x3_relu(512, 512)
        self.conv4_3 = conv3x3_relu(512, 512)
        self.conv4_4 = conv3x3_relu(512, 512, stride=2)

        self.conv5_1 = conv3x3_relu(512, 512)
        self.conv5_2 = conv3x3_relu(512, 512)
        self.conv5_3 = conv3x3_relu(512, 512)
        self.conv5_4 = conv3x3_relu(512, 512)
        self.conv5_5 = conv3x3_relu(512, 512)

    def forward(self, inputs):
        x = self.conv1_1(inputs)
        x = self.conv1_2(x)
        x = self.conv1_3(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.conv3_4(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.conv4_4(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.conv5_4(x)

        return x

class net_5x(nn.Module):
    def __init__(self):
        super(net_5x, self).__init__()
        self.conv1_1 = conv3x3_relu(3,64)
        self.conv1_2 = conv3x3_relu(64,64)
        self.conv1_3 = conv3x3_relu(64,64,stride=2)

        self.conv2_1 = conv3x3_relu(64, 128)
        self.conv2_2 = conv3x3_relu(128, 128)
        self.conv2_3 = conv3x3_relu(128, 128, stride=2)

        self.conv3_1 = conv3x3_relu(128, 256)
        self.conv3_2 = conv3x3_relu(256, 256)
        self.conv3_3 = conv3x3_relu(256, 256)
        self.conv3_4 = conv3x3_relu(256, 256, stride=2)

        self.conv4_1 = conv3x3_relu(256, 512)
        self.conv4_2 = conv3x3_relu(512, 512)
        self.conv4_3 = conv3x3_relu(512, 512)
        self.conv4_4 = conv3x3_relu(512, 512, stride=2)

        self.conv5_1 = conv3x3_relu(512, 512)
        self.conv5_2 = conv3x3_relu(512, 512)
        self.conv5_3 = conv3x3_relu(512, 512)
        self.conv5_4 = conv3x3_relu(512, 512)
        self.conv5_5 = conv3x3_relu(512, 512)

        self.aspp_block = ASPP(512, 512, atrous_rates = [3, 6, 8])


    def forward(self, inputs):
        x = self.conv1_1(inputs)
        x = self.conv1_2(x)
        x = self.conv1_3(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.conv3_4(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.conv4_4(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        output_5_3 = self.conv5_3(x)

        aspp = self.aspp_block(output_5_3)
        updsample_size = aspp.shape[2:]
        crop = central_crop_layer(aspp,central_fraction=0.25)
        upsample = nn.Upsample(size= updsample_size, mode='bilinear',align_corners=True)(crop)
        return upsample


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class net_20x_5x(nn.Module):
    def __init__(self,num_classes):
        super(net_20x_5x, self).__init__()
        # end_points = {}
        self.net_20x = net_20x()
        self.net_5x = net_5x()
        self.SE_block = SELayer(512*2,reduction=16)
        self.conv6_1 = conv3x3_relu(512*2,512)
        self.conv6_2 = conv3x3_relu(512, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self,input_20x,input_5x):
        output_20x = self.net_20x(input_20x)
        output_5x = self.net_5x(input_5x)
#         print(output_20x.shape,output_5x.shape)
        output_concat = torch.cat([output_20x,output_5x], dim=1)

        # for SE block
        # output_SE = self.SE_block(output_concat)
        # output_conv6_1 = self.conv6_1(output_SE)
        
        output_conv6_1 = self.conv6_1(output_concat)
        output_conv6_2 = self.conv6_2(output_conv6_1)
        output_avgpool = self.avgpool(output_conv6_2)
        flatten = torch.flatten(output_avgpool, 1)
        output_feature = self.classifier(flatten)

        return output_feature


# from torchsummary import summary
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = net_20x_5x(num_classes=2).to(device)
# summary(model,[(3,256,256),(3,256,256)])

# from torchsummary import summary
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = net_20x_5x(num_classes=2).to(device)
# summary(model,[(3,256,256),(3,256,256)])
# -



