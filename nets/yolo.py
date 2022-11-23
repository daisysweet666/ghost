"""yolo预测头部分"""

import torch
import torch.nn as nn
from collections import OrderedDict
from .ghostnet import ghostnet


class GhostNet(nn.Module):
    def __init__(self, pretrained=True):
        super(GhostNet, self).__init__()
        model = ghostnet()
        if pretrained:
            state_dict = torch.load("model_data/ghostnet_weights.pth")
            model.load_state_dict(state_dict)
        del model.global_pool
        del model.conv_head
        del model.act2
        del model.classifier
        del model.blocks[9]
        self.model = model

    def forward(self, x):
        x = self.model.conv_stem(x)
        x = self.model.bn1(x)
        x = self.model.act1(x)
        feature_maps = []

        for idx, block in enumerate(self.model.blocks):
            x = block(x)
            if idx in [2,4,6,8]:
                feature_maps.append(x)
        return feature_maps[1:]


# 卷积 + 上采样
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()
        self.upsample = nn.Sequential(conv2d(in_channels, out_channels, 1),
                                      nn.Upsample(scale_factor=2, mode='nearest'))

    def forward(self, x,):
        x = self.upsample(x)
        return x


def conv2d(filter_in, filter_out, kernel_size, groups=1, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride,
                           padding=pad, groups=groups, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.ReLU6(inplace=True)),
    ]))


def conv_dw(filter_in, filter_out, stride=1):
    return nn.Sequential(
        nn.Conv2d(filter_in, filter_in, 3, stride, 1, groups=filter_in, bias=False),
        nn.BatchNorm2d(filter_in),
        nn.ReLU6(inplace=True),

        nn.Conv2d(filter_in, filter_out, 1, 1, 0, bias=False),
        nn.BatchNorm2d(filter_out),
        nn.ReLU6(inplace=True),
    )


def make_five_conv(filters_list, in_filters):  # 五次卷积块
    m = nn.Sequential(conv2d(in_filters, filters_list[0], 1),
                      conv_dw(filters_list[0], filters_list[1]),
                      conv2d(filters_list[1], filters_list[0], 1),
                      conv_dw(filters_list[0], filters_list[1]),
                      conv2d(filters_list[1], filters_list[0], 1),)
    return m


def yolo_head(filters_list, in_filters):    # 最后获得yolov4的输出
    m = nn.Sequential(conv_dw(in_filters, filters_list[0]),
                      nn.Conv2d(filters_list[0], filters_list[1], 1),)
    return m


#   yolo_body
class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, backbone="ghostnet", pretrained=False):
        super(YoloBody, self).__init__()
        #   生成主干模型，获得三个有效特征层。
        if backbone == "ghostnet":
            #   52,52,40；26,26,112；13,13,160
            self.backbone = GhostNet(pretrained=pretrained)
            in_filters = [40, 112, 160]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use ghostnet.'.format(backbone))

        self.upsample1 = Upsample(512, 256)
        self.upsample2 = Upsample(256, 128)
        self.conv_for_P4 = conv2d(in_filters[1], 256, 1)
        self.conv_for_P3 = conv2d(in_filters[0], 128, 1)
        self.make_five_conv1 = make_five_conv([512, 1024], in_filters[2])
        self.make_five_conv2 = make_five_conv([256, 512], 512)
        self.make_five_conv3 = make_five_conv([128, 256], 256)
        self.yolo_head3 = yolo_head([256, len(anchors_mask[0]) * (5 + num_classes)], 128)
        self.yolo_head2 = yolo_head([512, len(anchors_mask[1]) * (5 + num_classes)], 256)
        self.yolo_head1 = yolo_head([1024, len(anchors_mask[2]) * (5 + num_classes)], 512)

    def forward(self, x):
        #  backbone  有效特征层的输出：stage3:：52,52,40 stage4：26,26,112 stage5：13,13,160
        x2, x1, x0 = self.backbone(x)
        # 13,13,1024 -> 13,13,512 (P5 yolo头)
        P5 = self.make_five_conv1(x0)
        # 13,13,512 -> 13,13,256 -> 26,26,256
        P5_upsample = self.upsample1(P5)

        P4 = self.conv_for_P4(x1)
        # 26,26,256 + 26,26,256 -> 26,26,512
        P4 = torch.cat([P4, P5_upsample], axis=1)
        # 26,26,512 -> 26,26,256 (P4 yolo头)
        P4 = self.make_five_conv2(P4)
        # 26,26,256 ->26,26,128 -> 52,52,128
        P4_upsample = self.upsample2(P4)
        P3 = self.conv_for_P3(x2)
        # 52,52,128 + 52,52,128 -> 52,52,256
        P3 = torch.cat([P3, P4_upsample], axis=1)
        # 52,52,256 -> 52,52,128 (P3 yolo头)
        P3 = self.make_five_conv3(P3)

        out2 = self.yolo_head3(P3)  # 52,52,128
        out1 = self.yolo_head2(P4)  # 26,26,256
        out0 = self.yolo_head1(P5)  # 13,13,512

        return out0, out1, out2
