"""yolo的backbone部分"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# __all__ = ['ghostnet']


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        # 简单的注意力机制 和SEnet相似
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, act_layer=nn.ReLU):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        # ratio一般指定为2，才能保证输出层的通道数等于exp
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        # 利用1x1卷积对输入特征图进行通道缩减，获得特征浓缩，跨通道特征提取
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        # 获得特征浓缩后使用逐层卷积，获得额外特征图(group)，跨特征点特征提取
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        # 将1x1卷积后的结果，和逐层卷积后的结果进行堆叠
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class GhostBottleneck(nn.Module):  # Ghost模块构成的瓶颈结构
    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3, stride=1, act_layer=nn.ReLU, se_ratio=0.):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride
        # 利用ghost模块进行特征提取，此时指定的通道数会比较大，可以看做是逆残差结构
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)

        if self.stride > 1:        # 步长为2：逐层卷积，进行特征图的高宽压缩
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                                     padding=(dw_kernel_size - 1) // 2,
                                     groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        if has_se:          # 是否使用注意力机制
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # 再次利用一个ghost模块进行特征提取
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)
        if in_chs == out_chs and self.stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                          padding=(dw_kernel_size - 1) // 2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x
        # 首先利用一个ghost模块进行特征提取，此时指定的通道数会比较大，可以看做是逆残差结构
        x = self.ghost1(x)
        # 是否进行特征图的高宽压缩（使用逐层卷积）
        if self.stride > 1:
            x = self.conv_dw(x)  # dw：深度可分离卷积
            x = self.bn_dw(x)
        # 进行注意力机制
        if self.se is not None:
            x = self.se(x)
        # 再次利用ghost模块进行特征提取
        x = self.ghost2(x)

        x += self.shortcut(residual)
        return x


class GhostNet(nn.Module):
    def __init__(self, cfgs, num_classes=1, width=1.0, dropout=0.2):
        super(GhostNet, self).__init__()
        # 一些基础设置
        self.cfgs = cfgs
        self.dropout = dropout

        # 第一个卷积标准化+激活函数
        output_channel = _make_divisible(16 * width, 4)
        # yolov4：416,416,3  ->  208,208,16
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)  # 进行高宽的压缩
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)

        # 计算瓶颈结构的输入
        input_channel = output_channel
        stages = []
        block = GhostBottleneck
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                # 根据cfg里面的内容构建瓶颈结构
                layers.append(block(input_channel, hidden_channel, output_channel, k, s,
                                    se_ratio=se_ratio))
                # 计算下一个瓶颈结构的输入
                input_channel = output_channel
            stages.append(nn.Sequential(*layers))
        # 卷积标准化+激活函数 （进行通道数的调整）
        output_channel = _make_divisible(exp_size * width, 4)
        stages.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1)))
        # 根据构建好的block序列模型
        self.blocks = nn.Sequential(*stages)
        # 构建分类层
        input_channel = output_channel
        output_channel = 1280
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_head = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=True)
        self.act2 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(output_channel, num_classes)

    def forward(self, x):
        # 第一个卷积标准化+激活函数
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.blocks(x)

        x = self.global_pool(x)

        x = self.conv_head(x)
        x = self.act2(x)
        x = x.view(x.size(0), -1)
        if self.dropout > 0.:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x

def ghostnet(**kwargs):
    """
    有效特征层的输出：stage3:：52,52,40 stage4：26,26,112 stage5：13,13,160
    """
    cfgs = [
        # k：卷积核大小, 表示跨特征点的特征提取能力
        # t：第一个ghost模块所设置的通道数大小,值会比较大
        # c：瓶颈结构的最终的输出通道数,
        # SE：是否使用注意力机制（不为0则使用）,
        # s：代表步长，如果为2 会对输入进来的特征图进行高和宽的压缩。
        # stage1: 208,208,16 -> 208,208,16
        [[3, 16, 16, 0, 1]],
        # stage2: 208,208,16 -> 104,104,24
        [[3, 48, 24, 0, 2]],
        [[3, 72, 24, 0, 1]],
        # stage3: 104,104,24 -> 52,52,40
        [[5, 72, 40, 0.25, 2]],
        [[5, 120, 40, 0.25, 1]],
        # stage4: 52,52,40 -> 26,26,80
        #         26,26,80 -> 26,26,112
        [[3, 240, 80, 0, 2]],
        [[3, 200, 80, 0, 1],
         [3, 184, 80, 0, 1],
         [3, 184, 80, 0, 1],
         [3, 480, 112, 0.25, 1],
         [3, 672, 112, 0.25, 1]],
        # stage5: 26,26,112 -> 13,13,160
        [[5, 672, 160, 0.25, 2]],
        [[5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1],
         [5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1]
         ]
    ]
    return GhostNet(cfgs, **kwargs)


# if __name__ == "__main__":
#     from torchsummary import summary
#
#     # 需要使用device来指定网络在GPU还是CPU运行
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = ghostnet().to(device)
#     summary(model, input_size=(3, 224, 224))
