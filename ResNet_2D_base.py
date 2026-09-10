import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

# __all__ 是一个特殊的Python列表，用于控制模块的公开接口。它定义了当使用from module import *时，哪些名称会被导入。
__all__ = [
    'ResNet', 'resnet10_2D', 'resnet18_2D', 'resnet34_2D', 'resnet50_2D', 'resnet101_2D',
    'resnet152_2D', 'resnet200_2D'
]

def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,           # 输入通道数
        out_planes,          # 输出通道数
        kernel_size=3,       # 3x3卷积核
        stride=stride,       # 卷积步长
        padding=dilation,    # 填充大小，确保输出尺寸与dilation相关
        bias=False,          # 不使用bias（通常与BN一起使用）
        dilation=dilation,   # 膨胀率（空洞卷积）
    )

def downsample_basic_block(x, planes, stride):
    """
    使用平均池化和零填充实现下采样，用于ResNet快捷连接类型A
    
    参数:
        x: 输入特征图，形状为(batch_size, channels, height, width)
        planes: 目标输出通道数
        stride: 池化步长，决定下采样倍数
    
    返回:
        Variable: 下采样后的特征图，通道数扩展到planes
    """
    # 1x1平均池化，如果stride=1，不改变特征图大小。如果stride>=2，则下采样特征图大小
    out = F.avg_pool2d(x, kernel_size=1, stride=stride)  # float16
    zero_pads_shape = (out.size(0), planes - out.size(1), out.size(2), out.size(3))
    zero_pads = torch.zeros(zero_pads_shape, dtype=out.dtype, device=out.device)
    out = torch.cat([out, zero_pads], dim=1) # 直接返回拼接结果，不再使用 Variable

    return out


class BasicBlock(nn.Module):
    expansion = 1 # 基本块的扩展因子为1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    '''
        实现ResNet 2D模型，支持不同深度（10、18、34、50、101、152、200），深度代表模型可学习参数的层的数量
    '''
    def __init__(self,
                 block,   # 残差块类(BasicBlock 或 Bottleneck)
                 layers,  # 每个stage中残差块的数量
                 input_channels,     # 输入数据的通道数
                 input_size,  # 输入数据的高度 宽度
                 shortcut_type='B'):  # shortcut类型 ('A' 无参下采样 或 'B' 有参下采样)
        self.inplanes = 64           # 初始通道数
        super(ResNet, self).__init__() # 调用父类的初始化方法

        # stem部分 二维卷积
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=64,
            kernel_size=7,
            stride=(2, 2),
            padding=(3, 3),
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1,dilation=1)

        # 四个stage部分
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2, dilation=1)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2, dilation=1)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2, dilation=1)
        # stride=2 标准实现
        # dilation=2/4,会扩大感受野，卷积范围，但是局部细节信息被"平均化" 对小物体或细粒度特征不敏感，设置为标准的1即可

        # ResNet原始实现中，结尾部分，平均池化和全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_seg_classes)
    

        # 初始化二维卷积层和二维批归一化层的权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')  # 过时方法，会产生警告
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        """
        构建一个残差块层，包含多个连续的残差块
        
        参数:
            block: 残差块类型 (BasicBlock 或 Bottleneck)
            planes: 输出通道数
            blocks: 该层包含的残差块数量
            shortcut_type: 快捷连接类型 ('A' 或 'B')
            stride: 步长，用于下采样 (默认1)
            dilation: 空洞卷积率 (默认1)
        
        返回:
            nn.Sequential: 包含多个残差块的序列
        """
        downsample = None # 下采样层
        if stride != 1 or self.inplanes != planes * block.expansion:
            # 若步长不为1 或 输入通道数与输出通道数不匹配，需要下采样。
            # 因为残差连接需要匹配通道数和空间尺寸
            # 如果卷积的步长 stride ≠ 1，会改变特征图的空间大小（H、W）；
            # 如果 in_channels ≠ out_channels，会改变通道数 C；
            if shortcut_type == 'A':
                # 类型A：使用平均池化和零填充进行下采样，无参数下采样
                downsample = partial(   # 部分应用函数，预先固定参数。
                    downsample_basic_block,          # 下采样基本块类
                    planes=planes * block.expansion, # 预先固定 planes 参数 输出通道数
                    stride=stride)                   # 预先固定 stride 参数 步长
            else:
                # 类型B：使用1x1卷积和批归一化进行下采样，有参数下采样
                downsample = nn.Sequential(
                    nn.Conv2d(
                        in_channels = self.inplanes, # 输入通道数
                        out_channels = planes * block.expansion, # 输出通道数
                        kernel_size=1, # 卷积核大小
                        stride=stride, # 步长
                        bias=False), # 偏置项
                    nn.BatchNorm2d(planes * block.expansion)) # 批归一化层

        layers = []
        # 第一个残差块：处理下采样和通道数变化
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        # 更新输入通道数为当前层的输出通道数
        self.inplanes = planes * block.expansion
        # 添加剩余的残差块（步长为1，不进行下采样）
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        # 将所有残差块组合成序列
        return nn.Sequential(*layers)

    def forward(self, x):
        # stem部分
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 四个stage
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 原始实现 结尾部分
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x


def resnet10_2D(**kwargs):
    """Constructs a ResNet-10 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18_2D(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34_2D(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50_2D(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101_2D(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152_2D(**kwargs):
    """Constructs a ResNet-152 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200_2D(**kwargs):
    """Constructs a ResNet-200 model.
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model

if __name__ == '__main__':
    # 图像尺寸
    # MR (3, 144, 320, 320)
    # MG (3, 640, 512)
    # US (3, 384, 512)
    # input_size=(224, 224)
    input_size=(640, 512)
    # input_size=(384, 512)

    # 这里创建二维的resnet18
    # net = resnet18_2D(input_channels=2, input_size=input_size)
    net = resnet34_2D(input_channels=2, input_size=input_size)
    # net = resnet50_2D(input_channels=2, input_size=input_size)
    a = torch.rand(2, 2, input_size[0], input_size[1])
    b = net(a)
    print(b.shape) # bs=1也可运行，bn1的eps允许，但是基本学不到东西