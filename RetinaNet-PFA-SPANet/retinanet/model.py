import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
from torchvision.ops import nms
import torch.nn.functional as F
from retinanet.attention import SpatialAttention, ChannelwiseAttention
from retinanet.utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
from retinanet.anchors import Anchors
from retinanet import losses

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class MultiScale(nn.Module):
    def __init__(self, in_channels, out_channels=64):
        super(MultiScale, self).__init__()

        self.dil_rates = [3, 5, 7]

        self.in_channels = in_channels

        # Define layers
        self.conv_1_1 = nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self.conv_dil_3 = nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels, kernel_size=3,
                                    stride=1, dilation=self.dil_rates[0], padding=self.dil_rates[0], bias=False)
        self.conv_dil_5 = nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels, kernel_size=3,
                                    stride=1, dilation=self.dil_rates[1], padding=self.dil_rates[1], bias=False)
        self.conv_dil_7 = nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels, kernel_size=3,
                                    stride=1, dilation=self.dil_rates[2], padding=self.dil_rates[2], bias=False)

        self.bn = nn.BatchNorm2d(out_channels*4)

    def forward(self, input_):
        # Extract features
        conv_1_1_feats = self.conv_1_1(input_)
        conv_dil_3_feats = self.conv_dil_3(input_)
        conv_dil_5_feats = self.conv_dil_5(input_)
        conv_dil_7_feats = self.conv_dil_7(input_)

        # Aggregate features
        concat_feats = torch.cat((conv_1_1_feats, conv_dil_3_feats, conv_dil_5_feats, conv_dil_7_feats), dim=1)
        bn_feats = F.relu(self.bn(concat_feats))
        # return 512 channels feature map
        return bn_feats


class CA_SA_Model(nn.Module):
    def __init__(self, layer1_size, layer2_size, layer3_size, layer4_size):
        super(CA_SA_Model, self).__init__()

        # Initialize layers for ResNet deep levels(hl) feature (layer3, layer4) processing
        self.MultiScale_layer3 = MultiScale(in_channels=layer3_size)
        self.MultiScale_layer4 = MultiScale(in_channels=layer4_size)

        self.cha_att = ChannelwiseAttention(in_channels=512)  # in_channels = 512

        self.hl_conv1 = nn.Conv2d(512, layer3_size, (3, 3), padding=1)
        self.hl_bn1 = nn.BatchNorm2d(layer3_size)

        # self.MultiScale_layer2 = MultiScale(in_channels=layer2_size) # 这个位置也要不同尺度的空洞卷积，若效果不行再change

        # down sample layer1 to merge with layer2
        self.ll_conv_1 = nn.Conv2d(layer1_size, layer2_size, kernel_size=1, stride=1)
        self.ll_bn_1 = nn.BatchNorm2d(layer2_size)

        # 这下面的操作是为了把layer1和layer2的特征图做一个concat，然后再经过特征融合
        self.ll_conv_2 = nn.Conv2d(2*layer2_size, layer2_size, kernel_size=3, stride=1, padding=1)
        self.ll_bn_2 = nn.BatchNorm2d(layer2_size)

        self.spa_att = SpatialAttention(in_channels=layer2_size)

        # "conv5 is obtained via a 3x3 stride-2 conv on conv_34_feats"
        self.conv_5 = nn.Conv2d(layer3_size, layer4_size, kernel_size=3, stride=2, padding=1)
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, inputs):
        feature_layer1, feature_layer2, feature_layer3, feature_layer4 = inputs
        # global layer1, layer2, layer3, layer4

        # Process high level features
        conv3_multiscale_feats = self.MultiScale_layer3(feature_layer3) # 256channels
        conv4_multiscale_feats = self.MultiScale_layer4(feature_layer4) # 256channels

        # 上采样，双线性插值，输入输出角点对齐
        conv4_multiscale_feats = F.interpolate(conv4_multiscale_feats, scale_factor=2, mode='bilinear', align_corners=True)

        conv_34_feats = torch.cat((conv3_multiscale_feats, conv4_multiscale_feats), dim=1) #512channels

        conv_34_ca = self.cha_att(conv_34_feats)
        conv_34_feats = torch.mul(conv_34_feats, conv_34_ca)

        conv_34_feats = self.hl_conv1(conv_34_feats)
        conv_34_feats = F.relu(self.hl_bn1(conv_34_feats))

        # Process low level features
        conv1_feats = self.ll_conv_1(feature_layer1) #channel升高一倍
        conv1_feats = F.interpolate(conv1_feats, scale_factor=0.5, mode='bilinear', align_corners=True) #下采样一倍

        conv_12_feats = torch.cat((conv1_feats, feature_layer2), dim=1) # 后面版本的torch需要加一个[]才可以融合

        conv_12_feats = self.ll_conv_2(conv_12_feats)
        conv_12_feats = F.relu(self.ll_bn_2(conv_12_feats))

        conv_12_sa = self.spa_att(conv_12_feats)
        conv_12_feats = torch.mul(conv_12_feats, conv_12_sa)

        # 把语义信息丰富的做一个步长为2的卷积代表fpn的第三层
        # conv5_feats = F.interpolate(conv_12_feats, scale_factor=0.5, mode='bilinear', align_corners=True)  # 下采样一倍
        conv_5_feats = self.conv_5(conv_34_feats) # 把512channels升高维度至2048
        return [conv_12_feats, conv_34_feats, conv_5_feats]

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        kernel_sizes = [1, 3, 3, 1]
        dilations = [1, 3, 6, 1]
        paddings = [0, 3, 6, 0]
        self.aspp = nn.ModuleList()
        for aspp_idx in range(len(kernel_sizes)):
            conv = torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_sizes[aspp_idx],
                stride=1,
                dilation=dilations[aspp_idx],
                padding=paddings[aspp_idx],
                bias=True)
            self.aspp.append(conv)
        self.gap = torch.nn.AdaptiveAvgPool2d(1) # 这个卷积之后，特征图大小默认就是1*1
        self.aspp_num = len(kernel_sizes)
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.fill_(0)

    def forward(self, x):
        # print("doing ASSP module!")
        avg_x = self.gap(x)
        out = []
        for aspp_idx in range(self.aspp_num):
            inp = avg_x if (aspp_idx == self.aspp_num - 1) else x
            out.append(F.relu_(self.aspp[aspp_idx](inp)))
        out[-1] = out[-1].expand_as(out[-2])
        out = torch.cat(out, dim=1)
        return out

class ConcatMerge(nn.Module):
    '''
    merge two features with the same size
    '''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_1 = nn.Conv2d(2 * in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x

class Conv2d_BN_Relu(nn.Module):
    """
    Conv2d_BN_Relu
    """
    def __init__(self, x1_channels, x2_channels, stride=1):
        super().__init__()

        self.conv_1 = nn.Conv2d(x2_channels, x2_channels, kernel_size=3, stride=stride, padding=1)
        self.conv_2 = nn.Conv2d(x1_channels + x2_channels, x1_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(x1_channels)

    def forward(self, x1, x2):
        x2 = self.conv_1(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv_2(x)
        x = self.bn(x)
        return F.relu(x)

class PANet(nn.Module):
    def __init__(self, feature_size=256):
        super(PANet, self).__init__()

        # upsample C5 to get P5
        # self.P5_1 = nn.Conv2d(C5_in_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.P5_3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4, decrease merging channels with a 3*3 convolution
        # self.P4_1 = nn.Conv2d(C4_in_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.P4_3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.P4_4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

        # add P4 elementwise to C3 and get downsample feature of merging fearture
        # self.P3_1 = nn.Conv2d(C3_in_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.P3_3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs
        # print("C5 : ", C5.shape)
        # P5_x = self.P5_1(C5)
        P5_x = C5
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        # P4_x = self.P4_1(C4)
        P4_x = C4
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        # P3_x = self.P3_1(C3)
        P3_x = C3
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P3_downsamples_x = self.P3_3(P3_x)
        P4_x = P4_x + P3_downsamples_x
        P4_x = self.P4_3(P4_x)

        P4_downsamples_x = self.P4_4(P4_x)
        P5_x =  P5_x + P4_downsamples_x
        P5_x = self.P5_3(P5_x)

        return [P3_x, P4_x, P5_x]

class CascadeFeaturesPyramid(nn.Module):
    def __init__(self, C3_in_size, C4_in_size, C5_in_size, cfp_step=2):
        super(CascadeFeaturesPyramid, self).__init__()

        self.feature_size = 256
        self.cfp_step = cfp_step
        self.assp_num = cfp_step - 1

        self.P3_1 = nn.Conv2d(C3_in_size, self.feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_1 = nn.Conv2d(C4_in_size, self.feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_1 = nn.Conv2d(C5_in_size, self.feature_size, kernel_size=1, stride=1, padding=0)

        self.PANet = PANet(self.feature_size)
        self.cfp_aspp = ASPP(self.feature_size, self.feature_size // 4)
        self.concatmerge = ConcatMerge(self.feature_size, self.feature_size)

        self.P6_1 = nn.Conv2d(2 * self.feature_size, self.feature_size, kernel_size=1, stride=1, padding=0)
        self.P6_2 = nn.Conv2d(self.feature_size, self.feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(self.feature_size, self.feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs
        C3 = self.P3_1(C3) # decrease channel to 256
        C4 = self.P4_1(C4)
        C5 = self.P5_1(C5)
        inputs = [C3, C4, C5]

        cfp_out = []
        global t_P3, t_P4, t_P5, P3, P4, P5
        for cfp_idx in range(self.cfp_step):
            PANet_out = self.PANet(inputs)
            t_P3, t_P4, t_P5 = PANet_out
            # print(str(cfp_idx) + ' ' + "PANet_out :", t_P3.shape)
            # print(str(cfp_idx) + ' ' + "PANet_out :", t_P4.shape)
            # print(str(cfp_idx) + ' ' + "PANet_out :", t_P5.shape)
            if self.assp_num > 0:
                for i in range(len(PANet_out)):
                    cfp_out.append(self.cfp_aspp(PANet_out[i]))
            else:
                cfp_out = PANet_out
            P3, P4, P5 = cfp_out
            if cfp_idx % 2 != 0:
                P3 = self.concatmerge(t_P3, P3)
                P4 = self.concatmerge(t_P4, P4)
                P5 = self.concatmerge(t_P5, P5)

            # print(str(cfp_idx) + ' ' + "P3 :", P3.shape)
            # print(str(cfp_idx) + ' ' + "P4 :", P4.shape)
            # print(str(cfp_idx) + ' ' + "P5 :", P5.shape)
            inputs = [P3, P4, P5]

            self.assp_num -= 1

        P6 = self.P6_2(self.P6_1(torch.cat([C5, P5], dim=1)))
        P7 = self.P7_2(self.P7_1(P6))

        return [P3, P4, P5, P6, P7]

class PANet1(nn.Module):
    def __init__(self, C3_in_size, C4_in_size, C5_in_size, feature_size=256):
        super(PANet1, self).__init__()

        # upsample C5 to get P5
        self.P5_1 = nn.Conv2d(C5_in_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.P5_3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4, decrease merging channels with a 3*3 convolution
        self.P4_1 = nn.Conv2d(C4_in_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.P4_3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.P4_4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

        # add P4 elementwise to C3 and get downsample feature of merging fearture
        self.P3_1 = nn.Conv2d(C3_in_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.P3_3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_in_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs
        # print("C5 : ", C5.shape)
        P5_x = self.P5_1(C5)
        # P5_x = C5
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        # P4_x = C4
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        # P3_x = C3
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P3_downsamples_x = self.P3_3(P3_x)
        P4_x = P4_x + P3_downsamples_x
        P4_x = self.P4_3(P4_x)

        P4_downsamples_x = self.P4_4(P4_x)
        P5_x =  P5_x + P4_downsamples_x
        P5_x = self.P5_3(P5_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]

class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out) # 这个地方将channel数降低至num_anchors*4

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)
        # view把tensor转成最后一个维度是4，contiguous（）是把tensor的内存连续起来，out.shape[0]是batchsize
        return out.contiguous().view(out.shape[0], -1, 4)

class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)


class ResNet(nn.Module):
    # model = ResNet(num_classes, Bottleneck, [3, 4, 6, 3], **kwargs)  # 对应resnet50
    def __init__(self, num_classes, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # channels : layer1 = 256, layer2 = 512, layer3 = 1024, layer4 = 2048
        self.layer1 = self._make_layer(block, 64, layers[0])  # 这个block是Bottleneck，utils.py下的一个类
        self.concat_merge1 = Conv2d_BN_Relu(x1_channels=256, x2_channels=64) # para1:64 + 256, para2:256
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.concat_merge2 = Conv2d_BN_Relu(x1_channels=512, x2_channels=256, stride=2) # para1:256 + 512, para2:512
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.concat_merge3 = Conv2d_BN_Relu(x1_channels=1024, x2_channels=512, stride=2) # para1:512 + 1024, para2:1024
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.concat_merge4 = Conv2d_BN_Relu(x1_channels=2048, x2_channels=1024, stride=2) # para1:1024 + 2048, para2:2048

        if block == BasicBlock:
            fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]
        elif block == Bottleneck:
            # 相当于取了每个layer部分的最后一层当作特征图size，构建特征金字塔
            fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels]
        else:
            raise ValueError(f"Block type {block} not understood")

        # add CA_SA_Model
        if block == BasicBlock:
            ca_sa_sizes = [self.layer1[layers[0] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]
        elif block == Bottleneck:
            # 相当于取了每个layer部分的最后一层当作特征图size，构建特征金字塔
            ca_sa_sizes = [self.layer1[layers[0] - 1].conv3.out_channels, self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels]
        else:
            raise ValueError(f"Block type {block} not understood")

        self.ca_sa = CA_SA_Model(ca_sa_sizes[0], ca_sa_sizes[1], ca_sa_sizes[2], ca_sa_sizes[3])

        # self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])
        # self.cfpn = CascadeFeaturesPyramid(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])
        self.panet = PANet1(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])

        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=num_classes)

        self.anchors = Anchors()

        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()

        self.focalLoss = losses.FocalLoss()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01

        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

        self.freeze_bn()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        # 在类ResNet中, self.inplanes = 64，block.eapansion = 4(这个是BottleNeck中，BasicBlock中为1)

        # 这个downsample作用是把特征图通道升维，然后短接到某个位置
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)] # 这个地方是BottleNeck类
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs):

        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs

        x = self.conv1(img_batch) # [1, 3, 640, 832]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x) # [8, 256, 72, 264]
        # print("x shape : ", x.shape)
        # print("x1 shape : ", x1.shape)

        csp_x1 = self.concat_merge1(x1, x)
        x2 = self.layer2(csp_x1) # [1, 512, 36, 132]
        # print("csp_x1 shape : ", csp_x1.shape)
        # print("x2 shape : ", x2.shape)
        csp_x2 = self.concat_merge2(x2, csp_x1)
        x3 = self.layer3(csp_x2) # [1, 1024, 18, 66]
        csp_x3 = self.concat_merge3(x3, csp_x2)
        x4 = self.layer4(csp_x3) # [1, 2048, 9, 33]
        csp_x4 = self.concat_merge4(x4, csp_x3)

        # add PFA struction
        ca_sa_out = self.ca_sa([csp_x1, csp_x2, csp_x3, csp_x4])

        # features = self.fpn([x2, x3, x4])
        # features = self.fpn(ca_sa_out)
        # features = self.cfpn(ca_sa_out)
        features = self.panet(ca_sa_out)

        #这个地方是分别对五个不同尺寸的特征图进行了回归和分类
        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)

        classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)

        anchors = self.anchors(img_batch)

        if self.training:
            return self.focalLoss(classification, regression, anchors, annotations)
        else:
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

            finalResult = [[], [], []]

            finalScores = torch.Tensor([])
            finalAnchorBoxesIndexes = torch.Tensor([]).long()
            finalAnchorBoxesCoordinates = torch.Tensor([])

            if torch.cuda.is_available():
                finalScores = finalScores.cuda()
                finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.cuda()
                finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates.cuda()

            for i in range(classification.shape[2]):
                scores = torch.squeeze(classification[:, :, i])
                scores_over_thresh = (scores > 0.05)
                if scores_over_thresh.sum() == 0:
                    # no boxes to NMS, just continue
                    continue

                scores = scores[scores_over_thresh]
                anchorBoxes = torch.squeeze(transformed_anchors)
                anchorBoxes = anchorBoxes[scores_over_thresh]
                anchors_nms_idx = nms(anchorBoxes, scores, 0.5)

                finalResult[0].extend(scores[anchors_nms_idx])
                finalResult[1].extend(torch.tensor([i] * anchors_nms_idx.shape[0]))
                finalResult[2].extend(anchorBoxes[anchors_nms_idx])

                finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
                finalAnchorBoxesIndexesValue = torch.tensor([i] * anchors_nms_idx.shape[0])
                if torch.cuda.is_available():
                    finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.cuda()

                finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
                finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))

            return [finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates]



def resnet18(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='.'), strict=False)
    return model


def resnet34(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir='.'), strict=False)
    return model


def resnet50(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='.'), strict=False)
    return model


def resnet101(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='.'), strict=False)
    return model


def resnet152(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir='.'), strict=False)
    return model
