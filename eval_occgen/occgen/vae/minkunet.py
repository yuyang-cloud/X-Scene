from typing import Sequence

import torch.nn as nn
import torchsparse
import torchsparse.nn as spnn
from mmengine.model import BaseModule
from torchsparse.nn.utils import fapply


class SyncBatchNorm(nn.SyncBatchNorm):

    def forward(self, input):
        return fapply(input, super().forward)


class BatchNorm(nn.BatchNorm1d):

    def forward(self, input):
        return fapply(input, super().forward)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 bn_momentum=0.1,
                 if_dist=False):
        super(BasicBlock, self).__init__()

        self.conv1 = spnn.Conv3d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            bias=False,
        )
        if if_dist:
            self.norm1 = SyncBatchNorm(planes, momentum=bn_momentum)
        else:
            self.norm1 = BatchNorm(planes, momentum=bn_momentum)
        self.conv2 = spnn.Conv3d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            dilation=dilation,
            bias=False,
        )
        if if_dist:
            self.norm2 = SyncBatchNorm(planes, momentum=bn_momentum)
        else:
            self.norm2 = BatchNorm(planes, momentum=bn_momentum)
        self.relu = spnn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 bn_momentum=0.1,
                 if_dist=False):
        super(Bottleneck, self).__init__()

        self.conv1 = spnn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        if if_dist:
            self.norm1 = SyncBatchNorm(planes, momentum=bn_momentum)
        else:
            self.norm1 = BatchNorm(planes, momentum=bn_momentum)

        self.conv2 = spnn.Conv3d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            bias=False,
        )
        if if_dist:
            self.norm2 = SyncBatchNorm(planes, momentum=bn_momentum)
        else:
            self.norm2 = BatchNorm(planes, momentum=bn_momentum)

        self.conv3 = spnn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        if if_dist:
            self.norm3 = SyncBatchNorm(planes * self.expansion, momentum=bn_momentum)
        else:
            self.norm3 = BatchNorm(planes * self.expansion, momentum=bn_momentum)

        self.relu = spnn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class MinkUNetBackbone(BaseModule):
    def __init__(self,
                #  in_channels: int = 4,
                 num_classes: int = 17,
                 base_channels: int = 32,
                 layers: Sequence[int] = [2, 3, 4, 6, 2, 2, 2, 2],
                 planes: Sequence[int] = [32, 64, 128, 256, 256, 128, 96, 96],
                 dilations: Sequence[int] = [1, 1, 1, 1, 1, 1, 1, 1],
                 block_type: str = 'basic',
                 bn_momentum: float = 0.05,
                 if_dist: bool = False,) -> None:
        super(MinkUNetBackbone, self).__init__(init_cfg=None)
        assert block_type in ['basic', 'bottleneck']
        if block_type == 'basic':
            block = BasicBlock
        else:
            block = Bottleneck

        self.inplanes = base_channels

        self.conv0p1s1 = spnn.Conv3d(
            num_classes,
            self.inplanes,
            kernel_size=3,
            stride=1,
            dilation=1,
            bias=False,
        )
        if if_dist:
            self.bn0 = SyncBatchNorm(self.inplanes, momentum=bn_momentum)
        else:
            self.bn0 = BatchNorm(self.inplanes, momentum=bn_momentum)

        self.conv1p1s2 = spnn.Conv3d(
            self.inplanes,
            self.inplanes,
            kernel_size=2,
            stride=2,
            dilation=1,
            bias=False,
        )
        if if_dist:
            self.bn1 = SyncBatchNorm(self.inplanes, momentum=bn_momentum)
        else:
            self.bn1 = BatchNorm(self.inplanes, momentum=bn_momentum)
        self.block1 = self._make_layer(
            block,
            planes[0],
            layers[0],
            dilation=dilations[0],
            bn_momentum=bn_momentum,
            if_dist=if_dist)

        self.conv2p2s2 = spnn.Conv3d(
            self.inplanes,
            self.inplanes,
            kernel_size=2,
            stride=2,
            dilation=1,
            bias=False,
        )
        if if_dist:
            self.bn2 = SyncBatchNorm(self.inplanes, momentum=bn_momentum)
        else:
            self.bn2 = BatchNorm(self.inplanes, momentum=bn_momentum)
        self.block2 = self._make_layer(
            block,
            planes[1],
            layers[1],
            dilation=dilations[1],
            bn_momentum=bn_momentum,
            if_dist=if_dist)

        self.conv3p4s2 = spnn.Conv3d(
            self.inplanes,
            self.inplanes,
            kernel_size=2,
            stride=2,
            dilation=1,
            bias=False,
        )
        if if_dist:
            self.bn3 = SyncBatchNorm(self.inplanes, momentum=bn_momentum)
        else:
            self.bn3 = BatchNorm(self.inplanes, momentum=bn_momentum)
        self.block3 = self._make_layer(
            block,
            planes[2],
            layers[2],
            dilation=dilations[2],
            bn_momentum=bn_momentum,
            if_dist=if_dist)

        self.conv4p8s2 = spnn.Conv3d(
            self.inplanes,
            self.inplanes,
            kernel_size=2,
            stride=2,
            dilation=1,
            bias=False,
        )
        if if_dist:
            self.bn4 = SyncBatchNorm(self.inplanes, momentum=bn_momentum)
        else:
            self.bn4 = BatchNorm(self.inplanes, momentum=bn_momentum)
        self.block4 = self._make_layer(
            block,
            planes[3],
            layers[3],
            dilation=dilations[3],
            bn_momentum=bn_momentum,
            if_dist=if_dist)

        self.convtr4p16s2 = spnn.Conv3d(
            self.inplanes,
            planes[4],
            kernel_size=2,
            stride=2,
            dilation=1,
            bias=False,
            transposed=True,
        )
        if if_dist:
            self.bntr4 = SyncBatchNorm(planes[4], momentum=bn_momentum)
        else:
            self.bntr4 = BatchNorm(planes[4], momentum=bn_momentum)
        self.inplanes = planes[4] + planes[2] * block.expansion
        self.block5 = self._make_layer(
            block,
            planes[4],
            layers[4],
            dilation=dilations[4],
            bn_momentum=bn_momentum,
            if_dist=if_dist)

        self.convtr5p8s2 = spnn.Conv3d(
            self.inplanes,
            planes[5],
            kernel_size=2,
            stride=2,
            dilation=1,
            bias=False,
            transposed=True,
        )
        if if_dist:
            self.bntr5 = SyncBatchNorm(planes[5], momentum=bn_momentum)
        else:
            self.bntr5 = BatchNorm(planes[5], momentum=bn_momentum)
        self.inplanes = planes[5] + planes[1] * block.expansion
        self.block6 = self._make_layer(
            block,
            planes[5],
            layers[5],
            dilation=dilations[5],
            bn_momentum=bn_momentum,
            if_dist=if_dist)
        
        self.convtr6p4s2 = spnn.Conv3d(
            self.inplanes,
            planes[6],
            kernel_size=2,
            stride=2,
            dilation=1,
            bias=False,
            transposed=True,
        )
        if if_dist:
            self.bntr6 = SyncBatchNorm(planes[6], momentum=bn_momentum)
        else:
            self.bntr6 = BatchNorm(planes[6], momentum=bn_momentum)
        self.inplanes = planes[6] + planes[0] * block.expansion
        self.block7 = self._make_layer(
            block,
            planes[6],
            layers[6],
            dilation=dilations[6],
            bn_momentum=bn_momentum,
            if_dist=if_dist)

        self.convtr7p2s2 = spnn.Conv3d(
            self.inplanes,
            planes[7],
            kernel_size=2,
            stride=2,
            dilation=1,
            bias=False,
            transposed=True,
        )
        if if_dist:
            self.bntr7 = SyncBatchNorm(planes[7], momentum=bn_momentum)
        else:
            self.bntr7 = BatchNorm(planes[7], momentum=bn_momentum)
        self.inplanes = planes[7] + base_channels
        self.block8 = self._make_layer(
            block,
            planes[7],
            layers[7],
            dilation=dilations[7],
            bn_momentum=bn_momentum,
            if_dist=if_dist)

        self.relu = spnn.ReLU(inplace=True)

        self.classifier = nn.Sequential(nn.Linear(planes[7], num_classes))

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self,
                    block: nn.Module,
                    planes: int,
                    blocks: int,
                    stride: int = 1,
                    dilation: int = 1,
                    bn_momentum: float = 0.1,
                    if_dist=False) -> nn.Module:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                spnn.Conv3d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                SyncBatchNorm(planes * block.expansion, momentum=bn_momentum)
                if if_dist else BatchNorm(planes * block.expansion, momentum=bn_momentum))
        layers = []

        layers.append(block(
            self.inplanes,
            planes,
            stride=stride,
            dilation=dilation,
            downsample=downsample,
            if_dist=if_dist))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(
                self.inplanes,
                planes,
                stride=1,
                dilation=dilation,
                if_dist=if_dist))

        return nn.Sequential(*layers)

    def forward(self, x):
        # voxel_features = feat_dict['voxels']
        # coors = feat_dict['coors']
        # x = torchsparse.SparseTensor(voxel_features, coors)

        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)

        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)

        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)

        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)

        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        encoder_out = self.block4(out)

        out = self.convtr4p16s2(encoder_out)
        out = self.bntr4(out)
        out = self.relu(out)
        out = torchsparse.cat((out, out_b3p8))
        out = self.block5(out)

        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)
        out = torchsparse.cat((out, out_b2p4))
        out = self.block6(out)

        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.relu(out)
        out = torchsparse.cat((out, out_b1p2))
        out = self.block7(out)

        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.relu(out)
        out = torchsparse.cat((out, out_p1))
        out = self.block8(out)

        out = self.classifier(out.F)

        return encoder_out, out