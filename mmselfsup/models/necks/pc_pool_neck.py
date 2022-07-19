# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch
from mmcv.runner import BaseModule

from ..builder import NECKS
import torch_scatter
from mmcv.cnn import build_norm_layer


@NECKS.register_module()
class PCPoolingNeck(BaseModule):
    """The average pooling 2d neck."""

    def __init__(self, in_channels, out_channels):
        super(PCPoolingNeck, self).__init__()
        self.projection_head = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, x, c_id):
        """Forward function."""
        uni_coords, value = torch.unique(c_id, return_inverse=True, dim=0)
        max_feat = torch_scatter.scatter_max(x, value, dim=0)[0]
        out = self.projection_head(max_feat)
        return out


@NECKS.register_module()
class NonLinearPCPoolingNeck(BaseModule):
    """The average pooling 2d neck. for simsiam"""

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 num_layers=2,
                 with_bias=False,
                 with_last_bn=True,
                 with_last_bn_affine=True,
                 with_last_bias=False,
                 with_avg_pool=True,
                 norm_cfg=dict(type='SyncBN'),
                 init_cfg=[
                     dict(
                         type='Constant',
                         val=1,
                         layer=['_BatchNorm', 'GroupNorm'])
                 ]):
        super(NonLinearPCPoolingNeck, self).__init__(init_cfg)
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.fc0 = nn.Linear(in_channels, hid_channels, bias=with_bias)
        self.bn0 = build_norm_layer(norm_cfg, hid_channels)[1]

        self.fc_names = []
        self.bn_names = []
        for i in range(1, num_layers):
            this_channels = out_channels if i == num_layers - 1 \
                else hid_channels
            if i != num_layers - 1:
                self.add_module(
                    f'fc{i}',
                    nn.Linear(hid_channels, this_channels, bias=with_bias))
                self.add_module(f'bn{i}',
                                build_norm_layer(norm_cfg, this_channels)[1])
                self.bn_names.append(f'bn{i}')
            else:
                self.add_module(
                    f'fc{i}',
                    nn.Linear(
                        hid_channels, this_channels, bias=with_last_bias))
                if with_last_bn:
                    self.add_module(
                        f'bn{i}',
                        build_norm_layer(
                            dict(**norm_cfg, affine=with_last_bn_affine),
                            this_channels)[1])
                    self.bn_names.append(f'bn{i}')
                else:
                    self.bn_names.append(None)
            self.fc_names.append(f'fc{i}')

    def forward(self, x, c_id=None):
        if isinstance(x, list):
            assert len(x) == 1
            x = x[0]
        # if self.with_avg_pool:
        #     uni_coords, value = torch.unique(c_id, return_inverse=True, dim=0)
        #     max_feat = torch_scatter.scatter_mean(x, value, dim=0)
        #     # Todo(yms) use max?
        #     # print(max_feat)
        #     x = max_feat
        x = x.view(x.size(0), -1)
        x = self.fc0(x)
        x = self.bn0(x)
        for fc_name, bn_name in zip(self.fc_names, self.bn_names):
            fc = getattr(self, fc_name)
            x = self.relu(x)
            x = fc(x)
            if bn_name is not None:
                bn = getattr(self, bn_name)
                x = bn(x)
        return [x]