import torch
from torch import nn
from timm.models.layers import trunc_normal_, DropPath

from algo_layer.models.model_utils import ConvMlp, sa_layer, spatial_attention_layer, channel_attention_layer
from algo_layer.models.model_utils import LayerNorm

from infrastructure_layer.basic_utils import count_vars_module


class MySimpleNet(nn.Module):
    def __init__(self, in_channels=3, classes=6, depths=None, dims=None, add_split_stem=True,
                 stem_dims=40, drop_path_rate=0., layer_scale_init_value=1e-6, head_init_scale=1.):
        super(MySimpleNet, self).__init__()

        self.add_att_layer = False

        if depths is None:
            depths = [2, 6, 2]
        if dims is None:
            dims = [stem_dims, 80, 160]

        self.depth_len = len(depths)
        # self.stem = SimpleStem(in_channels, dim=dims[0])
        if add_split_stem:
            self.stem = split_stem(in_channels, stem_dims=stem_dims, branch_ratio=0.25)
        else:
            self.stem = nn.Conv2d(in_channels, stem_dims, kernel_size=4, stride=4)
        '''
        self.incepres_1 = IncepResNet_unit(dims[0], dims[1], 1, add_att=add_att)
        self.incepres_2 = IncepResNet_unit(dims[1], dims[2], 1, add_att=add_att)
        self.incepres_3 = IncepResNet_unit(dims[2], dims[3], 1, add_att=add_att)
        if self.add_att_layer:
            self.att_layer = sa_layer(dims[-1])
        '''
        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(self.stem)
        for i in range(self.depth_len - 1):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(self.depth_len):
            stage = nn.Sequential(
                *[unit(dim=dims[i], drop_path=dp_rates[cur + j],
                       layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # self.conv = Conv2d(64, 256, 1, stride=1, padding=0, bias=False)
        # self.global_average_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], classes)

        self.apply(self._init_weights)
        # self.head.weight.data.mul_(head_init_scale)
        # self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(self.depth_len):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class SimpleStem(nn.Module):
    def __init__(self, in_channels, dim=32, ls_init_value=1e-6):
        super(SimpleStem, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=4, stride=4),
            LayerNorm(dim, eps=1e-6, data_format="channels_first")
            # Conv2d(32, 32, 3, stride=1, padding=0, bias=False),
            # Conv2d(32, 32, 3, stride=1, padding=1, bias=False),
            # nn.MaxPool2d(2, stride=1, padding=0),  # 73 x 73 x 64
        )

        self.branch_0 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)

        self.branch_1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2),
        )
        self.branch_2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3),
            # nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1),
        )
        self.branch_3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            # Conv2d(32, 32, 1, stride=1, padding=0, bias=False)
        )

        # self.norm = LayerNorm(dim * 3, data_format='channels_first')
        # self.mlp = ConvMlp(dim, int(4 * dim), act_layer=nn.GELU)
        # self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value else None

    def forward(self, x):
        shotcut = self.features(x)
        x0 = self.branch_0(shotcut)
        x1 = self.branch_1(shotcut)
        x2 = self.branch_2(shotcut)
        x3 = self.branch_3(shotcut)
        x = torch.cat((x0, x1, x2, x3), dim=1)
        # x = self.norm(x)
        # x = self.mlp(x)
        # if self.gamma is not None:
        #     x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        # x = x + shotcut
        return x


class split_stem(nn.Module):
    def __init__(self, in_channels, stem_dims, branch_ratio=0.125):
        super(split_stem, self).__init__()

        gc = int(stem_dims * branch_ratio)  # channel numbers of a convolution branch
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, stem_dims, kernel_size=4, stride=4, padding=0),  # pwconv
            LayerNorm(stem_dims, eps=1e-6, data_format="channels_first"),  # layernorm(chennel_first)
            # nn.Conv2d(stem_dims, stem_dims, kernel_size=1, stride=1, padding=0)  # pwconv
        )

        self.dwconv_1 = nn.Conv2d(gc, gc, 5, padding=2)  # , groups=gc
        self.dwconv_2 = nn.Conv2d(gc, gc, kernel_size=7, padding=3)
        # self.dwconv_2 = nn.MaxPool2d(3, stride=1)
        self.dwconv_3 = nn.Conv2d(gc, gc, kernel_size=11, padding=5)

        self.split_indexes = (stem_dims - 3 * gc, gc, gc, gc)
        # self.spa_att_layer = spatial_attention_layer(stem_dims)
        self.channel_att_layer = channel_attention_layer(stem_dims)
        # self.sa_layer = sa_layer(stem_dims)

    def forward(self, x):
        x = self.features(x)

        x_0, x_1, x_2, x_3 = torch.split(x, self.split_indexes, dim=1)
        stem_out = torch.cat(
            (x_0, self.dwconv_1(x_1), self.dwconv_2(x_2), self.dwconv_3(x_3)),
            dim=1,
        )
        # stem_out = self.sa_layer(stem)
        # 空间注意力
        # stem_out = self.spa_att_layer(stem_out)
        stem_out = self.channel_att_layer(stem_out)

        return stem_out


class IncepResNet_unit(nn.Module):
    def __init__(self, in_channels, out_channel=32, scale=1.0, add_att=False,
                 layer_scale_init_value=1e-6, drop_path=0.):
        super(IncepResNet_unit, self).__init__()
        self.scale = scale
        self.att = add_att
        if not self.att:
            self.branch_1 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 7, stride=1, padding=3),
                # nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.act = nn.GELU()  # nn.GELU()
        else:
            self.dwconv = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3,
                                    groups=in_channels)  # depthwise conv
            self.norm = LayerNorm(in_channels, eps=1e-6)
            self.pwconv1 = nn.Linear(in_channels,
                                     4 * in_channels)  # pointwise/1x1 convs, implemented with linear layers
            self.act = nn.GELU()
            self.pwconv2 = nn.Linear(4 * in_channels, in_channels)

            # 通道注意力
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(in_channels),
                                      requires_grad=True) if layer_scale_init_value > 0 else None
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        '''
        self.branch_2 = nn.Sequential(
            # Conv2d(in_channels, br_channel, 1, stride=1, padding=0, bias=False),
            Conv2d(in_channels, br_channel, 3, stride=1, padding=1),
            Conv2d(br_channel, br_channel, 3, stride=1, padding=1),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
        '''
        # self.relu = nn.ReLU(inplace=True)

        # 下采样层
        # self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.downsampler_layer = nn.Sequential(
            LayerNorm(in_channels, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(in_channels, out_channel, kernel_size=2, stride=2),
        )

    def forward(self, x):
        input = x
        if not self.att:
            x = self.branch_1(x)
            # x2 = self.branch_2(x)
            x = self.act(x)
            x = self.scale * x + input
            # x = self.max_pool(x)
        else:
            x = self.dwconv(x)
            x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
            x = self.norm(x)
            x = self.pwconv1(x)
            x = self.act(x)
            x = self.pwconv2(x)
            if self.gamma is not None:
                x = self.gamma * x
            x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
            x = self.drop_path(x) + input

        out = self.downsampler_layer(x)
        return out


class unit(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)

        return x


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding,
                 stride=1, bias=True, norm='gn'):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1)
        else:
            self.norm = nn.GroupNorm(32, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


if __name__ == "__main__":
    model = MySimpleNet(classes=6)
    # params = list(model.downsample_layers[1].parameters())
    params_num = count_vars_module(model, 110)
    print(params_num)
    print(model)
