import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, input_edge_size, method, BIL, caff, caff_enable, channel_attention,
                 channel_attention_enable):
        super(UpConvBlock, self).__init__()

        self.up_sampling = UDMBlock(in_channels, out_channels, method, channel_attention,
                                    channel_attention_enable)

        self.conv = ConvBlock(in_channels, out_channels)
        self.skip_connect = nn.Sequential()
        if BIL[0] != 0:
            self.skip_detach = True
        else:
            self.skip_detach = False
        for num in range(BIL[1]):
            self.skip_connect.add_module('conv' + str(num),
                                         nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
            self.skip_connect.add_module('BN' + str(num), nn.BatchNorm2d(out_channels))
            self.skip_connect.add_module('relu' + str(num), nn.ReLU(inplace=True))
        self.caff = get_caff(caff, caff_enable)

    def forward(self, x, skip_x):
        x = self.up_sampling(x)
        skip_x = self.skip_connect(skip_x.detach() if self.skip_detach else skip_x)
        x = torch.cat([x, skip_x], dim=1)
        if self.caff:
            x = self.caff(x)
        x = self.conv(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.K = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.Q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.V = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.dropout = nn.Dropout2d(p=0.1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x):
        batch_num, channels, height, width = x.size()
        k = self.K(x).view(batch_num, channels, -1)
        q = self.Q(x).view(batch_num, channels, -1)
        q = q.transpose(1, 2)
        v = self.V(x).view(batch_num, channels, -1)

        attn_weights = F.softmax(torch.bmm(q, k), dim=2)
        attn_weights = self.dropout(attn_weights)
        t = torch.bmm(v, attn_weights.transpose(1, 2))
        t = t.view(batch_num, channels, height, width)
        t = self.conv(t)

        return x + t


class FCAttentionBlock(nn.Module):
    def __init__(self, vector_dimension):
        super(FCAttentionBlock, self).__init__()
        self.vector_dimension = vector_dimension
        self.dropout = nn.Dropout2d(p=0.1)

    def forward(self, x):
        batch_num, channels, height, width = x.size()
        new_shape = (batch_num, height * width, channels, 1)

        input_channels = height * width
        output_channels = self.vector_dimension  # q、k向量的维度
        kernel_size = 1
        stride = 1
        padding = 0

        Q_weights = torch.empty((output_channels, input_channels, kernel_size, kernel_size))
        Q_bias = None
        K_weights = torch.empty((output_channels, input_channels, kernel_size, kernel_size))
        K_bias = None

        torch.nn.init.normal_(Q_weights)
        if Q_bias is not None:
            torch.nn.init.zeros_(Q_bias)
        torch.nn.init.normal_(K_weights)
        if K_bias is not None:
            torch.nn.init.zeros_(K_bias)

        original_x = x.clone()
        x = x.permute(0, 2, 3, 1)
        x = x.view(*new_shape)

        q = F.conv2d(x, Q_weights, bias=Q_bias, stride=stride, padding=padding)
        k = F.conv2d(x, K_weights, bias=K_bias, stride=stride, padding=padding)

        q = q.squeeze(-1)  # Remove last dimension
        k = k.squeeze(-1)
        q = q.permute(0, 2, 1)  # Swap second and third dimensions

        qk = torch.bmm(q, k)  # Matrix multiplication
        attn_weights = self.dropout(F.softmax(qk, dim=2))  # dropout
        v = torch.bmm(attn_weights, original_x.view(batch_num, channels, -1))
        v = v.view(batch_num, channels, height, width)

        return v


class ICAttentionBlock(nn.Module):
    def __init__(self):
        super(ICAttentionBlock, self).__init__()

    def forward(self, x):
        batch_num, channels, height, width = x.size()
        global_avg_pool = torch.mean(x, dim=[2, 3], keepdim=True)

        reshaped_tensor = global_avg_pool.view(batch_num, channels, 1)

        channel_attention = torch.matmul(reshaped_tensor, reshaped_tensor.transpose(1, 2))

        weighted_x = torch.matmul(reshaped_tensor.transpose(1, 2),
                                  channel_attention)  # (N, 1, C) x (N, C, C) -> (N, 1, C)

        softmax_x = torch.softmax(weighted_x.transpose(1, 2), dim=1).view(batch_num, channels, 1, 1)  # (N, C, 1, 1)

        result = x * softmax_x

        res_x = result + x

        return res_x


class UDMBlock(nn.Module):
    def __init__(self, in_channels, out_channels, method, channel_attention, channel_attention_enable):
        super(UDMBlock, self).__init__()
        self.method = method
        self.V = UDLayer(method=method, in_channels=in_channels, out_channels=out_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.channel_attention = get_channel_attention(channel_attention, channel_attention_enable)

    def forward(self, x):
        v = self.V(x)
        if self.method == 'DeepConv' or self.method == 'GroupConv':
            if self.method == 'GroupConv':
                v = self.conv(v)
            if self.channel_attention:
                v = self.channel_attention(v)
        return v


class UDLayer(nn.Module):
    def __init__(self, method, **kwargs):
        super(UDLayer, self).__init__()
        if method == 'MaxPool':
            self.du_sampling = nn.MaxPool2d(kernel_size=2, stride=2)
        elif method == 'AvgPool':
            self.du_sampling = nn.AvgPool2d(kernel_size=2, stride=2)
        elif method == 'ConvPool':
            in_channels = kwargs.get('in_channels')
            self.du_sampling = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2)
        elif method == 'DeepConv':
            in_channels = kwargs.get('in_channels')
            self.du_sampling = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2, groups=in_channels)
        elif method == 'GroupConv':
            in_channels = kwargs.get('in_channels')
            self.du_sampling = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2,
                                                  groups=in_channels)
        elif method == 'TransConv':
            in_channels = kwargs.get('in_channels')
            out_channels = kwargs.get('out_channels')
            self.du_sampling = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        else:
            raise ValueError("illegal sample method")

    def forward(self, x):
        x = self.du_sampling(x)
        return x


class PGCNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32, height=224,
                 up_sampling='TransConv', down_sampling='MaxPool', bil=None, caff=False, caff_enable=None,
                 channel_attention=None, channel_attention_enable=None, gradient_initial=None, gradm_stategylist=None):
        super(PGCNet, self).__init__()
        if gradient_initial is None:
            gradient_initial = [1, 1, 1]
        self.gm1 = GradientModifierModule(gradient_initial[0])
        self.gm2 = GradientModifierModule(gradient_initial[1])
        self.gm3 = GradientModifierModule(gradient_initial[2])
        self.gm_control = GradientModifierController([self.gm1, self.gm2, self.gm3], gradm_stategylist)

        features = init_features
        self.encoder1 = ConvBlock(in_channels, features)
        self.down_sampling1 = UDMBlock(features, features, method=down_sampling,
                                       channel_attention=channel_attention,
                                       channel_attention_enable=channel_attention_enable[0])

        self.encoder2 = ConvBlock(features, features * 2)
        self.down_sampling2 = UDMBlock(features * 2, features * 2,
                                       method=down_sampling,
                                       channel_attention=channel_attention,
                                       channel_attention_enable=channel_attention_enable[1])

        self.encoder3 = ConvBlock(features * 2, features * 4)
        self.down_sampling3 = UDMBlock(features * 4, features * 4,
                                       method=down_sampling, channel_attention=channel_attention,
                                       channel_attention_enable=channel_attention_enable[2])

        self.encoder4 = ConvBlock(features * 4, features * 8)

        self.attention = AttentionBlock(features * 8)

        self.decoder3 = UpConvBlock(features * 8, features * 4, height / 8,
                                    method=up_sampling, BIL=bil[2],
                                    caff=caff, caff_enable=caff_enable[0],
                                    channel_attention=channel_attention,
                                    channel_attention_enable=channel_attention_enable[3])

        self.decoder2 = UpConvBlock(features * 4, features * 2, height / 4,
                                    method=up_sampling, BIL=bil[1],
                                    caff=caff, caff_enable=caff_enable[1],
                                    channel_attention=channel_attention,
                                    channel_attention_enable=channel_attention_enable[4])

        self.decoder1 = UpConvBlock(features * 2, features, height / 2,
                                    method=up_sampling, BIL=bil[0],
                                    caff=caff, caff_enable=caff_enable[2],
                                    channel_attention=channel_attention,
                                    channel_attention_enable=channel_attention_enable[5])

        self.output = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        encoder1 = self.encoder1(x)
        down_sampling1 = self.down_sampling1(encoder1)

        encoder2 = self.encoder2(down_sampling1)
        down_sampling2 = self.down_sampling2(encoder2)

        encoder3 = self.encoder3(down_sampling2)
        down_sampling3 = self.down_sampling3(encoder3)

        encoder4 = self.encoder4(down_sampling3)

        center = self.attention(encoder4)

        decoder3 = self.decoder3(center, self.gm3(encoder3))

        decoder2 = self.decoder2(decoder3, self.gm2(encoder2))

        decoder1 = self.decoder1(decoder2, self.gm1(encoder1))

        output = self.output(decoder1)

        return output


def get_channel_attention(channel_attention: dict, channel_attention_enable: int):
    if channel_attention_enable == 0:
        return None
    if channel_attention['name'] == 'FC':
        module = FCAttentionBlock(**channel_attention['channel_attention_params'])
        return module
    elif channel_attention['name'] == 'ICA':
        pass
        # module = ICAttentionBlock(**channel_attention['channel_attention_params'])
        # return module
    elif channel_attention['name'] == 'SE':
        pass
        # module = SEAttentionBlock(**channel_attention['channel_attention_params'])
        # return module
    else:
        raise ValueError('Invalid channel_attention name: %s' % channel_attention['name'])


def get_caff(caff: dict, caff_enable: int):
    if caff_enable == 0:
        return None
    if caff['name'] == 'FC':
        module = FCAttentionBlock(**caff['channel_attention_params'])
        return module
    elif caff['name'] == 'ICA':
        pass
        module = ICAttentionBlock()
        return module
    elif caff['name'] == 'SE':
        pass
        # module = SEAttentionBlock(**caff['channel_attention_params'])
        # return module
    else:
        raise ValueError('Invalid channel_attention name: %s' % caff['name'])


def create_custom_function(weight_param):
    class GradientModifierFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            ctx.weight_param = weight_param
            return input.clone()

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output * ctx.weight_param
            return grad_input

    return GradientModifierFunction


class GradientModifierController():
    def __init__(self, module_list: list, updatestrategylist):
        self.module_list = module_list
        self.updatestrategylist = updatestrategylist

    def update(self, epoch=None):
        if self.updatestrategylist is not None and len(self.updatestrategylist) == len(self.module_list):
            if epoch is not None:
                for index, module in enumerate(self.module_list):
                    module.gradient_weight = self.updatestrategylist[index](epoch)
        return


class GradientModifierModule(nn.Module):
    def __init__(self, gradient_weight):
        super(GradientModifierModule, self).__init__()
        self.gradient_weight_init = gradient_weight
        self.gradient_weight = self.gradient_weight_init
        self.gmf = create_custom_function(self.gradient_weight)

    def forward(self, x):
        return self.gmf.apply(x)
