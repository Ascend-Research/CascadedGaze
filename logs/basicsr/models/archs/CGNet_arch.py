# ------------------------------------------------------------------------
# Modified from NAFNet (https://github.com/megvii-research/NAFNet)
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.archs.arch_util import LayerNorm2d
from basicsr.models.archs.local_arch import Local_Base

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
    
class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size = 3, padding = 0, stide = 1, bias=False):
        super(depthwise_separable_conv, self).__init__()
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, stride=stide, padding=padding, groups=nin, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
    

class UpsampleWithFlops(nn.Upsample):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super(UpsampleWithFlops, self).__init__(size, scale_factor, mode, align_corners)
        self.__flops__ = 0

    def forward(self, input):
        self.__flops__ += input.numel()
        return super(UpsampleWithFlops, self).forward(input)


class GlobalContextExtractor(nn.Module):
    def __init__(self, c, kernel_sizes=[3, 3, 5], strides=[3, 3, 5], padding=0, bias=False):
        super(GlobalContextExtractor, self).__init__()

        self.depthwise_separable_convs = nn.ModuleList([
            depthwise_separable_conv(c, c, kernel_size, padding, stride, bias)
            for kernel_size, stride in zip(kernel_sizes, strides)
        ])

    def forward(self, x):
        outputs = []
        for conv in self.depthwise_separable_convs:
            x = F.gelu(conv(x))
            outputs.append(x)
        return outputs


class CascadedGazeBlock(nn.Module):
    def __init__(self, c, GCE_Conv =2, DW_Expand=2, FFN_Expand=2, drop_out_rate=0):
        super().__init__()
        self.dw_channel = c * DW_Expand
        self.GCE_Conv = GCE_Conv
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=self.dw_channel, kernel_size=1,
                                padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=self.dw_channel, out_channels=self.dw_channel,
                                kernel_size=3, padding=1, stride=1, groups=self.dw_channel,
                               bias=True)

        
        if self.GCE_Conv == 3:
            self.GCE = GlobalContextExtractor(c=c, kernel_sizes=[3, 3, 5], strides=[2, 3, 4])

            self.project_out = nn.Conv2d(int(self.dw_channel*2.5), c, kernel_size=1)

            self.sca = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels=int(self.dw_channel*2.5), out_channels=int(self.dw_channel*2.5), kernel_size=1, padding=0, stride=1,
                        groups=1, bias=True))
        else:
            self.GCE = GlobalContextExtractor(c=c, kernel_sizes=[3, 3], strides=[2, 3])

            self.project_out = nn.Conv2d(self.dw_channel*2, c, kernel_size=1)

            self.sca = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels=self.dw_channel*2, out_channels=self.dw_channel*2, kernel_size=1, padding=0, stride=1,
                        groups=1, bias=True))


        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        b,c,h,w = x.shape
        # # Nearest neighbor upsampling as part of the range fusion process
        self.upsample = UpsampleWithFlops(size=(h,w), mode='nearest')


        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.gelu(x)


        # Global Context Extractor + Range fusion
        x_1 , x_2 = x.chunk(2, dim=1)
        if self.GCE_Conv == 3:
            x1, x2, x3 = self.GCE(x_1 + x_2)
            x = torch.cat([x, self.upsample(x1), self.upsample(x2), self.upsample(x3)], dim = 1)
        else:
            x1, x2 = self.GCE(x_1 + x_2)
            x = torch.cat([x, self.upsample(x1), self.upsample(x2)], dim = 1)
        x = self.sca(x) * x
        x = self.project_out(x)


        x = self.dropout1(x)
        #channel-mixing
        y = inp + x * self.beta
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)

        return y + x * self.gamma

class NAFBlock0(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.0):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        #Channel Mixing
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma
    

class CascadedGaze(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], GCE_CONVS_nums=[]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        # for num in enc_blk_nums:
        for i in range(len(enc_blk_nums)):
            num = enc_blk_nums[i]
            GCE_Convs = GCE_CONVS_nums[i]
            self.encoders.append(
                nn.Sequential(
                    *[CascadedGazeBlock(chan, GCE_Conv=GCE_Convs) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock0(chan) for _ in range(middle_blk_num)]
            )

        for i in range(len(dec_blk_nums)):
            num = dec_blk_nums[i]
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock0(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x



if __name__ == '__main__':
    from ptflops import get_model_complexity_info
    from arch_util import measure_inference_speed

    
    img_channel = 3

    width = 60
    enc_blks = [2, 2, 4, 6]
    middle_blk_num = 10
    dec_blks = [2, 2, 2, 2]

    # width = 8
    # enc_blks = [1, 1, 1, 1]
    # middle_blk_num = 1
    # dec_blks = [1, 1, 1, 1]
    
    GCE_CONVS_nums = [3,3,2,2]


    net = CascadedGaze(img_channel=img_channel,width=width, middle_blk_num=middle_blk_num,
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks,GCE_CONVS_nums=GCE_CONVS_nums)


    inp_shape = (3, 256, 256)
    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)
    print(macs, params)
    
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    data = torch.randn((1, *inp_shape))
    print(device)
    measure_inference_speed(net.to(device), (data.to(device),), max_iter=500, log_interval=50)


