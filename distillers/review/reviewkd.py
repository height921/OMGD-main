# from models import networks
# from .shufflenetv1 import ShuffleV1
# from .shufflenetv2 import ShuffleV2
import torch
from torch import nn
import torch.nn.functional as F


# from .resnet  import *


class ABF(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, fuse):
        super(ABF, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        if fuse:
            self.att_conv = nn.Sequential(
                nn.Conv2d(mid_channel * 2, 2, kernel_size=1),
                nn.Sigmoid(),
            )
        else:
            self.att_conv = None
        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)  # pyre-ignore
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)  # pyre-ignore

    def forward(self, x, y=None, shape=None, out_shape=None):
        n, _, h, w = x.shape
        # transform student features
        x = self.conv1(x)
        if self.att_conv is not None:
            # upsample residual features
            y = F.interpolate(y, (shape, shape), mode="nearest")
            # fusion
            z = torch.cat([x, y], dim=1)
            z = self.att_conv(z)
            x = (x * z[:, 0].view(n, 1, h, w) + y * z[:, 1].view(n, 1, h, w))
        # output 
        if x.shape[-1] != out_shape:
            x = F.interpolate(x, (out_shape, out_shape), mode="nearest")
        y = self.conv2(x)
        return y, x


# class ReviewKD():
#     def __init__(
#             self, Sacts, Tacts, in_channels, out_channels, shapes, out_shapes, device='cpu'
#     ):
#         super(ReviewKD, self).__init__()
#         self.Sacts = Sacts
#
#         self.shapes = shapes
#         self.out_shapes = shapes if out_shapes is None else out_shapes
#
#         abfs = nn.ModuleList()
#
#         mid_channel = min(512, in_channels[-1])
#         for idx, in_channel in enumerate(in_channels):
#             abfs.append(ABF(in_channel, mid_channel, out_channels[idx], idx < len(in_channels) - 1))
#         self.abfs = abfs[::-1]
#         self.to(device)
#
#     def forward(self, x):
#         x = self.Sacts[::-1]
#         results = []
#         out_features, res_features = self.abfs[0](x[0], out_shape=self.out_shapes[0])
#         results.append(out_features)
#         for features, abf, shape, out_shape in zip(x[1:], self.abfs[1:], self.shapes[1:], self.out_shapes[1:]):
#             out_features, res_features = abf(features, res_features, shape, out_shape)
#             results.insert(0, out_features)
#
#         return results


# def build_review_kd(model, opt, device):
#     out_shapes = None
#     if 'mobile_resnet_9blocks' in model:
#         student = networks.define_G(opt.input_nc, opt.output_nc, opt.student_ngf,
#                                     opt.student_netG, opt.norm, opt.student_dropout_rate,
#                                     opt.init_type, opt.init_gain, opt.gpu_ids, opt=opt)
#         in_channels = [64, 128, 256, 256]
#         out_channels = [64, 128, 256, 256]
#         shapes = [1, 8, 16, 32]
#     else:
#         assert False
#     backbone = ReviewKD(
#         Sacts=Sacts,
#         in_channels=in_channels,
#         out_channels=out_channels,
#         shapes=shapes,
#         out_shapes=out_shapes,
#         device=device
#     )
#     return backbone


def hcl(fstudent, fteacher):
    loss_all = 0.0
    for fs, ft in zip(fstudent, fteacher):
        n, c, h, w = fs.shape
        loss = F.mse_loss(fs, ft, reduction='mean')
        cnt = 1.0
        tot = 1.0
        for l in [4, 2, 1]:
            if l >= h:
                continue
            tmpfs = F.adaptive_avg_pool2d(fs, (l, l))
            tmpft = F.adaptive_avg_pool2d(ft, (l, l))
            cnt /= 2.0
            loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
            tot += cnt
        loss = loss / tot
        loss_all = loss_all + loss
    return loss_all


if __name__ == '__main__':
    x = torch.ones([2, 64, 64, 64])
    y = torch.ones([2, 64, 64, 64])
    # abf = ABF(48, 64, 64, True)
    # out = abf(x, y, 64, 64)
    hcl(x,y)