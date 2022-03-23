from models.modules.resnet_architecture.mobile_resnet_generator import MobileResnetGenerator
from models.networks import get_norm_layer
import torch
from torchviz import make_dot
from torchsummary import summary
norm = 'instance'
norm_layer = get_norm_layer(norm_type=norm)
#
# net = MobileResnetGenerator(3, 3, ngf=12, norm_layer=norm_layer,
#                             dropout_rate=0.1, n_blocks=9)
#
# summary(net,(3,255,255))


from models.modules.discriminators import NLayerDiscriminator

net = NLayerDiscriminator(3, 128, 3, norm_layer=norm_layer)

for n,m in net.named_modules():
    print("modele:"+n)

summary(net,(3,255,255))

def get_model_size(model):
    para_num = sum([p.numel() for p in model.parameters()])
    # para_size: 参数个数 * 每个4字节(float32) / 1024 / 1024，单位为 MB
    para_size = para_num * 4 / 1024 / 1024
    return para_size


# in_channels = [64, 128, 256, 256]*4
# print(in_channels)
# for idx, in_channel in enumerate(in_channels):
#     print(in_channel)

# print(get_model_size(net))


# x = torch.randn(1,3,256,156)
# y = net(x)
# g = make_dot(y)
# g.render('espnet_model', view=False)