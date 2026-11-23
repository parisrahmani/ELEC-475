import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from models.student import conv_bn_relu


def sep_conv_bn_relu(in_ch, out_ch, dilation=1):
    """Atrous separable conv: 3x3 depthwise (with dilation) + 1x1 pointwise,
    each followed by BatchNorm and ReLU. Keeps H and W the same."""
    return nn.Sequential(
        # depthwise: one 3x3 filter per input channel (groups=in_ch)
        nn.Conv2d(
            in_ch,
            in_ch,
            3,
            padding=dilation,
            dilation=dilation,
            groups=in_ch,
            bias=False,
        ),
        nn.BatchNorm2d(in_ch),
        nn.ReLU(inplace=True),
        # pointwise: 1x1 conv mixes the channels
        nn.Conv2d(in_ch, out_ch, 1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class SepASPP(nn.Module):
    """ASPP with atrous separable convolutions in place of full 3x3 convolutions."""

    def __init__(self, in_ch, out_ch=128, rates=(6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList(
            [conv_bn_relu(in_ch, out_ch, 1)]
            + [sep_conv_bn_relu(in_ch, out_ch, dilation=r) for r in rates]
        )
        self.project = conv_bn_relu(out_ch * (1 + len(rates)), out_ch, 1)

    def forward(self, x):
        return self.project(torch.cat([b(x) for b in self.branches], dim=1))


class MyStudentNet(nn.Module):
    """MobileNetV3-Small encoder, separable ASPP and a two-step decoder (mid, then low skip)."""

    def __init__(self, num_classes=21, pretrained=True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        features = mobilenet_v3_small(weights=weights).features
        # 24 channels, stride 8
        self.low = features[:4]
        # 40 channels, stride 16
        self.mid = features[4:7]
        # 576 channels, stride 32
        self.high = features[7:]
        self.aspp = SepASPP(576, 128)
        self.mid_proj = conv_bn_relu(40, 32, 1)
        self.fuse_mid = sep_conv_bn_relu(128 + 32, 96)
        self.low_proj = conv_bn_relu(24, 32, 1)
        self.fuse_low = sep_conv_bn_relu(96 + 32, 96)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Conv2d(96, num_classes, 1)

    def forward(self, x):
        size = x.shape[-2:]
        low = self.low(x)
        mid = self.mid(low)
        high = self.high(mid)
        y = self.aspp(high)
        # step 1: stride 32 -> 16, add mid-level detail
        y = F.interpolate(y, size=mid.shape[-2:], mode="bilinear", align_corners=False)
        y = self.fuse_mid(torch.cat([y, self.mid_proj(mid)], dim=1))
        # step 2: stride 16 -> 8, add low-level detail
        y = F.interpolate(y, size=low.shape[-2:], mode="bilinear", align_corners=False)
        y = self.fuse_low(torch.cat([y, self.low_proj(low)], dim=1))
        y = self.classifier(self.dropout(y))
        out = F.interpolate(y, size=size, mode="bilinear", align_corners=False)
        return {"out": out, "low": low, "mid": mid, "high": high}


if __name__ == "__main__":
    from torchinfo import summary
    from utils.profiler import ms_per_image

    model = MyStudentNet().eval()
    summary(model, input_size=(1, 3, 320, 320), depth=1)
    with torch.no_grad():
        out = model(torch.randn(1, 3, 320, 320))
    for k, v in out.items():
        print(f"{k:<5} {tuple(v.shape)}")
    print(f"ms/img (CPU): {ms_per_image(model, torch.device('cpu')):.1f}")
