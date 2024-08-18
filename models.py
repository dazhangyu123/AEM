import timm
import torch.nn as nn
import torch
from timm.models.vision_transformer import VisionTransformer
from torchvision.models.resnet import Bottleneck
from transformers import CLIPModel, CLIPProcessor

from torchvision.models.resnet import BasicBlock, Bottleneck
import torch
from torch import nn as nn
from torch.utils import model_zoo
import open_clip
from timm.layers import SwiGLUPacked


import os
os.environ['http_proxy'] = 'http://10.13.118.13:7890'
os.environ['https_proxy'] = 'http://10.13.118.13:7890'


model_urls = {
'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class ResNet(nn.Module):
    def __init__(self, block, layers, classes=100):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.class_classifier = nn.Linear(512 * block.expansion, classes)
        self.pecent = 1/3

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.class_classifier(x)


def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model

def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model







class ResNetTrunk(ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.fc  # remove FC layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def get_pretrained_url(key):
    URL_PREFIX = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
    model_zoo_registry = {
        "BT": "bt_rn50_ep200.torch",
        "MoCoV2": "mocov2_rn50_ep200.torch",
        "SwAV": "swav_rn50_ep200.torch",
        "DINO_p16": "dino_vit_small_patch16_ep200.torch",
        "DINO_p8": "dino_vit_small_patch8_ep200.torch",
    }
    pretrained_url = f"{URL_PREFIX}/{model_zoo_registry.get(key)}"
    return pretrained_url


def resnet50(pretrained, progress, key, **kwargs):
    model = ResNetTrunk(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_url = get_pretrained_url(key)
        verbose = model.load_state_dict(
            torch.hub.load_state_dict_from_url(pretrained_url, progress=progress)
        )
        print(verbose)
    return model



def vit_small(pretrained, progress, key, **kwargs):
    patch_size = kwargs.get("patch_size", 16)
    model = VisionTransformer(
        img_size=224, patch_size=patch_size, embed_dim=384, num_heads=6, num_classes=0
    )
    if pretrained:
        pretrained_url = get_pretrained_url(key)
        verbose = model.load_state_dict(
            torch.hub.load_state_dict_from_url(pretrained_url, progress=progress)
        )
        print(verbose)
    return model

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class CustomModel(nn.Module):
    def __init__(self, cfg, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.embed_dim, cfg.n_class)
        # if cfg['dataset'] == 'tct':
        #     self.head = MLP(encoder.embed_dim, 2048, cfg['nb_classes'])
        # else:
        #     self.head = nn.Linear(encoder.embed_dim, cfg['nb_classes'])

    def forward(self, image, return_feature=False):
        image_features = self.encoder(image)
        logits = self.head(image_features)
        if return_feature:
            return logits, image_features
        return logits



def build_model(cfg):
    if cfg.pretrain == 'natural_supervised':
        encoder = resnet18()
        encoder.class_classifier = nn.Identity()
        encoder.embed_dim = encoder.inplanes
    elif cfg.pretrain == 'medical_ssl':
        encoder = vit_small(pretrained=True, progress=False, key="DINO_p16", patch_size=16)
    elif cfg.pretrain == 'UNI':
        encoder = timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True, pretrained=False
        )
        encoder.load_state_dict(torch.load("/mnt/Xsky/syx/project/model/UNI/pytorch_model.bin", map_location="cpu"), strict=True)
    elif cfg.pretrain == 'GigaPath':
        encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
    elif cfg.pretrain == 'virchow':
        encoder = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)

    return encoder

def build_model_vlm(cfg):
    if cfg.pretrain == 'plip':
        encoder = CLIPModel.from_pretrained("/mnt/Xsky/syx/model/clip-based/PLIP")
    elif cfg.pretrain == 'quilt-net':
        encoder = CLIPModel.from_pretrained("/mnt/Xsky/syx/model/clip-based/Quilt-Net")
    elif cfg.pretrain == 'biomedclip':
        encoder, preprocess = open_clip.create_model_from_pretrained(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
            cache_dir='/mnt/Xsky/syx/model/clip/hub/models')
    else:
        if cfg.pretrain == 'path-clip-B':
            pretrained = '/mnt/Xsky/syx/model/clip/exp/ViT-B-16/0526/ori_pretrained/ratiotmp6/checkpoints/epoch_2.pt'  # ViT-B-16
            model_name = 'ViT-B-16'
        elif cfg.pretrain == 'path-clip-B-AAAI':
            pretrained = '/mnt/Xsky/syx/model/clip/exp/ViT-B-16/epoch_3_best1.pt'  # ViT-B-16
            model_name = 'ViT-B-16'
        elif cfg.pretrain == 'path-clip-L-336':
            pretrained = '/mnt/Xsky/syx/model/clip/exp/ViT-L-14-336/0522/ori_pretrained/ratiotmp5/checkpoints/epoch_2.pt'  #ViT-L-14-336
            model_name = 'ViT-L-14-336'
        elif cfg.pretrain == 'openai-clip-B':
            pretrained = '/mnt/Xsky/syx/clip_cached_model/clip/ViT-B-16.pt'
            model_name = 'ViT-B-16'
        elif cfg.pretrain == 'openai-clip-L-336':
            pretrained = '/mnt/Xsky/syx/clip_cached_model/clip/ViT-L-14-336px.pt'
            model_name = 'ViT-L-14-336'
        elif cfg.pretrain == 'path-clip-L-768':
            pretrained = '/mnt/Xsky/syx/model/clip/exp/ViT-L-32-768/240616/ori/ratiotmp5/checkpoints/epoch_2.pt'
            # although 'path-clip-L-768' also uses model_name of 'ViT-L-14-336', this model is
            # self-define in src.openclip fold. hence, use this model please turn import open_clip as
            # import src.open_clip as open_clip
            model_name = 'ViT-L-14-336'


        encoder, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained,
                                                                     cache_dir='/mnt/Xsky/syx/model/open_clip',
                                                                     force_quick_gelu=True)


    return encoder

