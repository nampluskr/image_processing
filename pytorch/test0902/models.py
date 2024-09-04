import torch
import torch.nn as nn
from torchvision.models import vgg16_bn, vgg19_bn
from torchvision.models import resnet34, resnet50
from torchvision.models import efficientnet_b0, efficientnet_b1, efficientnet_b2
from torchvision.models import efficientnet_b3, efficientnet_b7


class EncoderVgg16(nn.Module):
    def __init__(self, latent_dim, freezed=False):
        super().__init__()
        self.model = vgg16_bn()
        model_dir = "/home/namu/myspace/NAMU/pytorch/models/"
        self.model.load_state_dict(torch.load(model_dir + "vgg16_bn-6c64b313.pth"))

        if freezed:
            for param in self.model.parameters():
                param.requiers_grad = False

        self.model.classifier = nn.Linear(in_features=25088, out_features=latent_dim)

    def forward(self, x):
        x = self.model(x)
        return x


class EncoderVgg19(nn.Module):
    def __init__(self, latent_dim, freezed=False):
        super().__init__()
        self.model = vgg19_bn()
        model_dir = "/home/namu/myspace/NAMU/pytorch/models/"
        self.model.load_state_dict(torch.load(model_dir + "vgg19_bn-c79401a0.pth"))

        if freezed:
            for param in self.model.parameters():
                param.requiers_grad = False

        self.model.classifier = nn.Linear(in_features=25088, out_features=latent_dim)

    def forward(self, x):
        x = self.model(x)
        return x


class EncoderResnet34(nn.Module):
    def __init__(self, latent_dim, freezed=False):
        super().__init__()
        self.model = resnet34()
        model_dir = "/home/namu/myspace/NAMU/pytorch/models/"
        self.model.load_state_dict(torch.load(model_dir + "resnet34-b627a593.pth"))

        if freezed:
            for param in self.model.parameters():
                param.requiers_grad = False

        self.model.fc = nn.Linear(512, latent_dim)

    def forward(self, x):
        x = self.model(x)
        return x


class EncoderResnet50(nn.Module):
    def __init__(self, latent_dim, freezed=False):
        super().__init__()
        self.model = resnet50()
        model_dir = "/home/namu/myspace/NAMU/pytorch/models/"
        self.model.load_state_dict(torch.load(model_dir + "resnet50-11ad3fa6.pth"))

        if freezed:
            for param in self.model.parameters():
                param.requiers_grad = False

        self.model.fc = nn.Linear(2048, latent_dim)

    def forward(self, x):
        x = self.model(x)
        return x


class EncoderEfficientNetB0(nn.Module):
    def __init__(self, latent_dim, freezed=False):
        super().__init__()
        self.model = efficientnet_b0()
        model_dir = "/home/namu/myspace/NAMU/pytorch/models/"
        self.model.load_state_dict(torch.load(model_dir + "efficientnet_b0_rwightman-7f5810bc.pth"))

        if freezed:
            for param in self.model.parameters():
                param.requiers_grad = False

        self.model.classifier = nn.Linear(1280, latent_dim)

    def forward(self, x):
        x = self.model(x)
        return x


class EncoderEfficientNetB1(nn.Module):
    def __init__(self, latent_dim, freezed=False):
        super().__init__()
        self.model = efficientnet_b1()
        model_dir = "/home/namu/myspace/NAMU/pytorch/models/"
        self.model.load_state_dict(torch.load(model_dir + "efficientnet_b1-c27df63c.pth"))

        if freezed:
            for param in self.model.parameters():
                param.requiers_grad = False

        self.model.classifier = nn.Linear(1280, latent_dim)

    def forward(self, x):
        x = self.model(x)
        return x


class EncoderEfficientNetB2(nn.Module):
    def __init__(self, latent_dim, freezed=False):
        super().__init__()
        self.model = efficientnet_b2()
        model_dir = "/home/namu/myspace/NAMU/pytorch/models/"
        self.model.load_state_dict(torch.load(model_dir + "efficientnet_b2_rwightman-c35c1473.pth"))

        if freezed:
            for param in self.model.parameters():
                param.requiers_grad = False

        self.model.classifier = nn.Linear(1408, latent_dim)

    def forward(self, x):
        x = self.model(x)
        return x


class EncoderEfficientNetB3(nn.Module):
    def __init__(self, latent_dim, freezed=False):
        super().__init__()
        self.model = efficientnet_b3()
        model_dir = "/home/namu/myspace/NAMU/pytorch/models/"
        self.model.load_state_dict(torch.load(model_dir + "efficientnet_b3_rwightman-b3899882.pth"))

        if freezed:
            for param in self.model.parameters():
                param.requiers_grad = False

        self.model.classifier = nn.Linear(1536, latent_dim)

    def forward(self, x):
        x = self.model(x)
        return x


class EncoderEfficientNetB7(nn.Module):
    def __init__(self, latent_dim, freezed=False):
        super().__init__()
        self.model = efficientnet_b7()
        model_dir = "/home/namu/myspace/NAMU/pytorch/models/"
        self.model.load_state_dict(torch.load(model_dir + "efficientnet_b7_lukemelas-c5b4e57e.pth"))

        if freezed:
            for param in self.model.parameters():
                param.requiers_grad = False

        self.model.classifier = nn.Linear(2560, latent_dim)

    def forward(self, x):
        x = self.model(x)
        return x
