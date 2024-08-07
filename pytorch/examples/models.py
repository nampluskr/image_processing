import torch
import torchvision
import torch.nn as nn
from torchvision import models


def get_resnet50(out_features, freezed=False):

    model = torchvision.models.resnet50()
    model_weights = "/home/namu/myspace/pytorch_models/resnet50-11ad3fa6.pth"
    model.load_state_dict(torch.load(model_weights))

    if freezed:
        for param in model.parameters():
            param.requiers_grad = False
    
    model.fc = nn.Linear(2048, out_features)
    return model


def get_efficientnet(name, out_features, freezed=False):
    # https://pytorch.org/vision/stable/models/efficientnet.html

    model_dir = "/home/namu/myspace/pytorch_models/"
    models = {
        "b0": torchvision.models.efficientnet_b0(),
        "b1": torchvision.models.efficientnet_b1(),
        "b2": torchvision.models.efficientnet_b2(),
        "b3": torchvision.models.efficientnet_b3(),
        "b7": torchvision.models.efficientnet_b7(),
    }
    model_weights = {
        "b0": "efficientnet_b0_rwightman-7f5810bc.pth",
        "b1": "efficientnet_b1-c27df63c.pth",
        "b2": "efficientnet_b2_rwightman-c35c1473.pth",
        "b3": "efficientnet_b3_rwightman-b3899882.pth",
        "b7": "efficientnet_b7_lukemelas-c5b4e57e.pth",
    }
    model = models[name]
    model.load_state_dict(torch.load(model_dir + model_weights[name]))

    if freezed:
        for param in model.parameters():
            param.requiers_grad = False
    
    classifiers = {
        "b0": nn.Sequential(nn.Dropout(0.2, inplace=True), nn.Linear(1280, out_features)),
        "b1": nn.Sequential(nn.Dropout(0.2, inplace=True), nn.Linear(1280, out_features)),
        "b2": nn.Sequential(nn.Dropout(0.3, inplace=True), nn.Linear(1408, out_features)),
        "b3": nn.Sequential(nn.Dropout(0.3, inplace=True), nn.Linear(1536, out_features)),
        "b7": nn.Sequential(nn.Dropout(0.5, inplace=True), nn.Linear(2560, out_features)),
    }
    model.classifier = classifiers[name]
    return model


if __name__ == "__main__":

    pass
