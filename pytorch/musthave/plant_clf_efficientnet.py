seed = 50
import os
os.environ["PYTHONHASHEED"] = str(seed)

import random
import numpy as np
import pandas as pd
import skimage
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

os.environ['ALBUMENTATIONS_DISABLE_VERSION_CHECK'] = '1'
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from efficientnet_pytorch import EfficientNet

from trainer import Trainer, EarlyStopper


class ImageDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, is_test=False):
        super().__init__()
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx, 0]
        img_path = self.img_dir + img_id + ".jpg"
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.df.iloc[idx, 1]

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        if self.is_test:
            return image
        else:
            label = np.argmax(self.df.iloc[idx, 1:5])
            return image, label

def accuracy(y_pred, y_true):
    """ Multi-class classification (y_pred: logtis without softmax) """
    y_pred = y_pred.argmax(dim=1)                   ## int64 (long)
    return torch.eq(y_pred, y_true).float().mean()
    

if __name__ == "__main__":

    #################################################################
    ## Random Seed
    #################################################################
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
             
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enable = False

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    
    #################################################################
    ## Dataset
    #################################################################

    data_path = '/home/namu/myspace/data/plant-pathology-2020-fgvc7/'
    train = pd.read_csv(data_path + "train.csv")
    test = pd.read_csv(data_path + "test.csv")
    submission = pd.read_csv(data_path + "sample_submission.csv")
    label = ["healthy", "multiple_diseases", "rust", "scab"]

    train, valid = train_test_split(train, test_size=0.1,
                                stratify=train[label],
                                random_state=50)

    transform_train = A.Compose([
        A.Resize(450, 650),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.VerticalFlip(p=0.2),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.7, rotate_limit=30, p=0.3),
        A.OneOf([A.Emboss(p=1), A.Sharpen(p=1), A.Blur(p=1)], p=0.3),
        # A.PiecewiseAffine(p=0.3),
        A.Normalize(),
        ToTensorV2(),])

    transform_test = A.Compose([
        A.Resize(450, 650),
        A.Normalize(),
        ToTensorV2(),])

    img_dir = '/home/namu/myspace/data/plant-pathology-2020-fgvc7/images/'
    dataset_train = ImageDataset(train, img_dir=img_dir, transform=transform_train)
    dataset_valid = ImageDataset(valid, img_dir=img_dir, transform=transform_test)
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)
    batch_size = 4

    batch_size = 4
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                         worker_init_fn=seed_worker, generator=g, num_workers=4)
    loader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False,
                         worker_init_fn=seed_worker, generator=g, num_workers=4)
    
    # kwargs = {"num_workers": 2, "pin_memory": True} if use_cuda else {}
    # loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
    #                          **kwargs)
    # loader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False,
    #                          **kwargs)

    #################################################################
    ## Modeing
    #################################################################

    model = EfficientNet.from_name("efficientnet-b7")
    model.load_state_dict(torch.load("efficientnet-b7-dcc49843.pth"))
    model._fc = nn.Sequential(
        nn.Linear(model._fc.in_features, model._fc.out_features),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(model._fc.out_features, 4),)
    model = model.to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00006, weight_decay=0.0001)
    metrics={"acc": accuracy}
    
    clf = Trainer(model, optimizer, loss_fn, metrics=metrics)
    hist = clf.fit(loader_train, n_epochs=3, valid_loader=loader_valid,
                  early_stopper=EarlyStopper(patience=5, min_delta=0.001))

