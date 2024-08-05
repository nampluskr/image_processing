import os
from glob import glob
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

from efficientnet_pytorch import EfficientNet

from trainer import Trainer, EarlyStopper


class ImageDataset(Dataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, 0]
        # image = skimage.io.imread(img_path)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.df.iloc[idx, 1]

        if self.transform is not None:
            image = self.transform(image)

        return image, label



if __name__ == "__main__":

    #################################################################
    ## Random Seed
    #################################################################

    seed = 50
    os.environ["PYTHONHASHEED"] = str(seed)
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
    
    data_path = '/home/namu/myspace/data/chest-xray-pneumonia/chest_xray/'
    train_path = data_path + "train/"
    valid_path = data_path + "val/"
    test_path = data_path + "test/"
    
    print(f">> train: {len(glob(train_path + '*/*'))}")
    print(f">> valid: {len(glob(valid_path + '*/*'))}")
    print(f">> test : {len(glob(test_path + '*/*'))}")

    train_df_0 = pd.DataFrame({"img_path": glob(train_path + "NORMAL/*"), "label": 0 })
    train_df_1 = pd.DataFrame({"img_path": glob(train_path + "PNEUMONIA/*"), "label": 1 })
    train_df = pd.concat([train_df_0, train_df_1], ignore_index=True)
    train_df = train_df.sample(frac=1).reset_index(drop=True)

    valid_df_0 = pd.DataFrame({"img_path": glob(valid_path + "NORMAL/*"), "label": 0 })
    valid_df_1 = pd.DataFrame({"img_path": glob(valid_path + "PNEUMONIA/*"), "label": 1 })
    valid_df = pd.concat([valid_df_0, valid_df_1], ignore_index=True)
    valid_df = valid_df.sample(frac=1).reset_index(drop=True)

    test_df_0 = pd.DataFrame({"img_path": glob(test_path + "NORMAL/*"), "label": 0 })
    test_df_1 = pd.DataFrame({"img_path": glob(test_path + "PNEUMONIA/*"), "label": 1 })
    test_df = pd.concat([test_df_0, test_df_1], ignore_index=True)
    test_df = test_df.sample(frac=1).reset_index(drop=True)


    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((250, 250)),
        transforms.CenterCrop(180),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.2),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((250, 250)),
        transforms.CenterCrop(180),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    dataset_train = ImageDataset(train_df, transform=transform_train)
    dataset_valid = ImageDataset(valid_df, transform=transform_test)

    #################################################################
    ## Data Loader
    #################################################################
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    g = torch.Generator()
    g.manual_seed(0)
    batch_size = 8
    
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                         worker_init_fn=seed_worker, generator=g, num_workers=4)
    loader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False,
                         worker_init_fn=seed_worker, generator=g, num_workers=4)

    #################################################################
    ## Modeing
    #################################################################

    model = EfficientNet.from_name("efficientnet-b0")
    model_dir = "/home/namu/myspace/pytorch_models/"
    model_weight = "efficientnet-b0-355c32eb.pth"
    model.load_state_dict(torch.load(model_dir + model_weight))
    model._fc = nn.Sequential(
        nn.Linear(model._fc.in_features, model._fc.out_features),
        nn.ReLU(),
        nn.Linear(model._fc.out_features, 2),
    )

    def accuracy(y_pred, y_true):
        """ Multi-class classification (y_pred: logtis without softmax) """
        y_pred = y_pred.argmax(dim=1)                   ## int64 (long)
        return torch.eq(y_pred, y_true).float().mean()
    
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    metrics={"acc": accuracy}

    #################################################################
    ## Training
    #################################################################
    
    clf = Trainer(model, optimizer, loss_fn, metrics=metrics)
    hist = clf.fit(loader_train, n_epochs=50, valid_loader=loader_valid,
                  early_stopper=EarlyStopper(patience=5, min_delta=0.001))
